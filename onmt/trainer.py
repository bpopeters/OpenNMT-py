"""
    This is the loadable seq2seq trainer library that is
    in charge of training details, loss compute, and statistics.
    See train.py for a use case of this library.

    Note: To make this a general library, we implement *only*
          mechanism things here(i.e. what to do), and leave the strategy
          things to users(i.e. how to do it). Also see train.py (one of the
          users of this library) for the strategy things we do.
"""
import torch
import torch.nn as nn

import onmt.inputters as inputters
import onmt.utils

from onmt.utils import Statistics
from onmt.utils.logging import logger


def build_trainer(opt, model_opt, model, fields, optim, data_type):
    """
    Simplify `Trainer` creation based on user `opt`s*

    Args:
        opt (:obj:`Namespace`): user options (usually from argument parsing)
        model (:obj:`onmt.models.NMTModel`): the model to train
        fields (dict): dict of fields
        optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
        data_type (str): string describing the type of data
            e.g. "text", "img", "audio"
    """
    train_loss = onmt.utils.loss.build_loss_compute(
        model, fields["tgt"].vocab, opt)
    valid_loss = onmt.utils.loss.build_loss_compute(
        model, fields["tgt"].vocab, opt, train=False)

    trunc_size = opt.truncated_decoder
    shard_size = opt.max_generator_batches
    norm_method = opt.normalization
    grad_accum_count = opt.accum_count
    n_gpu = len(opt.gpuid)
    gpu_rank = opt.gpu_rank
    gpu_verbose_level = opt.gpu_verbose_level
    report_every = opt.report_every
    save_checkpoint_steps = opt.save_checkpoint_steps
    keep_checkpoint = opt.keep_checkpoint
    save_model = opt.save_model

    trainer = onmt.Trainer(model, model_opt, fields, train_loss, valid_loss,
                           optim, trunc_size, shard_size, data_type,
                           norm_method, grad_accum_count, n_gpu, gpu_rank,
                           gpu_verbose_level, report_every,
                           save_checkpoint_steps, keep_checkpoint, save_model)
    return trainer


class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.models.model.NMTModel`): translation model
                to train
            train_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.utils.optimizers.Optimizer`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            grad_accum_count(int): accumulate gradients this many times.
            model_saver(:obj:`onmt.models.ModelSaverBase`): the saver is
                used to save a checkpoint.
                Thus nothing will be saved if this parameter is None
    """

    def __init__(self, model, model_opt, fields, train_loss, valid_loss, optim,
                 trunc_size=0, shard_size=32, data_type='text',
                 norm_method="sents", grad_accum_count=1, n_gpu=1, gpu_rank=1,
                 gpu_verbose_level=0, report_every=50,
                 save_checkpoint_steps=5000, keep_checkpoint=-1,
                 save_model='model'):
        self.model = model
        self.model_opt = model_opt
        self.fields = fields
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.optim = optim
        self.trunc_size = trunc_size
        self.shard_size = shard_size
        self.data_type = data_type
        self.norm_method = norm_method
        self.grad_accum_count = grad_accum_count
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.gpu_verbose_level = gpu_verbose_level
        self.report_every = report_every
        self.save_checkpoint_steps = save_checkpoint_steps
        self.keep_checkpoint = keep_checkpoint
        self.base_path = save_model

        assert grad_accum_count > 0
        assert grad_accum_count == 1 or self.trunc_size == 0, \
            "To enable accumulated gradients, you must disable truncated BPTT."

        # Set model in training mode.
        self.model.train()

    @property
    def current_step(self):
        return self.optim._step + 1

    @property
    def learning_rate(self):
        return self.optim.learning_rate

    def _time_to_report(self):
        return self.current_step % self.report_every == 0

    def _time_to_save(self):
        return self.gpu_rank == 0 and self.keep_checkpoint != 0 and \
            self.current_step % self.save_checkpoint_steps == 0

    def train(self, train_iter_fct, valid_iter_fct, train_steps, valid_steps):
        """
        The main training loops.
        by iterating over training data (i.e. `train_iter_fct`)
        and running validation (i.e. iterating over `valid_iter_fct`

        Args:
            train_iter_fct(function): a function that returns the train
                iterator. e.g. something like
                train_iter_fct = lambda: generator(*args, **kwargs)
            valid_iter_fct(function): same as train_iter_fct, for valid data
            train_steps(int):
            valid_steps(int):
            save_checkpoint_steps(int):
        """
        logger.info('Start training...')

        total_stats = Statistics()
        report_stats = Statistics()

        start_time = total_stats.start_time

        while self.current_step <= train_steps:
            # reduce_counter = 0
            train_iter = train_iter_fct()
            for i, batch in enumerate(train_iter):
                if self.n_gpu != 0 and i % self.n_gpu != self.gpu_rank:
                    continue

                if self.gpu_verbose_level > 1:
                    logger.info("GpuRank %d: index: %d" % (self.gpu_rank, i))
                cur_dataset = train_iter.get_cur_dataset()
                self.train_loss.cur_dataset = cur_dataset

                if self.norm_method == "tokens":
                    norm = batch.tgt[1:].ne(self.train_loss.padding_idx).sum()
                else:
                    norm = batch.batch_size
                if self.n_gpu > 1:
                    norm = sum(onmt.utils.distributed.all_gather_list(norm))

                """
                reduce_counter += 1
                if self.gpu_verbose_level > 0:
                    logger.info("GpuRank %d: reduce_counter: %d n_minibatch %d"
                                % (self.gpu_rank, reduce_counter, 1))
                """

                self._train_batch(batch, norm, total_stats, report_stats)
                if (i + 1) % self.grad_accum_count == 0:
                    self.optim.step()
                    self.model.zero_grad()

                    if self.n_gpu > 1:
                        report_stats = Statistics.all_gather_stats(
                            report_stats)
                    if self._time_to_report():
                        report_stats.output(
                            self.current_step, train_steps,
                            self.learning_rate, start_time)
                        # you additionally do tensorboard writing here

                    report_stats = Statistics()

                    if self.current_step % valid_steps == 0:
                        if self.gpu_verbose_level > 0:
                            logger.info('GpuRank %d: validate step %d'
                                        % (self.gpu_rank, self.current_step))
                        valid_stats = self.validate(valid_iter_fct())
                        if self.gpu_verbose_level > 0:
                            logger.info('GpuRank %d: gather valid stat step %d'
                                        % (self.gpu_rank, self.current_step))
                        if valid_stats is not None and self.n_gpu > 1:
                            valid_stats = Statistics.all_gather_stats(
                                valid_stats)
                        if self.gpu_verbose_level > 0:
                            logger.info('GpuRank %d: report stat step %d'
                                        % (self.gpu_rank, self.current_step))
                        logger.info('Validation perplexity: %g'
                                    % valid_stats.ppl())
                        logger.info('Validation accuracy: %g'
                                    % valid_stats.accuracy())
                        # you additionally do tensorboard writing here

                    if self._time_to_save():
                        self._save()

            if self.gpu_verbose_level > 0:
                logger.info('GpuRank %d: we completed an epoch at step %d'
                            % (self.gpu_rank, self.current_step))

        return total_stats

    def validate(self, valid_iter):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`Statistics`: validation loss statistics
        """
        self.model.eval()

        stats = Statistics()

        for batch in valid_iter:
            cur_dataset = valid_iter.get_cur_dataset()
            self.valid_loss.cur_dataset = cur_dataset

            src = inputters.make_features(batch, 'src', self.data_type)
            src_lengths = batch.src[1] if self.data_type == 'text' else None

            tgt = inputters.make_features(batch, 'tgt')

            # F-prop through the model.
            outputs, attns, _ = self.model(src, tgt, src_lengths)

            # Compute loss.
            batch_stats = self.valid_loss.monolithic_compute_loss(
                batch, outputs, attns)

            # Update statistics.
            stats.update(batch_stats)

        self.model.train()

        return stats

    def _train_batch(self, batch, norm, total_stats, report_stats):
        """
        Previously, the batch training method (which was called
        _gradient_accumulation) was called less than once per batch
        if self.grad_accum_count > 1.
        But there is a simpler way to accumulate gradients:
        just don't step the optimizer and zero the gradients on every
        batch
        """
        target_size = batch.tgt.size(0)
        # Truncated BPTT
        trunc_size = self.trunc_size if self.trunc_size else target_size

        dec_state = None
        src = inputters.make_features(batch, 'src', self.data_type)
        src_lengths = batch.src[1] if self.data_type == 'text' else None
        if self.data_type == 'text':
            report_stats.n_src_words += src_lengths.sum().item()

        tgt_outer = inputters.make_features(batch, 'tgt')

        for j in range(0, target_size - 1, trunc_size):
            # 1. Create truncated target.
            tgt = tgt_outer[j: j + trunc_size]

            # 2. F-prop all but generator.
            # bpop: why was there zero_grad() on each truncation step?
            outputs, attns, dec_state = \
                self.model(src, tgt, src_lengths, dec_state)

            # 3. Compute loss in shards for memory efficiency.
            batch_stats = self.train_loss.sharded_compute_loss(
                batch, outputs, attns, j,
                trunc_size, self.shard_size, norm)
            total_stats.update(batch_stats)
            report_stats.update(batch_stats)

            # If truncated, don't backprop fully.
            if dec_state is not None:
                dec_state.detach()

        # 3.bis Multi GPU gradient gather
        if self.n_gpu > 1:
            grads = [p.grad for p in self.model.parameters()
                     if p.grad is not None]
            onmt.utils.distributed.all_reduce_and_rescale_tensors(
                grads, float(1))

    def _save(self):
        real_model = (self.model.module
                      if isinstance(self.model, nn.DataParallel)
                      else self.model)
        real_generator = (real_model.generator.module
                          if isinstance(real_model.generator, nn.DataParallel)
                          else real_model.generator)

        model_state_dict = real_model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items()
                            if 'generator' not in k}
        generator_state_dict = real_generator.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'generator': generator_state_dict,
            'vocab': onmt.inputters.save_fields_to_vocab(self.fields),
            'opt': self.model_opt,
            'optim': self.optim,
        }

        logger.info("Saving checkpoint %s_step_%d.pt"
                    % (self.base_path, self.current_step))
        checkpoint_path = '%s_step_%d.pt' % (self.base_path, self.current_step)
        torch.save(checkpoint, checkpoint_path)
        """
        if self.keep_checkpoint > 0:
            if len(self.checkpoint_queue) == self.checkpoint_queue.maxlen:
                todel = self.checkpoint_queue.popleft()
                self._rm_checkpoint(todel)
            self.checkpoint_queue.append(checkpoint_path)
        """
