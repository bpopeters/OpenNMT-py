"""
    This is the loadable seq2seq trainer library that is
    in charge of training details, loss compute, and statistics.
    See train.py for a use case of this library.

    Note: To make this a general library, we implement *only*
          mechanism things here(i.e. what to do), and leave the strategy
          things to users(i.e. how to do it). Also see train.py (one of the
          users of this library) for the strategy things we do.
"""
import os
from datetime import datetime
from collections import deque
from itertools import count

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
    validate_every = opt.valid_steps
    save_checkpoint_steps = opt.save_checkpoint_steps
    keep_checkpoint = opt.keep_checkpoint
    save_model = opt.save_model
    if opt.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(opt.tensorboard_log_dir
                               + datetime.now().strftime("/%b-%d_%H-%M-%S"),
                               comment="Unmt")
    else:
        writer = None

    trainer = onmt.Trainer(model, model_opt, fields, train_loss, valid_loss,
                           optim, trunc_size, shard_size, data_type,
                           norm_method, grad_accum_count, n_gpu, gpu_rank,
                           gpu_verbose_level, report_every, validate_every,
                           save_checkpoint_steps, keep_checkpoint, save_model,
                           writer)
    return trainer


def cycle_batches(train_iter_fct):
    "Kind of similar to itertools.cycle"
    for epoch in count(1):
        train_iter = train_iter_fct()
        current_dataset = train_iter.get_cur_dataset()
        for i, batch in enumerate(train_iter):
            yield current_dataset, epoch, i, batch


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
    """

    def __init__(self, model, model_opt, fields, train_loss, valid_loss, optim,
                 trunc_size=0, shard_size=32, data_type='text',
                 norm_method="sents", grad_accum_count=1, n_gpu=1, gpu_rank=1,
                 gpu_verbose_level=0, report_every=50, validate_every=10000,
                 save_checkpoint_steps=5000, keep_checkpoint=-1,
                 save_model='model', writer=None):
        assert grad_accum_count > 0
        assert grad_accum_count == 1 or trunc_size == 0, \
            "To enable accumulated gradients, you must disable truncated BPTT."

        self._model = model
        self._model_opt = model_opt
        self._fields = fields
        self._train_loss = train_loss
        self._valid_loss = valid_loss
        self._optim = optim
        self._trunc_size = trunc_size
        self._shard_size = shard_size
        self._data_type = data_type
        self._norm_method = norm_method
        self._grad_accum_count = grad_accum_count
        self._n_gpu = n_gpu
        self._gpu_rank = gpu_rank
        self._gpu_verbose_level = gpu_verbose_level
        self._report_every = report_every
        self._validate_every = validate_every
        self._save_checkpoint_steps = save_checkpoint_steps
        self._keep_checkpoint = keep_checkpoint
        self._base_path = save_model
        self._tensorboard_writer = writer

        if keep_checkpoint > 0:
            self._checkpoint_queue = deque([], maxlen=keep_checkpoint)

        # Set model in training mode.
        self._model.train()

    @property
    def current_step(self):
        return self._optim._step + 1

    @property
    def multigpu(self):
        return self._n_gpu > 1

    @property
    def learning_rate(self):
        return self._optim.learning_rate

    def train(self, train_iter_fct, valid_iter_fct, train_steps):
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
        """
        logger.info('Start training...')

        total_stats = Statistics()
        report_stats = Statistics()

        start_time = total_stats.start_time

        batch_count = range(0, train_steps * self._grad_accum_count)
        batches = cycle_batches(train_iter_fct)  # not sure I like this now
        for _, (cur_dataset, epoch, i, batch) in zip(batch_count, batches):
            if self._n_gpu != 0 and i % self._n_gpu != self._gpu_rank:
                continue

            if self._gpu_verbose_level > 1:
                logger.info("GpuRank %d: index: %d" % (self._gpu_rank, i))
            self._train_loss.cur_dataset = cur_dataset

            norm = self._norm(batch)

            """
            reduce_counter += 1
            if self._gpu_verbose_level > 0:
                logger.info("GpuRank %d: reduce_counter: %d n_minibatch %d"
                            % (self._gpu_rank, reduce_counter, 1))
            """

            # compute forward and backward passes for the batch
            self._train_batch(batch, norm, total_stats, report_stats)

            if not self._time_to_step(i):
                continue

            # update parameters
            self._optim.step()
            self._model.zero_grad()

            # do special update to stats in multigpu case
            if self.multigpu:
                report_stats = Statistics.all_gather_stats(report_stats)
            if self._time_to_report():
                report_stats.output(
                    self.current_step, train_steps,
                    self.learning_rate, start_time)
                if self._tensorboard_writer is not None:
                    self._log_tensorboard(report_stats, "progress")

            report_stats = Statistics()

            if self._time_to_validate():
                if self._gpu_verbose_level > 0:
                    logger.info('GpuRank %d: validate step %d'
                                % (self._gpu_rank, self.current_step))
                valid_stats = self.validate(valid_iter_fct())
                if self._gpu_verbose_level > 0:
                    logger.info('GpuRank %d: gather valid stat step %d'
                                % (self._gpu_rank, self.current_step))
                if valid_stats is not None and self._n_gpu > 1:
                    valid_stats = Statistics.all_gather_stats(
                        self._validate_every)
                if self._gpu_verbose_level > 0:
                    logger.info('GpuRank %d: report stat step %d'
                                % (self._gpu_rank, self.current_step))
                logger.info('Validation perplexity: %g' % valid_stats.ppl())
                logger.info('Validation accuracy: %g' % valid_stats.accuracy())
                if self._tensorboard_writer is not None:
                    self._log_tensorboard(report_stats, "valid")

            if self._time_to_save():
                self._save()
            """
            if self._gpu_verbose_level > 0:
                logger.info('GpuRank %d: we completed an epoch at step %d'
                            % (self._gpu_rank, self.current_step))
            """
        return total_stats

    def validate(self, valid_iter):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`Statistics`: validation loss statistics
        """
        self._model.eval()

        stats = Statistics()

        for batch in valid_iter:
            cur_dataset = valid_iter.get_cur_dataset()
            self._valid_loss.cur_dataset = cur_dataset

            src = inputters.make_features(batch, 'src', self._data_type)
            src_lengths = batch.src[1] if self._data_type == 'text' else None

            tgt = inputters.make_features(batch, 'tgt')

            outputs, attns, _ = self._model(src, tgt, src_lengths)

            batch_stats = self._valid_loss.monolithic_compute_loss(
                batch, outputs, attns)

            stats.update(batch_stats)

        self._model.train()

        return stats

    def _train_batch(self, batch, norm, total_stats, report_stats):
        """
        Previously, the batch training method (which was called
        _gradient_accumulation) was called less than once per batch
        if self._grad_accum_count > 1.
        But there is a simpler way to accumulate gradients:
        just don't step the optimizer and zero the gradients on every
        batch
        """
        target_size = batch.tgt.size(0)
        # Truncated BPTT
        trunc_size = self._trunc_size if self._trunc_size else target_size

        dec_state = None
        src = inputters.make_features(batch, 'src', self._data_type)
        src_lengths = batch.src[1] if self._data_type == 'text' else None
        if self._data_type == 'text':
            report_stats.n_src_words += src_lengths.sum().item()

        tgt_outer = inputters.make_features(batch, 'tgt')

        for j in range(0, target_size - 1, trunc_size):
            # 1. Create truncated target.
            tgt = tgt_outer[j: j + trunc_size]

            # 2. F-prop all but generator.
            # bpop: why was there zero_grad() on each truncation step?
            outputs, attns, dec_state = \
                self._model(src, tgt, src_lengths, dec_state)

            # 3. Compute loss in shards for memory efficiency.
            batch_stats = self._train_loss.sharded_compute_loss(
                batch, outputs, attns, j,
                trunc_size, self._shard_size, norm)
            total_stats.update(batch_stats)
            report_stats.update(batch_stats)

            # If truncated, don't backprop fully.
            if dec_state is not None:
                dec_state.detach()

        # 3.bis Multi GPU gradient gather
        if self._n_gpu > 1:
            grads = [p.grad for p in self._model.parameters()
                     if p.grad is not None]
            onmt.utils.distributed.all_reduce_and_rescale_tensors(
                grads, float(1))

    def _time_to_step(self, batch_step):
        return (batch_step + 1) % self._grad_accum_count == 0

    def _time_to_report(self):
        return self.current_step % self._report_every == 0

    def _time_to_validate(self):
        return self.current_step % self._validate_every == 0

    def _time_to_save(self):
        return self._gpu_rank == 0 and self._keep_checkpoint != 0 and \
            self.current_step % self._save_checkpoint_steps == 0

    def _norm(self, batch):
        if self._norm_method == "tokens":
            norm = batch.tgt[1:].ne(self._train_loss.padding_idx).sum()
        else:
            norm = batch.batch_size
        if self.multigpu:
            norm = sum(onmt.utils.distributed.all_gather_list(norm))
        return norm

    def _log_tensorboard(self, stats, prefix):
        stats.log_tensorboard(
                prefix, self._tensorboard_writer,
                self._learning_rate, self.current_step)

    def _save(self):
        real_model = (self._model.module
                      if isinstance(self._model, nn.DataParallel)
                      else self._model)
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
            'opt': self._model_opt,
            'optim': self._optim,
        }

        logger.info("Saving checkpoint %s_step_%d.pt"
                    % (self._base_path, self.current_step))
        checkpoint_path = '%s_step_%d.pt' \
            % (self._base_path, self.current_step)
        torch.save(checkpoint, checkpoint_path)
        if self._keep_checkpoint > 0:
            if len(self._checkpoint_queue) == self._checkpoint_queue.maxlen:
                todel = self._checkpoint_queue.popleft()
                os.remove(todel)
            self._checkpoint_queue.append(checkpoint_path)
