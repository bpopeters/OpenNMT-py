#!/usr/bin/env python
"""
    Training on a single process
"""
import argparse
import os
import random
import torch

import onmt.opts as opts

from onmt.inputters.inputter import build_dataset_iter, lazily_load_dataset, \
    collect_feature_vocabs
from onmt.model_builder import build_model
from onmt.utils.optimizers import build_optim
from onmt.trainer import build_trainer
from onmt.utils.logging import init_logger, logger


def _check_save_model_path(opt):
    save_model_path = os.path.abspath(opt.save_model)
    model_dirname = os.path.dirname(save_model_path)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)


def _tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        elif 'decoder' or 'generator' in name:
            dec += param.nelement()
    return n_params, enc, dec


def training_opt_postprocessing(opt):
    if opt.word_vec_size != -1:
        opt.src_word_vec_size = opt.word_vec_size
        opt.tgt_word_vec_size = opt.word_vec_size

    if opt.layers != -1:
        opt.enc_layers = opt.layers
        opt.dec_layers = opt.layers

    opt.brnn = opt.encoder_type == "brnn"

    if opt.rnn_type == "SRU" and not opt.gpuid:
        raise AssertionError("Using SRU requires -gpuid set.")

    if torch.cuda.is_available() and not opt.gpuid:
        logger.info("WARNING: You have a CUDA device, should run with -gpuid")

    if opt.gpuid:
        torch.cuda.set_device(opt.device_id)
        if opt.seed > 0:
            # this one is needed for torchtext random call (shuffled iterator)
            # in multi gpu it ensures datasets are read in the same order
            random.seed(opt.seed)
            # These ensure same initialization in multi gpu mode
            torch.manual_seed(opt.seed)
            torch.cuda.manual_seed(opt.seed)

    return opt


def main(opt):
    opt = training_opt_postprocessing(opt)
    init_logger(opt.log_file)
    # Load checkpoint if we resume from a previous training.
    if opt.train_from:
        logger.info('Loading checkpoint from %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from,
                                map_location=lambda storage, loc: storage)
        model_opt = checkpoint['opt']
    else:
        checkpoint = None
        model_opt = opt

    # Peek at the first dataset to determine the data_type.
    # (All datasets have the same data_type).
    first_dataset = next(lazily_load_dataset("train", opt.data))
    data_type = first_dataset.data_type
    fields = first_dataset.fields
    if data_type == 'text':
        logger.info(' * vocabulary size. source = %d; target = %d' %
                    (len(fields['src'].vocab), len(fields['tgt'].vocab)))
    else:
        logger.info(' * vocabulary size. target = %d' %
                    len(fields['tgt'].vocab))

    if model_opt.model_type == 'text':
        # is model_opt.model_type the same as data_type?
        src_dict = fields["src"].vocab
        src_feat_vocabs = collect_feature_vocabs(fields, 'src')
    else:
        src_dict = None
        src_feat_vocabs = []

    tgt_dict = fields["tgt"].vocab
    tgt_feat_vocabs = collect_feature_vocabs(fields, 'tgt')

    # Report src/tgt features.
    for j, src_feat_vocab in enumerate(src_feat_vocabs):
        logger.info(' * src feature %d size = %d' % (j, len(src_feat_vocab)))
    for j, tgt_feat_vocab in enumerate(tgt_feat_vocabs):
        logger.info(' * tgt feature %d size = %d' % (j, len(tgt_feat_vocab)))

    # Build model.
    # why does building the model also do data loading stuff?
    model = build_model(model_opt, opt, src_dict, src_feat_vocabs,
                        tgt_dict, tgt_feat_vocabs, checkpoint)
    n_params, enc, dec = _tally_parameters(model)
    logger.info('encoder: %d' % enc)
    logger.info('decoder: %d' % dec)
    logger.info('* number of parameters: %d' % n_params)
    _check_save_model_path(opt)

    # Build optimizer.
    optim = build_optim(model, opt, checkpoint)

    trainer = build_trainer(opt, model_opt, model, fields, optim, data_type)

    def train_iter_fct(): return build_dataset_iter(
        lazily_load_dataset("train", opt.data), opt)

    def valid_iter_fct(): return build_dataset_iter(
        lazily_load_dataset("valid", opt.data), opt)

    # Do training.
    trainer.train(train_iter_fct, valid_iter_fct, opt.train_steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    opts.add_md_help_argument(parser)
    opts.model_opts(parser)
    opts.train_opts(parser)

    opt = parser.parse_args()
    main(opt)
