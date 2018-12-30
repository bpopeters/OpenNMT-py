#!/usr/bin/env python
# -*- coding: utf-8 -*-

import configargparse
import glob
import sys
import torch

from onmt.utils.logging import init_logger, logger

import onmt.inputters as inputters
import onmt.opts as opts
from onmt.inputters.dataset_base import SigmorphonDataset


def check_existing_pt_files(opt):
    """ Check if there are existing .pt files to avoid overwriting them """
    pattern = opt.save_data + '.{}*.pt'
    for t in ['train', 'valid', 'vocab']:
        path = pattern.format(t)
        if glob.glob(path):
            sys.stderr.write("Please backup existing pt files: %s, "
                             "to avoid overwriting them!\n" % path)
            sys.exit(1)


def parse_args():
    parser = configargparse.ArgumentParser(
        description='preprocess.py',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)

    opts.config_opts(parser)
    opts.add_md_help_argument(parser)
    opts.preprocess_opts(parser)

    opt = parser.parse_args()
    torch.manual_seed(opt.seed)

    check_existing_pt_files(opt)

    return opt


def main():
    opt = parse_args()

    assert opt.max_shard_size == 0, \
        "-max_shard_size is deprecated. Please use \
        -shard_size (number of examples) instead."
    assert opt.shuffle == 0, \
        "-shuffle is not implemented. Please shuffle \
        your data before pre-processing."

    init_logger(opt.log_file)

    logger.info("Building `Fields` object...")
    fields = inputters.get_fields(
        language=opt.multilingual, share_vocab=opt.share_vocab)

    # make datasets
    train_dataset = SigmorphonDataset(fields, opt.train)
    logger.info('Train set size: {}'.format(len(train_dataset)))
    valid_dataset = SigmorphonDataset(fields, opt.valid)
    logger.info('Validation set size: {}'.format(len(valid_dataset)))

    # build vocab
    for in_label, column_fields in fields.items():
        for name, field in column_fields:
            if field.use_vocab:
                field.build_vocab(train_dataset)

    # log some stuff
    src_vocab_size = len(fields['src'][0][1].vocab)
    tgt_vocab_size = len(fields['tgt'][0][1].vocab)
    logger.info("Vocab sizes: src {} ; tgt {}".format(
        src_vocab_size, tgt_vocab_size))
    inflection_vocab_size = len(fields['inflection'][0][1].vocab)
    logger.info("Unique inflectional tags: {}".format(inflection_vocab_size))
    if 'language' in fields:
        n_languages = len(fields['language'][0][1].vocab)
        logger.info("Number of languages: {}".format(n_languages))

    train_dataset.save(opt.save_data + '.train.0.pt')
    valid_dataset.save(opt.save_data + '.valid.0.pt')
    torch.save(fields, opt.save_data + '.vocab.pt')


if __name__ == "__main__":
    main()
