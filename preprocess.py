#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Pre-process Data / features files and build vocabulary
"""

import configargparse
import glob
import sys
import gc
import os
import codecs
from itertools import islice
import torch
from onmt.utils.logging import init_logger, logger

import onmt.inputters as inputters
import onmt.opts as opts


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


def split_corpus(path, shard_size):
    with codecs.open(path, "r", encoding="utf-8") as f:
        while True:
            shard = list(islice(f, shard_size))
            if not shard:
                break
            yield shard


def build_datasets(src, tgt, fields, opt, use_filter_pred=True):
    logger.info("Reading source and target files: %s %s." % (src, tgt))

    src_shards = split_corpus(src, opt.shard_size)
    tgt_shards = split_corpus(tgt, opt.shard_size)
    shard_pairs = zip(src_shards, tgt_shards)

    for src_shard, tgt_shard in shard_pairs:
        assert len(src_shard) == len(tgt_shard)
        dataset = inputters.build_dataset(
            fields, opt.data_type,
            src=src_shard,
            tgt=tgt_shard,
            src_dir=opt.src_dir,
            src_seq_len=opt.src_seq_length,
            tgt_seq_len=opt.tgt_seq_length,
            src_seq_length_trunc=opt.src_seq_length_trunc,
            tgt_seq_length_trunc=opt.tgt_seq_length_trunc,
            dynamic_dict=opt.dynamic_dict,
            sample_rate=opt.sample_rate,
            window_size=opt.window_size,
            window_stride=opt.window_stride,
            window=opt.window,
            image_channel_size=opt.image_channel_size,
            use_filter_pred=use_filter_pred
        )
        yield dataset


def count_features(path):
    """
    path: location of a corpus file with whitespace-delimited tokens and
                    ￨-delimited features within the token
    returns: the number of features in the dataset
    """
    with codecs.open(path, "r", "utf-8") as f:
        first_tok = f.readline().split(None, 1)[0]
        return len(first_tok.split(u"￨")) - 1


def main():
    opt = parse_args()

    assert opt.max_shard_size == 0, \
        "-max_shard_size is deprecated. Please use \
        -shard_size (number of examples) instead."
    assert opt.shuffle == 0, \
        "-shuffle is not implemented. Please shuffle \
        your data before pre-processing."

    assert os.path.isfile(opt.train_src) and os.path.isfile(opt.train_tgt), \
        "Please check path of your train src and tgt files!"

    assert os.path.isfile(opt.valid_src) and os.path.isfile(opt.valid_tgt), \
        "Please check path of your valid src and tgt files!"

    init_logger(opt.log_file)
    logger.info("Extracting features...")

    src_nfeats = count_features(opt.train_src) if opt.data_type == 'text' \
        else 0
    tgt_nfeats = count_features(opt.train_tgt)  # tgt always text so far
    logger.info(" * number of source features: %d." % src_nfeats)
    logger.info(" * number of target features: %d." % tgt_nfeats)

    logger.info("Building `Fields` object...")
    fields = inputters.get_fields(opt.data_type, src_nfeats, tgt_nfeats)

    logger.info("Building training data...")
    train_shards = build_datasets(opt.train_src, opt.train_tgt, fields, opt)
    for i, shard in enumerate(train_shards):
        logger.info("Building shard %d." % i)
        for name, field in fields.items():
            if field.use_vocab:
                field.extend_vocab(shard)
        data_path = "{:s}.{:s}.{:d}.pt".format(opt.save_data, 'train', i)
        shard.save(data_path)
        logger.info(" * saving %sth %s data shard to %s."
                    % (i, 'train', data_path))
        del shard.examples
        gc.collect()
        del shard
        gc.collect()

    logger.info("Building validation data...")
    valid_shards = build_datasets(
        opt.valid_src, opt.valid_tgt, fields, opt, opt.filter_valid)
    for i, shard in enumerate(valid_shards):
        logger.info("Building shard %d." % i)
        data_path = "{:s}.{:s}.{:d}.pt".format(opt.save_data, 'valid', i)
        shard.save(data_path)
        logger.info(" * saving %sth %s data shard to %s."
                    % (i, 'valid', data_path))
        del shard.examples
        gc.collect()
        del shard
        gc.collect()

    # things you still need to do with vocab: vocab size, min_freq, sharing
    logger.info("Saving vocabulary...")
    vocab_path = opt.save_data + '.vocab.pt'
    torch.save(inputters.save_fields_to_vocab(fields), vocab_path)


if __name__ == "__main__":
    main()
