#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division
import six
import sys
import argparse
import numpy as np  # only here because you cannot fill a tensor with a list
import torch
from onmt.utils.logging import init_logger, logger
from onmt.inputters.inputter import old_style_vocab


def get_vocabs(dict_path):
    fields = torch.load(dict_path)

    if old_style_vocab(fields):
        enc_vocab = next((v for n, v in fields if n == 'src'), None)
        dec_vocab = next((v for n, v in fields if n == 'tgt'), None)
    else:
        enc_vocab = fields['src'][0][1].vocab
        dec_vocab = fields['tgt'][0][1].vocab

    logger.info("From: %s" % dict_path)
    logger.info("\t* source vocab: %d words" % len(enc_vocab))
    logger.info("\t* target vocab: %d words" % len(dec_vocab))

    return enc_vocab, dec_vocab


def read_embeddings(file_enc, skip_lines=0):
    embs = dict()
    with open(file_enc, 'rb') as f:
        for i, line in enumerate(f):
            if i < skip_lines:
                continue
            if not line:
                break
            if len(line) == 0:
                # is this reachable?
                continue

            l_split = line.decode('utf8').strip().split(' ')
            if len(l_split) == 2:
                continue
            embs[l_split[0]] = [float(em) for em in l_split[1:]]
    return embs


def filter_embeddings(vocab, emb, verbose=False):
    dim = len(six.next(six.itervalues(emb)))
    filtered_embeddings = np.zeros((len(vocab), dim))
    for i, w in enumerate(vocab.itos):
        if w in emb:
            filtered_embeddings[i] = emb[w]
        elif verbose:
            logger.info(u"not found:\t{}".format(w), file=sys.stderr)

    return torch.FloatTensor(filtered_embeddings)


def main():

    parser = argparse.ArgumentParser(description='embeddings_to_torch.py')
    parser.add_argument('-emb_file_enc', required=True,
                        help="source Embeddings from this file")
    parser.add_argument('-emb_file_dec', required=True,
                        help="target Embeddings from this file")
    parser.add_argument('-output_file', required=True,
                        help="Output file for the prepared data")
    parser.add_argument('-dict_file', required=True,
                        help="Dictionary file")
    parser.add_argument('-verbose', action="store_true", default=False)
    parser.add_argument('-skip_lines', type=int, default=0,
                        help="Skip first lines of the embedding file")
    parser.add_argument('-type', choices=["GloVe", "word2vec"],
                        default="GloVe")
    opt = parser.parse_args()

    src_vocab, tgt_vocab = get_vocabs(opt.dict_file)

    skip_lines = 1 if opt.type == "word2vec" else opt.skip_lines
    src_vectors = read_embeddings(opt.emb_file_enc, skip_lines)
    logger.info("Got {} encoder embeddings from {}".format(
        len(src_vectors), opt.emb_file_enc))

    tgt_vectors = read_embeddings(opt.emb_file_dec)
    logger.info("Got {} decoder embeddings from {}".format(
        len(tgt_vectors), opt.emb_file_dec))

    src_emb = filter_embeddings(src_vocab, src_vectors, opt.verbose)
    tgt_emb = filter_embeddings(tgt_vocab, tgt_vectors, opt.verbose)

    logger.info("\nMatching: ")
    src_matches = src_emb.size(0)
    src_misses = len(src_vocab) - src_matches
    src_match_percent = 100 * src_matches / len(src_vocab)
    logger.info("\t* enc: %d match, %d missing, (%.2f%%)"
                % (src_matches, src_misses, tgt_match_percent))

    tgt_matches = tgt_emb.size(0)
    tgt_misses = len(tgt_vocab) - tgt_matches
    tgt_match_percent = 100 * tgt_matches / len(tgt_vocab)
    logger.info("\t* dec: %d match, %d missing, (%.2f%%)"
                % (tgt_matches, tgt_misses, tgt_match_percent))

    logger.info("\nFiltered embeddings:")
    logger.info("\t* enc: %s" % str(src_emb.size()))
    logger.info("\t* dec: %s" % str(tgt_emb.size()))

    enc_output_file = opt.output_file + ".enc.pt"
    dec_output_file = opt.output_file + ".dec.pt"
    logger.info("\nSaving embedding as:\n\t* enc: %s\n\t* dec: %s"
                % (enc_output_file, dec_output_file))
    torch.save(src_emb, enc_output_file)
    torch.save(tgt_emb, dec_output_file)


if __name__ == "__main__":
    init_logger('embeddings_to_torch.log')
    main()
