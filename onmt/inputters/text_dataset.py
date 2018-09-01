# -*- coding: utf-8 -*-

import io

import torch

from onmt.inputters.dataset_base import DatasetBase


class TextDataset(DatasetBase):
    """ Dataset for data_type=='text'

        Build `Example` objects, `Field` objects, and filter_pred function
        from text corpus.

        Args:
            fields (dict): a dictionary of `torchtext.data.Field`.
                Keys are like 'src', 'tgt', 'src_map', and 'alignment'.
            src_examples_iter (dict iter): preprocessed source example
                dictionary iterator.
            tgt_examples_iter (dict iter): preprocessed target example
                dictionary iterator.
            num_src_feats (int): number of source side features.
            num_tgt_feats (int): number of target side features.
            src_seq_length (int): maximum source sequence length.
            tgt_seq_length (int): maximum target sequence length.
            dynamic_dict (bool): create dynamic dictionaries?
            use_filter_pred (bool): use a custom filter predicate to filter
                out examples?
    """
    data_type = 'text'

    @staticmethod
    def sort_key(ex):
        """ Sort using length of source sentences. """
        # Default to a balanced sort, prioritizing tgt len match.
        # TODO: make this configurable.
        if hasattr(ex, "tgt"):
            return len(ex.src), len(ex.tgt)
        return len(ex.src)

    @staticmethod
    def collapse_copy_scores(scores, batch, tgt_vocab, src_vocabs):
        """
        Given scores from an expanded dictionary
        corresponeding to a batch, sums together copies,
        with a dictionary word when it is ambigious.
        """
        # this is a static method, but at least one of the arguments is always
        # an attribute of a TextDataset instance. It is not clear why this
        # is in the Dataset at all because it's used for computing scores,
        # not anything relating to iterating over the data.
        offset = len(tgt_vocab)
        for b in range(batch.batch_size):
            blank = []
            fill = []
            index = batch.indices[b]
            src_vocab = src_vocabs[index]
            for i in range(1, len(src_vocab)):
                sw = src_vocab.itos[i]
                ti = tgt_vocab.stoi[sw]
                if ti != 0:
                    blank.append(offset + i)
                    fill.append(ti)
            if blank:
                blank = torch.Tensor(blank).type_as(batch.indices)
                fill = torch.Tensor(fill).type_as(batch.indices)
                scores[:, b].index_add_(1, fill,
                                        scores[:, b].index_select(1, blank))
                scores[:, b].index_fill_(1, blank, 1e-10)
        return scores

    @classmethod
    def _make_example(cls, line, truncate, side, **kwargs):
        return {side: line}

    @classmethod
    def _make_iterator_from_file(cls, path, **kwargs):
        with io.open(path, "r", encoding="utf-8") as corpus_file:
            for line in corpus_file:
                yield line
