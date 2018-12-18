# coding: utf-8

from itertools import chain
from collections import Counter

import torch
from torchtext.data import Example, Dataset
from torchtext.vocab import Vocab


class DatasetBase(Dataset):
    """
    A dataset basically supports iteration over all the examples
    it contains. We currently have 3 datasets inheriting this base
    for 3 types of corpus respectively: "text", "img", "audio".

    Internally it initializes an `torchtext.data.Dataset` object with
    the following attributes:

     `examples`: a sequence of `torchtext.data.Example` objects.
     `fields`: a dictionary associating str keys with `torchtext.data.Field`
        objects, and not necessarily having the same keys as the input fields.
    """

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, _d):
        self.__dict__.update(_d)

    def __reduce_ex__(self, proto):
        # This is a hack. Something is broken with torch pickle.
        return super(DatasetBase, self).__reduce_ex__()

    def __init__(self, fields, src_examples_iter, tgt_examples_iter,
                 dynamic_dict=False, filter_pred=None):

        # Each element of an example is a dictionary whose keys represents
        # at minimum the src tokens and their indices and potentially also
        # the src and tgt features and alignment information.
        if tgt_examples_iter is not None:
            examples_iter = (self._join_dicts(src, tgt) for src, tgt in
                             zip(src_examples_iter, tgt_examples_iter))
        else:
            examples_iter = src_examples_iter

        # self.src_vocabs is used in collapse_copy_scores and in Translator.py
        self.src_vocabs = []
        if dynamic_dict:
            unk = fields['src'][0][1].unk_token
            pad = fields['src'][0][1].pad_token
            examples_iter = (self._dynamic_dict(ex, unk, pad)
                             for ex in examples_iter)

        # the field gets filtered because there are problems if the field
        # contains keys not present in the example
        examples = \
            [Example.fromdict(ex, {k: v for k, v in fields.items() if k in ex})
             for ex in examples_iter]

        super(DatasetBase, self).__init__(examples, fields, filter_pred)

    def save(self, path, remove_fields=True):
        if remove_fields:
            self.fields = []
        torch.save(self, path)

    def _join_dicts(self, *args):
        """
        Args:
            dictionaries with disjoint keys.

        Returns:
            a single dictionary that has the union of these keys.
        """
        return dict(chain(*[d.items() for d in args]))

    def _dynamic_dict(self, example, unk, pad):
        # it would not be necessary to pass unk and pad if the method were
        # called after fields becomes an attribute of self
        src = example["src"]
        src_vocab = Vocab(Counter(src), specials=[unk, pad])
        self.src_vocabs.append(src_vocab)
        # Map source tokens to indices in the dynamic dict.
        src_map = torch.LongTensor([src_vocab.stoi[w] for w in src])
        example["src_map"] = src_map

        if "tgt" in example:
            tgt = example["tgt"]
            mask = torch.LongTensor(
                [0] + [src_vocab.stoi[w] for w in tgt] + [0])
            example["alignment"] = mask
        return example
