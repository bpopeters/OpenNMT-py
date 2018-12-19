# coding: utf-8

from itertools import chain
from collections import Counter

import torch
from torchtext.data import Example, Dataset
from torchtext.vocab import Vocab


class DatasetBase(Dataset):
    """
    A dataset is an object that accepts sequences of raw data (sentence pairs
    in the case of machine translation) and fields which describe how this
    raw data should be processed to produce tensors. When a dataset is
    instantiated, it applies the fields' preprocessing pipeline (but not
    the bit that numericalizes it or turns it into batch tensors) to the raw
    data, producing a list of torchtext.data.Example objects. torchtext's
    iterators then know how to use these examples to make batches.

    Datasets in OpenNMT take three positional arguments:

    `fields`: a dict with the structure returned by inputters.get_fields().
        keys match the keys of items yielded by the src_examples_iter or
        tgt_examples_iter, while values are lists of (name, Field) pairs.
        An attribute with this name will be created for each Example object,
        and its value will be the result of applying the Field to the data
        that matches the key. The advantage of having sequences of fields
        for each piece of raw input is that it allows for the dataset to store
        multiple `views` of each input, which allows for easy implementation
        of token-level features, mixed word- and character-level models, and
        so on.
    `src_examples_iter`: a sequence of dicts. Each dict's keys should be a
        subset of the keys in `fields`.
    `tgt_examples_iter`: like `src_examples_iter`, but may be None (this is
        the case at translation time if no target is specified).

    (todo: describe the optional arguments)

    The resulting dataset will have three attributes (todo: also src_vocabs):

     `examples`: a list of `torchtext.data.Example` objects with attributes as
        described above.
     `fields`: a dictionary whose keys are strings that correspond to the
        attributes of the elements of `examples` and whose values are
        the corresponding `torchtext.data.Field` objects. NOTE: this is not
        the same structure as in the fields argument passed to the constructor.
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

        examples = \
            [Example.fromdict(
                ex, {k: v for k, v in fields.items() if k in ex.__dict__})
             for ex in examples_iter]

        fields = dict(chain.from_iterable(fields.values()))  # flatten fields

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
