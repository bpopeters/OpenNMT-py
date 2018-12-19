# -*- coding: utf-8 -*-
import glob
import os
import codecs

from collections import Counter, defaultdict, OrderedDict
from itertools import count, chain
from functools import partial

import torch
import torchtext.data
from torchtext.data import Field
from torchtext.vocab import Vocab

from onmt.inputters.text_dataset import TextDataset
from onmt.inputters.image_dataset import ImageDataset
from onmt.inputters.audio_dataset import AudioDataset
from onmt.utils.logging import logger

import gc


def _getstate(self):
    return dict(self.__dict__, stoi=dict(self.stoi))


def _setstate(self, state):
    self.__dict__.update(state)
    self.stoi = defaultdict(lambda: 0, self.stoi)


Vocab.__getstate__ = _getstate
Vocab.__setstate__ = _setstate


def make_src(data, vocab):
    src_size = max([t.size(0) for t in data])
    src_vocab_size = max([t.max() for t in data]) + 1
    alignment = torch.zeros(src_size, len(data), src_vocab_size)
    for i, sent in enumerate(data):
        for j, t in enumerate(sent):
            alignment[j, i, t] = 1
    return alignment


def make_tgt(data, vocab):
    tgt_size = max([t.size(0) for t in data])
    alignment = torch.zeros(tgt_size, len(data)).long()
    for i, sent in enumerate(data):
        alignment[:sent.size(0), i] = sent
    return alignment


def make_img(data, vocab):
    c = data[0].size(0)
    h = max([t.size(1) for t in data])
    w = max([t.size(2) for t in data])
    imgs = torch.zeros(len(data), c, h, w).fill_(1)
    for i, img in enumerate(data):
        imgs[i, :, 0:img.size(1), 0:img.size(2)] = img
    return imgs


def make_audio(data, vocab):
    """ batch audio data """
    nfft = data[0].size(0)
    t = max([t.size(1) for t in data])
    sounds = torch.zeros(len(data), 1, nfft, t)
    for i, spect in enumerate(data):
        sounds[i, :, :, 0:spect.size(1)] = spect
    return sounds


# mix this with partial
def _feature_tokenize(string, layer=0, tok_delim=None, feat_delim=u"￨"):
    return [t.split(feat_delim)[layer] for t in string.split(tok_delim)]


def get_fields(
    src_data_type,
    n_src_feats,
    n_tgt_feats,
    pad='<blank>',
    bos='<s>',
    eos='</s>'
):
    """
    Args:
        src_data_type: type of the source input. Options are [text|img|audio].
        n_src_features: the number of source features to
            create `torchtext.data.Field` for.
        n_tgt_features: the number of target features to
            create `torchtext.data.Field` for.

    Returns:
        A dictionary whose keys are strings and whose values are the
        corresponding Field objects.
    """
    assert src_data_type in ['text', 'img', 'audio'], \
        "Data type not implemented"
    fields = {'src': [], 'tgt': []}

    if src_data_type == 'text':
        for i in range(n_src_feats + 1):
            name = "src_feat_" + str(i - 1) if i > 0 else "src"
            if n_src_feats > 0:
                tokenize = partial(_feature_tokenize, layer=i)
            else:
                tokenize = None
            use_len = i == 0
            feat = Field(
                pad_token=pad, tokenize=tokenize, include_lengths=use_len)
            fields['src'].append((name, feat))

    elif src_data_type == 'img':
        img = Field(
            use_vocab=False, dtype=torch.float,
            postprocessing=make_img, sequential=False)
        fields["src"].append(('src', img))
    else:
        audio = Field(
            use_vocab=False, dtype=torch.float,
            postprocessing=make_audio, sequential=False)
        fields["src"].append(('src', audio))

    if src_data_type == 'audio':
        # only audio has src_lengths
        length = Field(use_vocab=False, dtype=torch.long, sequential=False)
        fields["src_lengths"] = [("src_lengths", length)]
    else:
        # everything except audio has src_map and alignment
        src_map = Field(
            use_vocab=False, dtype=torch.float,
            postprocessing=make_src, sequential=False)
        fields["src_map"] = [("src_map", src_map)]

        align = Field(
            use_vocab=False, dtype=torch.long,
            postprocessing=make_tgt, sequential=False)
        fields["alignment"] = [('alignment', align)]

    # below this: things defined no matter what the data source type is
    for i in range(n_tgt_feats + 1):
        name = "tgt_feat_" + str(i - 1) if i > 0 else "tgt"
        if n_tgt_feats > 0:
            tokenize = partial(_feature_tokenize, layer=i)
        else:
            tokenize = None
        feat = Field(
            init_token=bos,
            eos_token=eos,
            pad_token=pad,
            tokenize=tokenize)
        fields['tgt'].append((name, feat))

    indices = Field(use_vocab=False, dtype=torch.long, sequential=False)
    fields["indices"] = [('indices', indices)]

    return fields


def load_fields_from_vocab(vocab, data_type="text"):
    """
    vocab: a list of (field name, torchtext.vocab.Vocab) pairs
    data_type: text, img, or audio
    returns: a dictionary whose keys are the field names and whose values
             are field objects with the vocab set to the corresponding vocab
             object from the input.
    """
    # this might break if you try to train with old preprocessed models
    n_src_features = len(vocab['src']) - 1
    n_tgt_features = len(vocab['tgt']) - 1
    fields = get_fields(data_type, n_src_features, n_tgt_features)
    for k, values in vocab.items():
        for i, (n, v) in enumerate(values):
            fields[k][i][1].vocab = v
    return fields


def save_fields_to_vocab(fields):
    """
    fields: a dict with the structure of the object returned by get_fields
    returns: a dict which is similar to this one, but with the
        torchtext.data.Field objects replaced by either torchtext.vocab.Vocab
        objects (if the Field had a vocab) and None (otherwise)
    """
    vocabs = dict()
    for key, field_seq in fields.items():
        vocabs[key] = []
        for n, f in field_seq:
            # the f is not None check is a relic: not sure it is still needed
            v = f.vocab if f is not None and 'vocab' in f.__dict__ else None
            vocabs[key].append((n, v))
    return vocabs


def make_features(batch, side, data_type='text'):
    """
    Args:
        batch (Tensor): a batch of source or target data.
        side (str): for source or for target.
        data_type (str): type of the source input.
            Options are [text|img|audio].
    Returns:
        A sequence of src/tgt tensors with optional feature tensors
        of size (len x batch).
    """
    assert side in ['src', 'tgt']
    if isinstance(batch.__dict__[side], tuple):
        data = batch.__dict__[side][0]
    else:
        data = batch.__dict__[side]

    feat_start = side + "_feat_"
    keys = sorted([k for k in batch.__dict__ if feat_start in k])
    features = [batch.__dict__[k] for k in keys]
    levels = [data] + features

    if data_type == 'text':
        return torch.cat([level.unsqueeze(2) for level in levels], 2)
    else:
        return levels[0]


# min_len is misnamed
def filter_example(ex, use_src_len=True, use_tgt_len=True,
                   min_src_len=0, max_src_len=float('inf'),
                   min_tgt_len=0, max_tgt_len=float('inf')):
    """
    A generalized function for filtering examples based on the length of their
    src or tgt values. Rather than being used by itself as the filter_pred
    argument to a dataset, it should be partially evaluated with everything
    specified except the value of the example.
    """
    return (not use_src_len or min_src_len < len(ex.src) <= max_src_len) and \
        (not use_tgt_len or min_tgt_len < len(ex.tgt) <= max_tgt_len)


def build_dataset(fields, data_type, src,
                  src_dir=None, tgt=None,
                  src_seq_len=0, tgt_seq_len=0,
                  src_seq_length_trunc=0, tgt_seq_length_trunc=0,
                  dynamic_dict=False, sample_rate=0,
                  window_size=0, window_stride=0, window=None,
                  normalize_audio=True, use_filter_pred=True,
                  image_channel_size=3):
    """
    src: path to corpus file or iterator over source data
    tgt: path to corpus file, iterator over target data, or None
    """
    dataset_classes = {
        'text': TextDataset, 'img': ImageDataset, 'audio': AudioDataset
    }
    assert data_type in dataset_classes
    assert src is not None
    assert not dynamic_dict or data_type == 'text', \
        'it is not possible to use dynamic_dict with non-text input'
    if data_type == 'text':
        src_examples_iter = TextDataset.make_examples(
            src, src_seq_length_trunc, "src"
        )
    elif data_type == 'img':
        # there is a truncate argument as well, but it was never set to
        # anything besides None before
        src_examples_iter = ImageDataset.make_examples(
            src, src_dir, 'src', channel_size=image_channel_size
        )
    else:
        src_examples_iter = AudioDataset.make_examples(
            src, src_dir, "src", sample_rate,
            window_size, window_stride, window,
            normalize_audio, None)

    if tgt is None:
        tgt_examples_iter = None
    else:
        tgt_examples_iter = TextDataset.make_examples(
            tgt, tgt_seq_length_trunc, "tgt")

    # the second conjunct means nothing will be filtered at translation time
    # if there is no target data
    if use_filter_pred and tgt_examples_iter is not None:
        filter_pred = partial(
            filter_example, use_src_len=data_type == 'text',
            max_src_len=src_seq_len, max_tgt_len=tgt_seq_len
        )
    else:
        filter_pred = None

    dataset_cls = dataset_classes[data_type]
    dataset = dataset_cls(
        fields, src_examples_iter, tgt_examples_iter,
        dynamic_dict=dynamic_dict, filter_pred=filter_pred)
    return dataset


def _build_field_vocab(field, counter, **kwargs):
    specials = list(OrderedDict.fromkeys(
        tok for tok in [field.unk_token, field.pad_token, field.init_token,
                        field.eos_token]
        if tok is not None))
    field.vocab = field.vocab_cls(counter, specials=specials, **kwargs)


def build_vocab(train_dataset_files, fields, data_type, share_vocab,
                src_vocab_path, src_vocab_size, src_words_min_frequency,
                tgt_vocab_path, tgt_vocab_size, tgt_words_min_frequency):
    """
    Args:
        train_dataset_files: a list of train dataset pt file.
        fields (dict): fields to build vocab for.
        data_type: "text", "img" or "audio"?
        share_vocab(bool): share source and target vocabulary?
        src_vocab_path(string): Path to src vocabulary file.
        src_vocab_size(int): size of the source vocabulary.
        src_words_min_frequency(int): the minimum frequency needed to
                include a source word in the vocabulary.
        tgt_vocab_path(string): Path to tgt vocabulary file.
        tgt_vocab_size(int): size of the target vocabulary.
        tgt_words_min_frequency(int): the minimum frequency needed to
                include a target word in the vocabulary.

    Returns:
        Dict of Fields
    """
    # Prop src from field to get lower memory using when training with image
    if data_type == 'img' or data_type == 'audio':
        fields.pop("src")
    counters = {k: Counter() for k, v in chain.from_iterable(fields.values())}

    # Load vocabulary
    if src_vocab_path:
        src_vocab = load_vocabulary(src_vocab_path, "src")
        src_vocab_size = len(src_vocab)
        logger.info('Loaded source vocab has %d tokens.' % src_vocab_size)
        for i, token in enumerate(src_vocab):
            # keep the order of tokens specified in the vocab file by
            # adding them to the counter with decreasing counting values
            counters['src'][token] = src_vocab_size - i
    else:
        src_vocab = None

    if tgt_vocab_path:
        tgt_vocab = load_vocabulary(tgt_vocab_path, "tgt")
        tgt_vocab_size = len(tgt_vocab)
        logger.info('Loaded source vocab has %d tokens.' % tgt_vocab_size)
        for i, token in enumerate(tgt_vocab):
            counters['tgt'][token] = tgt_vocab_size - i
    else:
        tgt_vocab = None

    for i, path in enumerate(train_dataset_files):
        dataset = torch.load(path)
        logger.info(" * reloading %s." % path)
        for ex in dataset.examples:
            for name, field in chain.from_iterable(fields.values()):
                has_vocab = (name == 'src' and src_vocab) or \
                    (name == 'tgt' and tgt_vocab)
                if field.sequential and not has_vocab:
                    val = getattr(ex, name, None)
                    counters[name].update(val)

        # Drop the none-using from memory but keep the last
        if i < len(train_dataset_files) - 1:
            dataset.examples = None
            gc.collect()
            del dataset.examples
            gc.collect()
            del dataset
            gc.collect()

    for name, field in fields["tgt"]:
        _build_field_vocab(field, counters[name])
        logger.info(" * %s vocab size: %d." % (name, len(field.vocab)))
    if data_type == 'text':
        for name, field in fields["src"]:
            _build_field_vocab(field, counters[name])
            logger.info(" * %s vocab size: %d." % (name, len(field.vocab)))
        if share_vocab:
            # `tgt_vocab_size` is ignored when sharing vocabularies
            logger.info(" * merging src and tgt vocab...")
            src_field = fields['src'][0][1]
            tgt_field = fields['tgt'][0][1]
            _merge_field_vocabs(
                src_field, tgt_field, vocab_size=src_vocab_size,
                min_freq=src_words_min_frequency)
            logger.info(" * merged vocab size: %d." % len(src_field.vocab))

    return fields  # is the return necessary?


def _merge_field_vocabs(src_field, tgt_field, vocab_size, min_freq):
    # in the long run, shouldn't it be possible to do this by calling
    # build_vocab with both the src and tgt data?
    specials = [tgt_field.unk_token, tgt_field.pad_token,
                tgt_field.init_token, tgt_field.eos_token]
    merged = sum(
        [src_field.vocab.freqs, tgt_field.vocab.freqs], Counter()
    )
    merged_vocab = Vocab(
        merged, specials=specials,
        max_size=vocab_size, min_freq=min_freq
    )
    src_field.vocab = merged_vocab
    tgt_field.vocab = merged_vocab
    assert len(src_field.vocab) == len(tgt_field.vocab)


def load_vocabulary(vocab_path, tag):
    """
    Loads a vocabulary from the given path.
    :param vocabulary_path: path to load vocabulary from
    :param tag: tag for vocabulary (only used for logging)
    :return: vocabulary or None if path is null
    """
    logger.info("Loading {} vocabulary from {}".format(tag, vocab_path))

    if not os.path.exists(vocab_path):
        raise RuntimeError(
            "{} vocabulary not found at {}".format(tag, vocab_path))
    else:
        with codecs.open(vocab_path, 'r', 'utf-8') as f:
            return [line.strip().split()[0] for line in f if line.strip()]


class OrderedIterator(torchtext.data.Iterator):

    def create_batches(self):
        """ Create batches """
        if self.train:
            def _pool(data, random_shuffler):
                for p in torchtext.data.batch(data, self.batch_size * 100):
                    p_batch = torchtext.data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b

            self.batches = _pool(self.data(), self.random_shuffler)
        else:
            self.batches = []
            for b in torchtext.data.batch(self.data(), self.batch_size,
                                          self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))


class DatasetLazyIter(object):
    """ An Ordered Dataset Iterator, supporting multiple datasets,
        and lazy loading.

    Args:
        datasets (list): a list of datasets, which are lazily loaded.
        fields (dict): fields dict for the datasets.
        batch_size (int): batch size.
        batch_size_fn: custom batch process function.
        device: the GPU device.
        is_train (bool): train or valid?
    """

    def __init__(self, datasets, fields, batch_size, batch_size_fn,
                 device, is_train):
        self.datasets = datasets
        self.fields = fields
        self.batch_size = batch_size
        self.batch_size_fn = batch_size_fn
        self.device = device
        self.is_train = is_train

        self.cur_iter = self._next_dataset_iterator(datasets)
        # We have at least one dataset.
        assert self.cur_iter is not None

    def __iter__(self):
        dataset_iter = (d for d in self.datasets)
        while self.cur_iter is not None:
            for batch in self.cur_iter:
                yield batch
            self.cur_iter = self._next_dataset_iterator(dataset_iter)

    def __len__(self):
        # We return the len of cur_dataset, otherwise we need to load
        # all datasets to determine the real len, which loses the benefit
        # of lazy loading.
        assert self.cur_iter is not None
        return len(self.cur_iter)

    def _next_dataset_iterator(self, dataset_iter):
        try:
            # Drop the current dataset for decreasing memory
            if hasattr(self, "cur_dataset"):
                self.cur_dataset.examples = None
                gc.collect()
                del self.cur_dataset
                gc.collect()

            self.cur_dataset = next(dataset_iter)
        except StopIteration:
            return None

        # We clear `fields` when saving, restore when loading.
        self.cur_dataset.fields = self.fields

        # Sort batch by decreasing lengths of sentence required by pytorch.
        # sort=False means "Use dataset's sortkey instead of iterator's".
        return OrderedIterator(
            dataset=self.cur_dataset, batch_size=self.batch_size,
            batch_size_fn=self.batch_size_fn,
            device=self.device, train=self.is_train,
            sort=False, sort_within_batch=True,
            repeat=False)


def build_dataset_iter(datasets, fields, opt, is_train=True):
    """
    This returns user-defined train/validate data iterator for the trainer
    to iterate over. We implement simple ordered iterator strategy here,
    but more sophisticated strategy like curriculum learning is ok too.
    """
    batch_size = opt.batch_size if is_train else opt.valid_batch_size
    if is_train and opt.batch_type == "tokens":
        def batch_size_fn(new, count, sofar):
            """
            In token batching scheme, the number of sequences is limited
            such that the total number of src/tgt tokens (including padding)
            in a batch <= batch_size
            """
            # Maintains the longest src and tgt length in the current batch
            global max_src_in_batch, max_tgt_in_batch
            # Reset current longest length at a new batch (count=1)
            if count == 1:
                max_src_in_batch = 0
                max_tgt_in_batch = 0
            # Src: <bos> w1 ... wN <eos>
            max_src_in_batch = max(max_src_in_batch, len(new.src) + 2)
            # Tgt: w1 ... wN <eos>
            max_tgt_in_batch = max(max_tgt_in_batch, len(new.tgt) + 1)
            src_elements = count * max_src_in_batch
            tgt_elements = count * max_tgt_in_batch
            return max(src_elements, tgt_elements)
    else:
        batch_size_fn = None

    device = "cuda" if opt.gpu_ranks else "cpu"

    # the dataset and iterator want a fields object that has flat fields
    dataset_fields = dict(chain.from_iterable(fields.values()))
    return DatasetLazyIter(datasets, dataset_fields, batch_size, batch_size_fn,
                           device, is_train)


def lazily_load_dataset(corpus_type, opt):
    """
    Dataset generator. Don't do extra stuff here, like printing,
    because they will be postponed to the first loading time.

    Args:
        corpus_type: 'train' or 'valid'
    Returns:
        A list of dataset, the dataset(s) are lazily loaded.
    """
    assert corpus_type in ["train", "valid"]

    def _lazy_dataset_loader(pt_file, corpus_type):
        dataset = torch.load(pt_file)
        logger.info('Loading %s dataset from %s, number of examples: %d' %
                    (corpus_type, pt_file, len(dataset)))
        return dataset

    # Sort the glob output by file name (by increasing indexes).
    pts = sorted(glob.glob(opt.data + '.' + corpus_type + '.[0-9]*.pt'))
    if pts:
        for pt in pts:
            yield _lazy_dataset_loader(pt, corpus_type)
    else:
        # Only one inputters.*Dataset, simple!
        pt = opt.data + '.' + corpus_type + '.pt'
        yield _lazy_dataset_loader(pt, corpus_type)


def load_fields(dataset, opt, checkpoint):
    if isinstance(dataset, TextDataset):
        data_type = 'text'
    elif isinstance(dataset, AudioDataset):
        data_type = 'audio'
    else:
        data_type = 'img'
    if checkpoint is not None:
        logger.info('Loading vocab from checkpoint at %s.' % opt.train_from)
        vocab = checkpoint['vocab']
    else:
        vocab = torch.load(opt.data + '.vocab.pt')

    fields = load_fields_from_vocab(vocab, data_type)

    tgt_field = fields['tgt'][0][1]
    if data_type == 'text':
        src_field = fields['src'][0][1]
        logger.info(' * vocabulary size. source = %d; target = %d' %
                    (len(src_field.vocab), len(tgt_field.vocab)))
    else:
        logger.info(' * vocabulary size. target = %d' %
                    (len(tgt_field.vocab)))

    return fields
