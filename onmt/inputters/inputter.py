# -*- coding: utf-8 -*-
import glob

from collections import Counter, defaultdict, OrderedDict
from itertools import count, cycle, chain
from functools import partial

import torch
import torchtext.data
from torchtext.vocab import Vocab

from onmt.inputters.dataset_base import PAD_WORD, BOS_WORD, EOS_WORD
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


class Field(torchtext.data.Field):
    def extend_vocab(self, *args, **kwargs):
        """Construct the Vocab object for this field from one or more datasets.
        Arguments:
            Positional arguments: Dataset objects or other iterable data
                sources from which to construct the Vocab object that
                represents the set of possible values for this field. If
                a Dataset object is provided, all columns corresponding
                to this field are used; individual columns can also be
                provided directly.
            Remaining keyword arguments: Passed to the constructor of Vocab.
        """
        counter = Counter()
        sources = []
        for arg in args:
            if isinstance(arg, torchtext.data.Dataset):
                sources += [getattr(arg, name) for name, field in
                            arg.fields.items() if field is self]
            else:
                sources.append(arg)
        for data in sources:
            for x in data:
                if not self.sequential:
                    x = [x]
                try:
                    counter.update(x)
                except TypeError:
                    counter.update(chain.from_iterable(x))
        specials = list(OrderedDict.fromkeys(
            tok for tok in [self.unk_token, self.pad_token, self.init_token,
                            self.eos_token]
            if tok is not None))
        if hasattr(self, 'vocab'):
            self.vocab.extend(
                self.vocab_cls(counter, specials=specials, **kwargs))
        else:
            self.vocab = self.vocab_cls(counter, specials=specials, **kwargs)


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


def get_fields(
    src_data_type, n_src_features, n_tgt_features, share_vocab=False
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
    fields = dict()

    if share_vocab:
        shared_field = Field(
            init_token=BOS_WORD,
            eos_token=EOS_WORD,
            pad_token=PAD_WORD,
            include_lengths=True)
    else:
        shared_field = None

    if src_data_type == 'text':
        if shared_field is not None:
            fields["src"] = shared_field
        else:
            fields["src"] = Field(pad_token=PAD_WORD, include_lengths=True)
        for i in range(n_src_features):
            fields["src_feat_" + str(i)] = Field(pad_token=PAD_WORD)
    elif src_data_type == 'img':
        fields["src"] = Field(
            use_vocab=False, dtype=torch.float,
            postprocessing=make_img, sequential=False)
    else:
        fields["src"] = Field(
            use_vocab=False, dtype=torch.float,
            postprocessing=make_audio, sequential=False)

    if src_data_type == 'audio':
        # only audio has src_lengths
        fields["src_lengths"] = Field(
            use_vocab=False, dtype=torch.long, sequential=False)
    else:
        # everything except audio has src_map and alignment
        fields["src_map"] = Field(
            use_vocab=False, dtype=torch.float,
            postprocessing=make_src, sequential=False)

        fields["alignment"] = Field(
            use_vocab=False, dtype=torch.long,
            postprocessing=make_tgt, sequential=False)

    # below this: things defined no matter what the data source type is
    if shared_field is not None:
        fields["tgt"] = shared_field
    else:
        fields["tgt"] = Field(
            init_token=BOS_WORD, eos_token=EOS_WORD, pad_token=PAD_WORD)

    for i in range(n_tgt_features):
        fields["tgt_feat_" + str(i)] = Field(
            init_token=BOS_WORD, eos_token=EOS_WORD, pad_token=PAD_WORD)

    fields["indices"] = Field(
        use_vocab=False, dtype=torch.long, sequential=False)

    return fields


def load_fields_from_vocab(vocab, data_type="text"):
    """
    vocab: a list of (field name, torchtext.vocab.Vocab) pairs
    data_type: text, img, or audio
    returns: a dictionary whose keys are the field names and whose values
             are field objects with the vocab set to the corresponding vocab
             object from the input.
    """
    vocab = dict(vocab)
    n_src_features = len(collect_features(vocab, 'src'))
    n_tgt_features = len(collect_features(vocab, 'tgt'))
    fields = get_fields(data_type, n_src_features, n_tgt_features)
    for k, v in vocab.items():
        fields[k].vocab = v
    return fields


def save_fields_to_vocab(fields):
    """
    fields: a dictionary whose keys are field names and whose values are
            Field objects
    returns: a list of (field name, vocab) pairs for the fields that have a
             vocabulary
    """
    return [(k, f.vocab) for k, f in fields.items()
            if f is not None and 'vocab' in f.__dict__]


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


def collect_features(fields, side="src"):
    assert side in ["src", "tgt"]
    feats = []
    for j in count():
        key = side + "_feat_" + str(j)
        if key not in fields:
            break
        feats.append(key)
    return feats


def filter_example(ex, use_src_len=True, use_tgt_len=True,
                   min_src_len=1, max_src_len=float('inf'),
                   min_tgt_len=1, max_tgt_len=float('inf')):
    """
    A generalized function for filtering examples based on the length of their
    src or tgt values. Rather than being used by itself as the filter_pred
    argument to a dataset, it should be partially evaluated with everything
    specified except the value of the example.
    """
    return (not use_src_len or min_src_len <= len(ex.src) <= max_src_len) and \
        (not use_tgt_len or min_tgt_len <= len(ex.tgt) <= max_tgt_len)


def build_dataset(fields, data_type, src,
                  src_dir=None, tgt=None,
                  src_seq_len=50, tgt_seq_len=50,
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
    """
    dataset_paths: a list containing the locations of datasets
    fields (dict): fields dict for the datasets.
    batch_size (int): batch size.
    batch_size_fn: custom batch process function.
    device: the GPU device.
    is_train (bool): train or valid?
    """

    def __init__(self, dataset_paths, fields, batch_size, batch_size_fn,
                 device, is_train):
        self._paths = dataset_paths
        self.fields = fields
        self.batch_size = batch_size
        self.batch_size_fn = batch_size_fn
        self.device = device
        self.is_train = is_train

    def __iter__(self):
        paths = cycle(self._paths) if self.is_train else self._paths
        for path in paths:
            cur_dataset = torch.load(path)
            logger.info('Loading dataset from %s, number of examples: %d' %
                        (path, len(cur_dataset)))
            cur_dataset.fields = self.fields
            cur_iter = OrderedIterator(
                dataset=cur_dataset,
                batch_size=self.batch_size,
                batch_size_fn=self.batch_size_fn,
                device=self.device,
                train=self.is_train,
                sort=False,
                sort_within_batch=True,
                repeat=False
            )
            for batch in cur_iter:
                yield batch

            cur_dataset.examples = None
            gc.collect()
            del cur_dataset
            gc.collect()


def max_tok_len(new, count, sofar):
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


def build_dataset_iter(corpus_type, fields, opt, is_train=True):
    """
    This returns user-defined train/validate data iterator for the trainer
    to iterate over. We implement simple ordered iterator strategy here,
    but more sophisticated strategy like curriculum learning is ok too.
    """
    dataset_paths = sorted(glob.glob(opt.data + '.' + corpus_type + '*.pt'))
    batch_size = opt.batch_size if is_train else opt.valid_batch_size
    batch_fn = max_tok_len if is_train and opt.batch_type == "tokens" else None

    device = "cuda" if opt.gpu_ranks else "cpu"

    return DatasetLazyIter(dataset_paths, fields, batch_size, batch_fn,
                           device, is_train)


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

    ex_fields = dataset.examples[0].__dict__
    fields = {k: f for k, f in fields.items() if k in ex_fields}

    if data_type == 'text':
        logger.info(' * vocabulary size. source = %d; target = %d' %
                    (len(fields['src'].vocab), len(fields['tgt'].vocab)))
    else:
        logger.info(' * vocabulary size. target = %d' %
                    (len(fields['tgt'].vocab)))

    return fields


def _collect_report_features(fields):
    src_features = collect_features(fields, side='src')
    tgt_features = collect_features(fields, side='tgt')

    return src_features, tgt_features
