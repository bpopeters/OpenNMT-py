""" Audio encoder """
import math

import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from onmt.utils.rnn_factory import rnn_factory
from onmt.encoders.encoder import EncoderBase


class AudioEncoder(EncoderBase):
    """
    A simple encoder convolutional -> recurrent neural network for
    audio input.

    Args:
        num_layers (int): number of encoder layers.
        bidirectional (bool): bidirectional encoder.
        rnn_size (int): size of hidden states of the rnn.
        dropout (float): dropout probablity.
        sample_rate (float): input spec
        window_size (int): input spec

    """
    def __init__(self, rnn_type, enc_layers, dec_layers, brnn,
                 enc_rnn_size, dec_rnn_size, enc_pooling, dropout,
                 sample_rate, window_size):
        super(AudioEncoder, self).__init__()
        self.rnn_type = rnn_type
        self.dec_layers = dec_layers
        num_directions = 2 if brnn else 1
        self.num_directions = num_directions
        assert enc_rnn_size % num_directions == 0
        enc_rnn_size_real = enc_rnn_size // num_directions
        assert dec_rnn_size % num_directions == 0
        self.dec_rnn_size = dec_rnn_size
        dec_rnn_size_real = dec_rnn_size // num_directions
        self.dec_rnn_size_real = dec_rnn_size_real
        self.dec_rnn_size = dec_rnn_size
        input_size = int(math.floor((sample_rate * window_size) / 2) + 1)
        enc_pooling = enc_pooling.split(',')
        assert len(enc_pooling) == enc_layers or len(enc_pooling) == 1
        if len(enc_pooling) == 1:
            enc_pooling = enc_pooling * enc_layers
        enc_pooling = [int(p) for p in enc_pooling]

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        input_sizes = [input_size] + [enc_rnn_size] * (enc_layers - 1)
        self.rnns = nn.ModuleList(
            [rnn_factory(rnn_type,
                         input_size=input_size,
                         hidden_size=enc_rnn_size_real,
                         num_layers=1,
                         dropout=dropout,
                         bidirectional=brnn)
             for input_size in input_sizes]
        )
        self.pools = nn.ModuleList(
            [nn.MaxPool1d(enc_pool) for enc_pool in enc_pooling]
        )
        self.batchnorms = nn.ModuleList(
            [nn.BatchNorm1d(enc_rnn_size, affine=True)
             for i in range(enc_layers)]
        )

        self.W = nn.Linear(enc_rnn_size, dec_rnn_size, bias=False)

    @classmethod
    def from_opt(cls, opt, embeddings=None):
        if embeddings is not None:
            raise ValueError("Cannot use embeddings with AudioEncoder.")
        return cls(
            opt.rnn_type,
            opt.enc_layers,
            opt.dec_layers,
            opt.brnn,
            opt.enc_rnn_size,
            opt.dec_rnn_size,
            opt.audio_enc_pooling,
            opt.dropout,
            opt.sample_rate,
            opt.window_size)

    def forward(self, src, lengths=None):
        "See :obj:`onmt.encoders.encoder.EncoderBase.forward()`"

        batch_size, _, nfft, t = src.size()
        src = src.transpose(0, 1).transpose(0, 3).contiguous() \
                 .view(t, batch_size, nfft)
        orig_lengths = lengths
        lengths = lengths.view(-1).tolist()

        layers = zip(self.rnns, self.pools, self.batchnorms)
        for i, (rnn, pool, batchnorm) in enumerate(layers):
            stride = pool.kernel_size
            packed_emb = pack(src, lengths)
            memory_bank, _ = rnn(packed_emb)
            memory_bank = unpack(memory_bank)[0]
            t = memory_bank.size(0)
            memory_bank = pool(memory_bank.transpose(0, 2)).transpose(0, 2)
            lengths = [int(math.floor((length - stride) / stride + 1))
                       for length in lengths]
            src = memory_bank
            t, _, num_feat = src.size()
            src = batchnorm(src.contiguous().view(-1, num_feat))
            src = src.view(t, -1, num_feat)
            if self.dropout and i + 1 != len(self.rnns):
                src = self.dropout(src)

        memory_bank = memory_bank.contiguous().view(-1, memory_bank.size(2))
        memory_bank = self.W(memory_bank).view(-1, batch_size,
                                               self.dec_rnn_size)

        state = memory_bank.new_full((self.dec_layers * self.num_directions,
                                      batch_size, self.dec_rnn_size_real), 0)

        encoder_final = state, state if self.rnn_type == 'LSTM' else state
        return encoder_final, memory_bank, orig_lengths.new_tensor(lengths)
