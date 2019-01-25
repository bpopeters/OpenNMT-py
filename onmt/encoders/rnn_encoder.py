import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from onmt.encoders.encoder import EncoderBase
from onmt.utils.rnn_factory import rnn_factory


class RNNEncoder(EncoderBase):
    """ A generic recurrent neural network encoder.

    Args:
       rnn_type (:obj:`str`):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional (bool) : use a bidirectional RNN
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       dropout (float) : dropout value for :obj:`nn.Dropout`
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """

    def __init__(self, rnn_type, bidirectional, num_layers,
                 hidden_size, dropout=0.0, embeddings=None,
                 use_bridge=False):
        super(RNNEncoder, self).__init__()
        assert embeddings is not None

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.embeddings = embeddings

        self.rnn, self.no_pack_padded_seq = rnn_factory(
            rnn_type,
            input_size=embeddings.embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional)

        # Initialize the bridge layer
        if use_bridge:
            num_states = 2 if rnn_type == "LSTM" else 1
            self.bridge = nn.ModuleList(
                [Bridge(hidden_size * num_layers) for i in range(num_states)]
            )
        else:
            self.bridge = None

    def forward(self, src, lengths=None):
        "See :obj:`EncoderBase.forward()`"
        self._check_args(src, lengths)

        emb = self.embeddings(src)

        packed_emb = emb
        if lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Tensor.
            lengths_list = lengths.view(-1).tolist()
            packed_emb = pack(emb, lengths_list)

        memory_bank, encoder_final = self.rnn(packed_emb)

        if lengths is not None and not self.no_pack_padded_seq:
            memory_bank = unpack(memory_bank)[0]

        if self.bridge is not None:
            if isinstance(encoder_final, tuple):  # LSTM
                encoder_final = tuple(layer(enc_state)
                                      for enc_state, layer
                                      in zip(encoder_final, self.bridge))
            else:
                encoder_final = self.bridge[0](encoder_final)
        return encoder_final, memory_bank, lengths


class Bridge(nn.Module):
    def __init__(self, bridge_size):
        super(Bridge, self).__init__()
        self.dense_layer = nn.Sequential(
            nn.Linear(bridge_size, bridge_size), nn.ReLU()
        )

    def forward(self, enc_state):
        unbottled_size = enc_state.size()
        total_dim = self.dense_layer[0].in_features
        out = self.dense_layer(enc_state.view(-1, total_dim))
        return out.view(unbottled_size)
