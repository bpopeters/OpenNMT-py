"""
Implementation of "Attention is All You Need"
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

import onmt
from onmt.modules.Decoder import DecoderState
from onmt.Utils import aeq


MAX_SIZE = 5000


class PositionwiseFeedForward(nn.Module):
    """ A two-layer Feed-Forward-Network."""
    def __init__(self, size, hidden_size, dropout=0.1):
        """
        Args:
            size(int): the size of input for the first-layer of the FFN.
            hidden_size(int): the hidden layer size of the second-layer
                              of the FNN.
            droput(float): dropout probability(0-1.0).
        """
        super(PositionwiseFeedForward, self).__init__()
        self.layers = nn.Sequential(
            onmt.modules.BottleLinear(size, hidden_size),
            nn.ReLU(),
            onmt.modules.BottleLinear(hidden_size, size),
            nn.Dropout(dropout)
        )
        self.layer_norm = onmt.modules.BottleLayerNorm(size)

    def forward(self, x):
        residual = x
        output = self.layers(x)
        return self.layer_norm(output + residual)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, size, dropout,
                 head_count=8, hidden_size=2048):
        """
        Args:
            size(int): the dimension of keys/values/queries in
                       MultiHeadedAttention, also the input size of
                       the first-layer of the PositionwiseFeedForward.
            droput(float): dropout probability(0-1.0).
            head_count(int): the number of head for MultiHeadedAttention.
            hidden_size(int): the second-layer of the PositionwiseFeedForward.
        """
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = onmt.modules.MultiHeadedAttention(
            head_count, size, p=dropout)
        self.feed_forward = PositionwiseFeedForward(
            size, hidden_size, dropout)

    def forward(self, input, mask):
        mid, _ = self.self_attn(input, input, input, mask=mask)
        out = self.feed_forward(mid)
        return out


class TransformerEncoder(nn.Module):
    """
    Transformer encoder from "Attention is All You Need"
    """
    def __init__(self, hidden_size, dropout, num_layers, padding_idx):
        self.padding_idx = padding_idx
        super(TransformerEncoder, self).__init__()
        self.transformer = nn.ModuleList(
                [TransformerEncoderLayer(hidden_size, dropout)
                 for i in range(num_layers)])

    def forward(self, emb, input, **kwargs):
        """
        emb: src_len x batch x embedding dimension
        input (LongTensor): src_len x batch x nfeat
        """
        out = emb.transpose(0, 1).contiguous()
        words = input[:, :, 0].transpose(0, 1)
        # CHECKS
        out_batch, out_len, _ = out.size()
        w_batch, w_len = words.size()
        aeq(out_batch, w_batch)
        aeq(out_len, w_len)
        # END CHECKS

        mask = words.data.eq(self.padding_idx).unsqueeze(1) \
            .expand(w_batch, w_len, w_len)

        for tf in self.transformer:
            out = tf(out, mask)

        # Is Variable(emb.data) different from emb?
        return Variable(emb.data), out.transpose(0, 1).contiguous()


class TransformerDecoderLayer(nn.Module):
    def __init__(self, size, dropout, head_count=8, hidden_size=2048):
        """
        Args:
            size(int): the dimension of keys/values/queries in
                       MultiHeadedAttention, also the input size of
                       the first-layer of the PositionwiseFeedForward.
            droput(float): dropout probability(0-1.0).
            head_count(int): the number of head for MultiHeadedAttention.
            hidden_size(int): the second-layer of the PositionwiseFeedForward.
        """
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = onmt.modules.MultiHeadedAttention(
                head_count, size, p=dropout)
        self.context_attn = onmt.modules.MultiHeadedAttention(
                head_count, size, p=dropout)
        self.feed_forward = PositionwiseFeedForward(
            size, hidden_size, dropout)
        self.dropout = dropout
        mask = self._get_attn_subsequent_mask(MAX_SIZE)
        # Register self.mask as a buffer in TransformerDecoderLayer, so
        # it gets TransformerDecoderLayer's cuda behavior automatically.
        self.register_buffer('mask', mask)

    def forward(self, input, context, src_pad_mask, tgt_pad_mask):
        # CHECKS
        input_batch, input_len, _ = input.size()
        context_batch, context_len, _ = context.size()
        aeq(input_batch, context_batch)

        src_batch, t_len, s_len = src_pad_mask.size()
        tgt_batch, t_len_, t_len__ = tgt_pad_mask.size()
        aeq(input_batch, context_batch, src_batch, tgt_batch)
        aeq(t_len, t_len_, t_len__, input_len)
        aeq(s_len, context_len)
        # END CHECKS

        dec_mask = torch.gt(tgt_pad_mask + self.mask[:, :tgt_pad_mask.size(1),
                            :tgt_pad_mask.size(1)]
                            .expand_as(tgt_pad_mask), 0)
        query, attn = self.self_attn(input, input, input, mask=dec_mask)
        mid, attn = self.context_attn(context, context, query,
                                      mask=src_pad_mask)
        output = self.feed_forward(mid)

        # CHECKS
        output_batch, output_len, _ = output.size()
        aeq(input_len, output_len)
        aeq(context_batch, output_batch)

        n_batch_, t_len_, s_len_ = attn.size()
        aeq(input_batch, n_batch_)
        aeq(context_len, s_len_)
        aeq(input_len, t_len_)
        # END CHECKS

        return output, attn

    def _get_attn_subsequent_mask(self, size):
        ''' Get an attention mask to avoid using the subsequent info.'''
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        subsequent_mask = torch.from_numpy(subsequent_mask)
        return subsequent_mask


class TransformerDecoder(nn.Module):
    """"""
    def __init__(self, num_layers, hidden_size,
                 attn_type, copy_attn, dropout):
        super(TransformerDecoder, self).__init__()

        self.transformer_layers = nn.ModuleList(
            [TransformerDecoderLayer(hidden_size, dropout)
             for _ in range(num_layers)])

        assert not copy_attn, "copy attention is not implemented yet"

    def forward(self, emb, tgt, context, state, padding_idx, **kwargs):
        """
        Args:
            emb (FloatTensor): tgt_len x batch x embedding_dim
            tgt (LongTensor): tgt_len x batch x tgt_nfeat
            context (FloatTensor): output(tensor sequence) from the encoder
                                of size (src_len x batch x hidden_size).
            state (TransformerDecoderState): hidden state from the encoder
                RNN for initializing the decoder.
            padding_idx:
        Returns:
            outputs (FloatTensor): a Tensor sequence of output from the decoder
                                   of shape (len x batch x hidden_size).
            state (TransformerDecoderState):
                                    final hidden state from the decoder.
            attns (dict of (str, FloatTensor)): a dictionary of different
                                type of attention Tensor from the decoder
                                of shape (src_len x batch).
        """
        # CHECKS
        assert isinstance(state, TransformerDecoderState)
        input_len, input_batch, _ = emb.size()
        context_len, context_batch, _ = context.size()
        aeq(input_batch, context_batch)

        src = state.src
        src_words = src[:, :, 0].transpose(0, 1)
        tgt_words = tgt[:, :, 0].transpose(0, 1)
        src_batch, src_len = src_words.size()
        tgt_batch, tgt_len = tgt_words.size()
        aeq(input_batch, context_batch, src_batch, tgt_batch)
        aeq(context_len, src_len)
        # aeq(input_len, tgt_len)
        # END CHECKS

        attns = {"std": []}

        output = emb.transpose(0, 1).contiguous()
        src_context = context.transpose(0, 1).contiguous()

        src_pad_mask = src_words.data.eq(padding_idx).unsqueeze(1) \
            .expand(src_batch, tgt_len, src_len)
        tgt_pad_mask = tgt_words.data.eq(padding_idx).unsqueeze(1) \
            .expand(tgt_batch, tgt_len, tgt_len)

        for transformer_layer in self.transformer_layers:
            output, attn = transformer_layer(output, src_context,
                                             src_pad_mask, tgt_pad_mask)

        # Process the result and update the attentions.
        outputs = output.transpose(0, 1).contiguous()
        if state.previous_input is not None:
            outputs = outputs[state.previous_input.size(0):]
            attn = attn[:, state.previous_input.size(0):].squeeze()
            attn = torch.stack([attn])  # what does this do?
        attns["std"] = attn

        # Update the state.
        state.update_state(tgt)

        return outputs, state, attns

    def init_decoder_state(self, src, **kwargs):
        return TransformerDecoderState(src)


class TransformerDecoderState(DecoderState):
    def __init__(self, src):
        """
        Args:
            src (FloatTensor): a sequence of source words tensors
                    with optional feature tensors, of size (len x batch).
        """
        self.src = src
        self.previous_input = None

    @property
    def _all(self):
        """
        Contains attributes that need to be updated in self.beam_update().
        """
        return (self.previous_input, self.src)

    def update_state(self, input):
        """ Called for every decoder forward pass. """
        self.previous_input = input

    def repeat_beam_size_times(self, beam_size):
        """ Repeat beam_size times along batch dimension. """
        self.src = Variable(self.src.data.repeat(1, beam_size, 1),
                            volatile=True)
