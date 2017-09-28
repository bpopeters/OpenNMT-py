"""
Implementation of "Convolutional Sequence to Sequence Learning"
"""
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

import onmt.modules
from onmt.modules.WeightNorm import WeightNormConv2d
from onmt.modules.Decoder import DecoderState
from onmt.Utils import aeq


SCALE_WEIGHT = 0.5 ** 0.5


def shape_transform(x):
    """ Tranform the size of the tensors to fit for conv input. """
    return torch.unsqueeze(torch.transpose(x, 1, 2), 3)


class GatedConv(nn.Module):
    def __init__(self, input_size, width=3, dropout=0.2, nopad=False):
        super(GatedConv, self).__init__()
        self.conv = WeightNormConv2d(input_size, 2 * input_size,
                                     kernel_size=(width, 1), stride=(1, 1),
                                     padding=(width // 2 * (1 - nopad), 0))
        init.xavier_uniform(self.conv.weight, gain=(4 * (1 - dropout))**0.5)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_var, hidden=None):
        x_var = self.dropout(x_var)
        x_var = self.conv(x_var)
        out, gate = x_var.split(int(x_var.size(1) / 2), 1)
        out = out * F.sigmoid(gate)
        return out


class StackedCNN(nn.Module):
    def __init__(self, num_layers, input_size, cnn_kernel_width=3,
                 dropout=0.2):
        super(StackedCNN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                GatedConv(input_size, cnn_kernel_width, dropout))

    def forward(self, x, hidden=None):
        for conv in self.layers:
            x = x + conv(x)
            x *= SCALE_WEIGHT
        return x


class CNNEncoder(nn.Module):
    """
    Convolutional Encoder
    """
    def __init__(self, input_size, hidden_size, dropout,
                 num_layers, kernel_width):
        super(CNNEncoder, self).__init__()

        self.linear = nn.Linear(input_size, hidden_size)
        self.cnn = StackedCNN(num_layers, hidden_size,
                              kernel_width, dropout)

    def forward(self, emb, **kwargs):
        """
        emb (FloatTensor): src_len x batch x emb_size
        """
        emb = emb.transpose(0, 1).contiguous()
        emb_reshape = emb.view(emb.size(0) * emb.size(1), -1)
        emb_remap = self.linear(emb_reshape)
        emb_remap = emb_remap.view(emb.size(0), emb.size(1), -1)
        emb_remap = shape_transform(emb_remap)
        out = self.cnn(emb_remap)

        return emb_remap.squeeze(3).transpose(0, 1).contiguous(), \
            out.squeeze(3).transpose(0, 1).contiguous()


class CNNDecoder(nn.Module):
    """
    Decoder built on CNN, which consists of residual convolutional layers,
    with ConvMultiStepAttention.
    """
    def __init__(self, num_layers, input_size, hidden_size, attn_type,
                 copy_attn, cnn_kernel_width, dropout):
        assert not copy_attn, "copy attention is not implemented yet"
        self.cnn_kernel_width = cnn_kernel_width
        super(CNNDecoder, self).__init__()

        # Build the CNN.
        self.linear = nn.Linear(input_size, hidden_size)
        self.conv_layers = nn.ModuleList(
            [GatedConv(hidden_size, cnn_kernel_width, dropout, True)
             for i in range(num_layers)]
        )

        self.attn_layers = nn.ModuleList(
            [onmt.modules.ConvMultiStepAttention(hidden_size)
             for i in range(num_layers)]
        )

    def forward(self, emb, tgt, context, state, **kwargs):
        """
        Args:
            emb (FloatTensor): tgt_len x batch x emb_size
            context (FloatTensor): output(tensor sequence) from the encoder
                        CNN of size (src_len x batch x hidden_size).
            state (FloatTensor): hidden state from the encoder CNN for
                                 initializing the decoder.
        Returns:
            outputs (FloatTensor): tgt_len x batch x hidden_size
            state (FloatTensor): final hidden state from the decoder.
            attns (dict of (str, FloatTensor)): a dictionary of different
                                type of attention Tensor from the decoder
                                of shape (src_len x batch).
        """
        # CHECKS
        assert isinstance(state, CNNDecoderState)
        input_len, input_batch, _ = emb.size()
        context_len, context_batch, _ = context.size()
        aeq(input_batch, context_batch)
        # END CHECKS

        attns = {"std": []}

        tgt_emb = emb.transpose(0, 1).contiguous()
        # The output of CNNEncoder.
        src_context_t = context.transpose(0, 1).contiguous()
        # The combination of output of CNNEncoder and source embeddings.
        src_context_c = state.init_src.transpose(0, 1).contiguous()

        # Run the forward pass of the CNNDecoder.
        emb_reshape = tgt_emb.contiguous().view(
            tgt_emb.size(0) * tgt_emb.size(1), -1)
        linear_out = self.linear(emb_reshape)
        x = linear_out.view(tgt_emb.size(0), tgt_emb.size(1), -1)
        x = shape_transform(x)

        pad = Variable(torch.zeros(x.size(0), x.size(1),
                                   self.cnn_kernel_width - 1, 1))
        pad = pad.type_as(x)
        base_target_emb = x

        for conv, attention in zip(self.conv_layers, self.attn_layers):
            new_target_input = torch.cat([pad, x], 2)
            out = conv(new_target_input)
            c, attn = attention(base_target_emb, out,
                                src_context_t, src_context_c)
            x = (x + (c + out) * SCALE_WEIGHT) * SCALE_WEIGHT
        output = x.squeeze(3).transpose(1, 2)

        # Process the result and update the attentions.
        outputs = output.transpose(0, 1).contiguous()
        if state.previous_input is not None:
            outputs = outputs[state.previous_input.size(0):]
            attn = attn[:, state.previous_input.size(0):].squeeze()
            attn = torch.stack([attn])
        attns["std"] = attn

        # Update the state.
        state.update_state(tgt)

        return outputs, state, attns

    def init_decoder_state(self, context, enc_hidden, **kwargs):
        return CNNDecoderState(context, enc_hidden)


class CNNDecoderState(DecoderState):
    def __init__(self, context, enc_hidden):
        self.init_src = (context + enc_hidden) * SCALE_WEIGHT
        self.previous_input = None

    @property
    def _all(self):
        """
        Contains attributes that need to be updated in self.beam_update().
        """
        return (self.previous_input,)

    def update_state(self, input):
        """ Called for every decoder forward pass. """
        self.previous_input = input

    def repeat_beam_size_times(self, beam_size):
        """ Repeat beam_size times along batch dimension. """
        self.init_src = Variable(
            self.init_src.data.repeat(1, beam_size, 1), volatile=True)
