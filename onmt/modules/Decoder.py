from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable

import onmt
from onmt.Utils import aeq


class InputFeedRNNDecoderLayers(nn.Module):
    """
    The recurrent layers of a Decoder that uses input feeding.
    """
    def __init__(self, rnn_type, input_size, rnn_size,
                 num_layers, attn_type, dropout):
        assert rnn_type in ['LSTM', 'GRU']
        assert rnn_type != "SRU", "SRU doesn't support input feed. " \
            "Please set -input_feed 0"
        super(InputFeedRNNDecoderLayers, self).__init__()
        if rnn_type == "LSTM":
            stacked_cell = onmt.modules.StackedLSTM
        else:
            stacked_cell = onmt.modules.StackedGRU
        self.rnn = stacked_cell(
            num_layers, input_size, rnn_size, dropout)

        # todo: this does not use the coverage option
        self.attn = onmt.modules.GlobalAttention(
            rnn_size, attn_type=attn_type)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, emb, context, state, **kwargs):
        outputs = []
        attns = {"std": []}
        output = state.input_feed.squeeze(0)
        hidden = state.hidden
        # coverage = state.coverage.squeeze(0) \
        #     if state.coverage is not None else None

        for emb_t in emb.split(1):
            emb_t = emb_t.squeeze(0)
            # concatenate hidden state with input at every time step
            emb_t = torch.cat([emb_t, output], 1)

            rnn_output, hidden = self.rnn(emb_t, hidden)
            attn_output, attn = self.attn(
                rnn_output, context.transpose(0, 1))
            '''
            if self.context_gate is not None:
                output = self.context_gate(
                    emb_t, rnn_output, attn_output
                )
                output = self.dropout(output)
            else:
            '''
            output = self.dropout(attn_output)
            outputs.append(output)
            attns["std"].append(attn)

            '''
            # Update the coverage attention.
            if self._coverage:
                coverage = coverage + attn \
                    if coverage is not None else attn
                attns["coverage"] += [coverage]

            # Run the forward pass of the copy attention layer.
            if self._copy:
                _, copy_attn = self.copy_attn(output,
                                              context.transpose(0, 1))
                attns["copy"] += [copy_attn]
            '''
        outputs = torch.stack(outputs)
        for k in attns:
            attns[k] = torch.stack(attns[k])

        state.update_state(hidden, outputs[-1].unsqueeze(0), None)
        return outputs, state, attns

    def init_decoder_state(self, context, enc_hidden, **kwargs):
        """
        context (FloatTensor): src_len x batch x rnn_size
        enc_hidden (FloatTensor or tuple thereof)
        returns:
            The initial RNNDecoderState
        """
        return RNNDecoderState(context, self.rnn.hidden_size, enc_hidden)


class RNNDecoderLayers(nn.Module):
    """
    The recurrent layers of a Decoder that does not do input feed
    """
    def __init__(self, rnn_type, embedding_dim, rnn_size,
                 num_layers, attn_type, dropout):
        assert rnn_type in ['LSTM', 'GRU']
        super(RNNDecoderLayers, self).__init__()
        self.rnn = getattr(nn, rnn_type)(
            input_size=embedding_dim,
            hidden_size=rnn_size,
            num_layers=num_layers,
            dropout=dropout)
        # todo: this does not use the coverage option
        self.attn = onmt.modules.GlobalAttention(
            rnn_size, attn_type=attn_type)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, emb, context, state, **kwargs):
        """
        emb: tgt_len x batch x embedding_dim
        context: src_len x batch x rnn_size
        state: an RNNDecoderState object
        """
        assert isinstance(state, RNNDecoderState)
        attns = dict()
        rnn_output, hidden = self.rnn(emb, state.hidden)
        attn_outputs, attn_scores = self.attn(
            rnn_output.transpose(0, 1).contiguous(),
            context.transpose(0, 1)
        )
        attns["std"] = attn_scores
        outputs = self.dropout(attn_outputs)
        state.update_state(hidden, outputs[-1].unsqueeze(0), None)
        return outputs, state, attns

    def init_decoder_state(self, context, enc_hidden, **kwargs):
        """
        context (FloatTensor): src_len x batch x rnn_size
        enc_hidden (FloatTensor or tuple thereof)
        returns:
            The initial RNNDecoderState
        """
        return RNNDecoderState(context, self.rnn.hidden_size, enc_hidden)


class Decoder(nn.Module):
    """
    Decoder neural network. The decoder consists of three basic
    parts: an embedding matrix for mapping input tokens to fixed-dimensional
    vectors, a layered unit which processes these embedded tokens (an RNN with
    attention, a transformer, or a convolutional network), and an
    output layer (generally weight matrix + softmax) which generates a
    probability distribution over the target vocabulary.
    """
    def __init__(
            self, decoder_type, rnn_type, num_layers, rnn_size, global_attn,
            coverage_attn, copy_attn, input_feed, context_gate, dropout,
            embeddings, cnn_kernel_width):

        # emb_size is the size of the input to the decoder layers unless
        # using a model with input feeding
        emb_size = embeddings.embedding_size

        super(Decoder, self).__init__()
        self.embeddings = embeddings

        # possible decoders: a self-attentional transformer, convolutional,
        #                    a stacked RNN, a normal RNN
        if decoder_type == "transformer":
            self.decoder = onmt.modules.TransformerDecoder(
                num_layers, rnn_size, global_attn, copy_attn, dropout)
        elif decoder_type == "cnn":
            self.decoder = onmt.modules.CNNDecoder(
                num_layers, rnn_size, emb_size, global_attn,
                copy_attn, cnn_kernel_width, dropout)
        elif input_feed:
            self.decoder = InputFeedRNNDecoderLayers(
                rnn_type, emb_size + rnn_size, rnn_size,
                num_layers, global_attn, dropout)
        else:
            self.decoder = RNNDecoderLayers(
                rnn_type, emb_size, rnn_size,
                num_layers, global_attn, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, context, state, **kwargs):
        """
        Args:
            tgt (LongTensor):  tgt_len x batch x tgt_nfeats
            context (FloatTensor): src_len x batch x rnn_size
            state (DecoderState)
        Returns:
            outputs: (len x batch x rnn_size)
            final_states: an object of the same form as above
            attns: Dictionary of (src_len x batch)
        """
        # CHECKS
        tgt_len, tgt_batch, _ = tgt.size()
        context_len, context_batch, _ = context.size()
        aeq(tgt_batch, context_batch)
        # END CHECKS
        if isinstance(self.decoder, onmt.modules.TransformerDecoder) or \
                isinstance(self.decoder, onmt.modules.CNNDecoder):
            if state.previous_input:
                tgt = torch.cat([state.previous_input, tgt], 0)

        emb = self.embeddings(tgt)
        outputs, state, attns = self.decoder(
            emb=emb,
            tgt=tgt,
            context=context,
            state=state,
            padding_idx=self.embeddings.word_padding_idx)
        # todo: output layer stuff
        return outputs, state, attns

    def init_decoder_state(self, **kwargs):
        """
        Dispatches to the init_decoder_state of this module's intermediate
        layers. Different model types require different sorts of values
        to initialize the decoder state. Therefore all arguments are passed
        as keywords.
        """
        return self.decoder.init_decoder_state(**kwargs)


class DecoderState(object):
    """
    DecoderState is a base class for models, used during translation
    for storing translation states.
    """
    def detach(self):
        """
        Detaches all Variables from the graph
        that created it, making it a leaf.
        """
        for h in self._all:
            if h is not None:
                h.detach_()

    def beam_update(self, idx, positions, beam_size):
        """ Update when beam advances. """
        for e in self._all:
            a, br, d = e.size()
            sent_states = e.view(a, beam_size, br // beam_size, d)[:, :, idx]
            sent_states.data.copy_(
                sent_states.data.index_select(1, positions))


class RNNDecoderState(DecoderState):
    def __init__(self, context, hidden_size, rnnstate):
        """
        Args:
            context (FloatTensor): output from the encoder of size
                                   len x batch x rnn_size.
            hidden_size (int): the size of hidden layer of the decoder.
            rnnstate (Variable): final hidden state from the encoder.
                transformed to shape: layers x batch x (directions*dim).
            input_feed (FloatTensor): output from last layer of the decoder.
            coverage (FloatTensor): coverage output from the decoder.
        """
        if not isinstance(rnnstate, tuple):
            self.hidden = (rnnstate,)
        else:
            self.hidden = rnnstate
        self.coverage = None

        # Init the input feed.
        batch_size = context.size(1)
        h_size = (batch_size, hidden_size)
        self.input_feed = Variable(context.data.new(*h_size).zero_(),
                                   requires_grad=False).unsqueeze(0)

    @property
    def _all(self):
        return self.hidden + (self.input_feed,)

    def update_state(self, rnnstate, input_feed, coverage):
        if not isinstance(rnnstate, tuple):
            self.hidden = (rnnstate,)
        else:
            self.hidden = rnnstate
        self.input_feed = input_feed
        self.coverage = coverage

    def repeat_beam_size_times(self, beam_size):
        """ Repeat beam_size times along batch dimension. """
        vars = [Variable(e.data.repeat(1, beam_size, 1), volatile=True)
                for e in self._all]
        self.hidden = tuple(vars[:-1])
        self.input_feed = vars[-1]
