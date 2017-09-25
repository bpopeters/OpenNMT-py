from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable

import onmt
from onmt.Utils import aeq


class RNNDecoderBase(nn.Module):
    """
    RNN decoder base class.
    """
    def __init__(self, rnn_type, bidirectional_encoder, num_layers,
                 hidden_size, attn_type, coverage_attn, context_gate,
                 copy_attn, dropout, embeddings):
        super(RNNDecoderBase, self).__init__()

        # Basic attributes.
        self.decoder_type = 'rnn'
        self.bidirectional_encoder = bidirectional_encoder
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embeddings = embeddings
        self.dropout = nn.Dropout(dropout)

        # Build the RNN.
        self.rnn = self._build_rnn(rnn_type, self._input_size, hidden_size,
                                   num_layers, dropout)

        # Set up the context gate.
        self.context_gate = None
        if context_gate is not None:
            self.context_gate = onmt.modules.ContextGateFactory(
                context_gate, self._input_size,
                hidden_size, hidden_size, hidden_size
            )

        # Set up the standard attention.
        self._coverage = coverage_attn
        self.attn = onmt.modules.GlobalAttention(
            hidden_size, coverage=coverage_attn,
            attn_type=attn_type
        )

        # Set up a separated copy attention layer, if needed.
        self._copy = False
        if copy_attn:
            self.copy_attn = onmt.modules.GlobalAttention(
                hidden_size, attn_type=attn_type
            )
            self._copy = True

    def forward(self, input, context, state):
        """
        Forward through the decoder.
        Args:
            input (LongTensor): a sequence of input tokens tensors
                                of size (len x batch x nfeats).
            context (FloatTensor): output(tensor sequence) from the encoder
                        RNN of size (src_len x batch x hidden_size).
            state (FloatTensor): hidden state from the encoder RNN for
                                 initializing the decoder.
        Returns:
            outputs (FloatTensor): a Tensor sequence of output from the decoder
                                   of shape (len x batch x hidden_size).
            state (FloatTensor): final hidden state from the decoder.
            attns (dict of (str, FloatTensor)): a dictionary of different
                                type of attention Tensor from the decoder
                                of shape (src_len x batch).
        """
        # Args Check
        assert isinstance(state, RNNDecoderState)
        input_len, input_batch, _ = input.size()
        contxt_len, contxt_batch, _ = context.size()
        aeq(input_batch, contxt_batch)
        # END Args Check

        # Run the forward pass of the RNN.
        hidden, outputs, attns, coverage = \
            self._run_forward_pass(input, context, state)

        # Update the state with the result.
        final_output = outputs[-1]
        state.update_state(hidden, final_output.unsqueeze(0),
                           coverage.unsqueeze(0)
                           if coverage is not None else None)

        # Concatenates sequence of tensors along a new dimension.
        outputs = torch.stack(outputs)
        for k in attns:
            attns[k] = torch.stack(attns[k])

        return outputs, state, attns

    def _fix_enc_hidden(self, h):
        """
        The encoder hidden is  (layers*directions) x batch x dim.
        We need to convert it to layers x batch x (directions*dim).
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def init_decoder_state(self, src, context, enc_hidden):
        if isinstance(enc_hidden, tuple):  # GRU
            return RNNDecoderState(context, self.hidden_size,
                                   tuple([self._fix_enc_hidden(enc_hidden[i])
                                         for i in range(len(enc_hidden))]))
        else:  # LSTM
            return RNNDecoderState(context, self.hidden_size,
                                   self._fix_enc_hidden(enc_hidden))


class StdRNNDecoder(RNNDecoderBase):
    """
    Stardard RNN decoder, with Attention.
    Currently no 'coverage_attn' and 'copy_attn' support.
    """
    def _run_forward_pass(self, input, context, state):
        """
        Private helper for running the specific RNN forward pass.
        Must be overriden by all subclasses.
        Args:
            input (LongTensor): a sequence of input tokens tensors
                                of size (len x batch x nfeats).
            context (FloatTensor): output(tensor sequence) from the encoder
                        RNN of size (src_len x batch x hidden_size).
            state (FloatTensor): hidden state from the encoder RNN for
                                 initializing the decoder.
        Returns:
            hidden (Variable): final hidden state from the decoder.
            outputs ([FloatTensor]): an array of output of every time
                                     step from the decoder.
            attns (dict of (str, [FloatTensor]): a dictionary of different
                            type of attention Tensor array of every time
                            step from the decoder.
            coverage (FloatTensor, optional): coverage from the decoder.
        """
        assert not self._copy  # TODO, no support yet.
        assert not self._coverage  # TODO, no support yet.

        # Initialize local and return variables.
        outputs = []
        attns = {"std": []}
        coverage = None

        emb = self.embeddings(input)

        # Run the forward pass of the RNN.
        rnn_output, hidden = self.rnn(emb, state.hidden)
        # Result Check
        input_len, input_batch, _ = input.size()
        output_len, output_batch, _ = rnn_output.size()
        aeq(input_len, output_len)
        aeq(input_batch, output_batch)
        # END Result Check

        # Calculate the attention.
        attn_outputs, attn_scores = self.attn(
            rnn_output.transpose(0, 1).contiguous(),  # (output_len, batch, d)
            context.transpose(0, 1)                   # (contxt_len, batch, d)
        )
        attns["std"] = attn_scores

        # Calculate the context gate.
        if self.context_gate is not None:
            outputs = self.context_gate(
                emb.view(-1, emb.size(2)),
                rnn_output.view(-1, rnn_output.size(2)),
                attn_outputs.view(-1, attn_outputs.size(2))
            )
            outputs = outputs.view(input_len, input_batch, self.hidden_size)
            outputs = self.dropout(outputs)
        else:
            outputs = self.dropout(attn_outputs)    # (input_len, batch, d)

        # Return result.
        return hidden, outputs, attns, coverage

    def _build_rnn(self, rnn_type, input_size,
                   hidden_size, num_layers, dropout):
        """
        Private helper for building standard decoder RNN.
        """
        # Use pytorch version when available.
        if rnn_type == "SRU":
            return onmt.modules.SRU(
                    input_size, hidden_size,
                    num_layers=num_layers,
                    dropout=dropout)

        return getattr(nn, rnn_type)(
            input_size, hidden_size,
            num_layers=num_layers,
            dropout=dropout)

    @property
    def _input_size(self):
        """
        Private helper returning the number of expected features.
        """
        return self.embeddings.embedding_size


class InputFeedRNNDecoder(RNNDecoderBase):
    """
    Stardard RNN decoder, with Input Feed and Attention.
    """
    def _run_forward_pass(self, input, context, state):
        """
        See StdRNNDecoder._run_forward_pass() for description
        of arguments and return values.
        """
        # Additional args check.
        output = state.input_feed.squeeze(0)
        output_batch, _ = output.size()
        input_len, input_batch, _ = input.size()
        aeq(input_batch, output_batch)
        # END Additional args check.

        # Initialize local and return variables.
        outputs = []
        attns = {"std": []}
        if self._copy:
            attns["copy"] = []
        if self._coverage:
            attns["coverage"] = []

        emb = self.embeddings(input)
        assert emb.dim() == 3  # len x batch x embedding_dim

        hidden = state.hidden
        coverage = state.coverage.squeeze(0) \
            if state.coverage is not None else None

        # Input feed concatenates hidden state with
        # input at every time step.
        for i, emb_t in enumerate(emb.split(1)):
            emb_t = emb_t.squeeze(0)
            emb_t = torch.cat([emb_t, output], 1)

            rnn_output, hidden = self.rnn(emb_t, hidden)
            attn_output, attn = self.attn(rnn_output,
                                          context.transpose(0, 1))
            if self.context_gate is not None:
                output = self.context_gate(
                    emb_t, rnn_output, attn_output
                )
                output = self.dropout(output)
            else:
                output = self.dropout(attn_output)
            outputs += [output]
            attns["std"] += [attn]

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

        # Return result.
        return hidden, outputs, attns, coverage

    def _build_rnn(self, rnn_type, input_size,
                   hidden_size, num_layers, dropout):
        assert not rnn_type == "SRU", "SRU doesn't support input feed! " \
                "Please set -input_feed 0!"
        if rnn_type == "LSTM":
            stacked_cell = onmt.modules.StackedLSTM
        else:
            stacked_cell = onmt.modules.StackedGRU
        return stacked_cell(num_layers, input_size,
                            hidden_size, dropout)

    @property
    def _input_size(self):
        """
        Using input feed by concatenating input with attention vectors.
        """
        return self.embeddings.embedding_size + self.hidden_size


class InputFeedRNNDecoderLayers(nn.Module):
    """
    The recurrent layers of a Decoder that uses input feeding.
    """
    def __init__(self, rnn_type, embedding_dim, rnn_size,
                 num_layers, attn_type, dropout):
        assert rnn_type in ['LSTM', 'GRU']
        super(InputFeedRNNDecoderLayers, self).__init__()
        self.rnn = None  # gotta make this thing
        # I think the input size will also be different...
        '''
        self.rnn = getattr(nn, rnn_type)(
            input_size=embedding_dim,
            hidden_size=rnn_size,
            num_layers=num_layers,
            dropout=dropout)
        '''
        # todo: this does not use the coverage option
        self.attn = onmt.modules.GlobalAttention(
            rnn_size, attn_type=attn_type)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, emb, context, state, **kwargs):
        """
        
        """
        output = state.input_feed.squeeze(0)
        hidden = state.hidden
        # coverage = state.coverage.squeeze(0) \
        #     if state.coverage is not None else None

        # Input feed concatenates hidden state with
        # input at every time step.
        for emb_t in emb.split(1):
            emb_t = emb_t.squeeze(0)
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
            outputs += [output]
            attns["std"] += [attn]

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

        # does doing the stte like this still work?
        state.update_state(hidden, outputs[-1].unsqueeze(0), None)
        return outputs, state, attns


class RNNDecoderLayers(nn.Module):
    """
    The recurrent layers of a Decoder. This one doesn't do input feeding,
    at least not yet.
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
        # this thing is making the RNNDecoderState with the wrong args
        state.update_state(hidden, outputs[-1].unsqueeze(0), None)
        return outputs, state, attns


class Decoder(nn.Module):
    """
    Decoder recurrent neural network. The decoder consists of three basic
    parts: an embedding matrix for mapping input tokens to fixed-dimensional
    vectors, a unit which processes these embedded tokens (an RNN with
    attention, a transformer, or a convolutional network), and an
    output layer (generally weight matrix + softmax) which generates a
    probability distribution over the target vocabulary.
    """

    def __init__(
        self, decoder_type, rnn_type, num_layers, rnn_size, global_attn,
        coverage_attn, copy_attn, input_feed, context_gate, dropout,
        embeddings, cnn_kernel_width):
        # goal: obliterate attributes
        '''
        self.decoder_type = decoder_type
        self.layers = num_layers
        self._coverage = coverage_attn
        self.hidden_size = rnn_size
        input_size = embeddings.embedding_size
        if self.input_feed:
            input_size += rnn_size
        '''
        self.bidirectional_encoder = False  # total kludge

        input_size = embeddings.embedding_size  # different if input_feed

        super(Decoder, self).__init__()
        self.embeddings = embeddings

        # pad_id = embeddings.padding_idx
        # possible decoders: a self-attentional transformer
        #                    a stacked RNN

        # standard RNN Decoder layers
        # (the attention happens in here as well)
        self.decoder = RNNDecoderLayers(
            rnn_type, input_size, rnn_size,
            num_layers, global_attn, dropout)

        self.dropout = nn.Dropout(dropout)

        # copy and coverage attention are not currently implemented.
        # It's not immediately clear to me where they would go if they
        # were.
        
        '''
        if self.decoder_type == "transformer":
            self.transformer = nn.ModuleList(
                [onmt.modules.TransformerDecoder(self.hidden_size, opt, pad_id)
                 for _ in range(opt.dec_layers)])
        else:
            if self.input_feed:
                if opt.rnn_type == "LSTM":
                    stackedCell = onmt.modules.StackedLSTM
                else:
                    stackedCell = onmt.modules.StackedGRU
                self.rnn = stackedCell(opt.dec_layers, input_size,
                                       opt.rnn_size, opt.dropout)
            else:
                self.rnn = getattr(nn, opt.rnn_type)(
                     input_size, opt.rnn_size,
                     num_layers=opt.dec_layers,
                     dropout=opt.dropout
                )
            self.context_gate = None
            if opt.context_gate is not None:
                self.context_gate = ContextGateFactory(
                    opt.context_gate, input_size,
                    opt.rnn_size, opt.rnn_size,
                    opt.rnn_size
                )
        '''

    def forward(self, input, context, state, **kwargs):
        """
        Forward through the decoder.
        Args:
            input (LongTensor):  (len x batch x nfeats) -- Input tokens
            src (LongTensor): not present anymore; what happened?
            context:  (src_len x batch x rnn_size)  -- Memory bank
            state: an object initializing the decoder.
        Returns:
            outputs: (len x batch x rnn_size)
            final_states: an object of the same form as above
            attns: Dictionary of (src_len x batch)
        """
        # CHECKS
        tgt_len, n_batch, _ = input.size()
        #s_len, n_batch_, _ = src.size()
        context_len, context_batch, _ = context.size()
        aeq(tgt_batch, context_batch)
        # aeq(s_len, s_len_)
        # END CHECKS
        '''
        if self.decoder_type == "transformer":
            if state.previous_input:
                input = torch.cat([state.previous_input, input], 0)
        '''

        emb = self.embeddings(input)
        outputs, state, attns = self.decoder(emb, context, state)
        return outputs, state, attns

    def _fix_enc_hidden(self, h):
        """
        The encoder hidden is  (layers*directions) x batch x dim.
        We need to convert it to layers x batch x (directions*dim).
        """
        # It makes a lot more sense for this to be part of the encoder
        # (that would also make it easier to create new methods of
        # brnn-merging
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def init_decoder_state(self, src, context, enc_hidden):
        # is the case labeling backwards here?
        if isinstance(enc_hidden, tuple):  # GRU
            return RNNDecoderState(context, self.decoder.rnn.hidden_size,
                                   tuple([self._fix_enc_hidden(e_h)
                                         for e_h in enc_hidden]))
        else:  # LSTM
            return RNNDecoderState(context, self.decoder.rnn.hidden_size,
                                   self._fix_enc_hidden(enc_hidden))


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
            sentStates = e.view(a, beam_size, br // beam_size, d)[:, :, idx]
            sentStates.data.copy_(
                sentStates.data.index_select(1, positions))


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
