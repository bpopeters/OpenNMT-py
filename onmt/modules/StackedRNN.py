import torch
import torch.nn as nn


class StackedRNN(nn.Module):
    def __init__(self, rnn_type, num_layers, input_size, rnn_size, dropout):
        assert rnn_type in ['LSTM', 'GRU']
        cell = nn.LSTMCell if rnn_type == 'LSTM' else nn.GRUCell
        self.hidden_size = rnn_size
        super(StackedRNN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(cell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input, hidden):
        """
        input (FloatTensor):
            batch x input_feed_input_size
        hidden (tuple of FloatTensor):
            each element is layers x batch x rnn_size
        returns:
            output (FloatTensor): batch x rnn_size
            out_hidden: hidden state of same form as input hidden state
        """
        hidden_states = []
        output = input

        for i, layer in enumerate(self.layers):
            if isinstance(self, StackedLSTM):
                hidden_i = tuple(h[i] for h in hidden)
            else:
                hidden_i = hidden[0][i]
            h_1_i = layer(output, hidden_i)
            if isinstance(self, StackedLSTM):
                output = h_1_i[0]
            else:
                output = h_1_i
            if i + 1 != len(self.layers):
                output = self.dropout(output)
            hidden_states.append(h_1_i)

        if isinstance(self, StackedLSTM):
            out_hidden = tuple(torch.stack(h) for h in zip(*hidden_states))
        else:
            out_hidden = (torch.stack(hidden_states),)

        return output, out_hidden


class StackedLSTM(StackedRNN):
    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedLSTM, self).__init__(
            'LSTM', num_layers, input_size, rnn_size, dropout)


class StackedGRU(StackedRNN):
    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedGRU, self).__init__(
            'GRU', num_layers, input_size, rnn_size, dropout)
