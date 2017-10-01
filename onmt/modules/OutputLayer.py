import torch.nn as nn
import torch.nn.functional as F


class OutputLayer(nn.Module):
    """
    A basic output layer combining a weight matrix and a log softmax.
    The functionality here could be implemented with nn.Sequential;
    however, it is kept separate as this code may be extended to allow
    for other sorts of output layers.
    """
    def __init__(self, rnn_size, vocab_size):
        super(OutputLayer, self).__init__()
        self.output_weights = nn.Linear(rnn_size, vocab_size)

    def forward(self, layer):
        """
        layer: (tgt_len * batch) x rnn_size
        returns: (tgt_len * batch) x vocab_size
        """
        return F.log_softmax(self.output_weights(layer))
