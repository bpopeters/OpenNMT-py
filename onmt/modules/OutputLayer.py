import torch.nn as nn


class OutputLayer(nn.Module):
    """
    Basically a wrapper around the nn.Sequential generator
    """
    def __init__(self, rnn_size, vocab_size, share_decoder_embeddings=False):
        super(OutputLayer, self).__init__()
        self.softmax_layer = nn.Sequential(
            nn.Linear(rnn_size, vocab_size),
            nn.LogSoftmax()
        )
        '''
        if share_decoder_embeddings:
            self.softmax_layer[0].weight = decoder.embeddings.word_lut.weight
        '''

    def forward(self, layer):
        return self.softmax_layer(layer)
