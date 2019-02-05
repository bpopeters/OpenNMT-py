import torch.nn as nn
import onmt.models


def rnn_factory(rnn_type, **kwargs):
    """ rnn factory, Use pytorch version when available. """
    if rnn_type == "SRU":
        rnn = onmt.models.sru.SRU(**kwargs)
    else:
        rnn = getattr(nn, rnn_type)(**kwargs)
    return rnn
