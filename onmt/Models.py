import torch.nn as nn


class NMTModel(nn.Module):
    """
    The encoder + decoder Neural Machine Translation Model.
    """
    def __init__(self, encoder, decoder, multigpu=False):
        """
        Args:
            encoder(*Encoder): the various encoder.
            decoder(*Decoder): the various decoder.
            multigpu(bool): run parellel on multi-GPU?
        """
        self.multigpu = multigpu
        assert not multigpu  # Not yet supported on multi-gpu
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, lengths, dec_state=None):
        """
        Args:
            src (LongTensor): src_len x batch x src_nfeat
            tgt (LongTensor): tgt_len x batch x tgt_nfeat
            lengths (LongTensor): batch: the length of each source
                                         sentence in the batch
            dec_state (DecoderState or None)
        Returns:
            outputs (FloatTensor): (tgt_len x batch x hidden_size)
            attns (dict): keys denote attn types, values are
                          src_len x batch (FloatTensor)
            dec_hidden (DecoderState)
        """
        tgt = tgt[:-1]  # exclude last target from inputs
        enc_hidden, context = self.encoder(src, lengths)
        if dec_state is None:
            dec_state = self.decoder.init_decoder_state(
                src=src,
                context=context,
                enc_hidden=enc_hidden
            )
        out, dec_state, attns = self.decoder(
            tgt=tgt,
            src=src,
            context=context,
            state=dec_state
        )
        return out, attns, dec_state
