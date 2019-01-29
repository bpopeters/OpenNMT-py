"""
GlobalAttention mechanism is the primary attention mechanism for RNNs.
It is also used for copy attention for both RNNs and CNNs.
"""

import torch
import torch.nn as nn

from onmt.modules.sparse_activations import Sparsemax
from onmt.utils.misc import aeq, sequence_mask


class MLPScorer(nn.Module):
    def __init__(self, dim):
        super(MLPScorer, self).__init__()
        self.linear_context = nn.Linear(dim, dim, bias=False)
        self.linear_query = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, 1, bias=False)

    def forward(self, h_t, h_s):
        """
        Args:
          h_t (`FloatTensor`): sequence of queries `[batch x tgt_len x dim]`
          h_s (`FloatTensor`): sequence of sources `[batch x src_len x dim]`

        Returns:
          :obj:`FloatTensor`:
           raw attention scores (unnormalized) for each src index
          `[batch x tgt_len x src_len]`
        """
        wq = self.linear_query(h_t).unsqueeze(2)
        wq = wq.expand(-1, -1, h_s.size(1), -1)

        uh = self.linear_context(h_s).unsqueeze(1)
        uh = uh.expand(-1, h_t.size(1), -1, -1)

        wquh = torch.tanh(wq + uh)

        return self.v(wquh).squeeze(3)


class DotScorer(nn.Module):
    def __init__(self):
        super(DotScorer, self).__init__()

    def forward(self, h_t, h_s):
        return torch.bmm(h_t, h_s.transpose(1, 2))


class GeneralScorer(nn.Module):
    def __init__(self, dim):
        super(GeneralScorer, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)

    def forward(self, h_t, h_s):
        h_t = self.linear_in(h_t)
        return torch.bmm(h_t, h_s.transpose(1, 2))


class GlobalAttention(nn.Module):
    """
    Global attention takes a matrix and a query vector. It
    then computes a parameterized convex combination of the matrix
    based on the input query.

    Constructs a unit mapping a query `q` of size `dim`
    and a source matrix `H` of size `n x dim`, to an output
    of size `dim`.


    .. mermaid::

       graph BT
          A[Query]
          subgraph RNN
            C[H 1]
            D[H 2]
            E[H N]
          end
          F[Attn]
          G[Output]
          A --> F
          C --> F
          D --> F
          E --> F
          C -.-> G
          D -.-> G
          E -.-> G
          F --> G

    All models compute the output as
    :math:`c = sum_{j=1}^{SeqLength} a_j H_j` where
    :math:`a_j` is the softmax of a score function.
    Then then apply a projection layer to [q, c].

    However they
    differ on how they compute the attention score.

    * Luong Attention (dot, general):
       * dot: :math:`score(H_j,q) = H_j^T q`
       * general: :math:`score(H_j, q) = H_j^T W_a q`


    * Bahdanau Attention (mlp):
       * :math:`score(H_j, q) = v_a^T tanh(W_a q + U_a h_j)`


    Args:
       dim (int): dimensionality of query and key
       coverage (bool): NOT IMPLEMENTED
       attn_type (str): type of attention to use, options [dot,general,mlp]
    """

    def __init__(self, dim, attn_type="dot", attn_func="softmax", **kwargs):
        super(GlobalAttention, self).__init__()

        if isinstance(attn_type, nn.Module):
            self.score = attn_type
        else:
            str2scorer = {
                "mlp": MLPScorer(dim),
                "dot": DotScorer(),
                "general": GeneralScorer(dim)
            }
            assert attn_type in str2scorer, "Please select a valid attn type."
            self.score = str2scorer[attn_type]

        if isinstance(attn_func, nn.Module):
            self.attn_func = attn_func
        else:
            str2func = {
                "softmax": nn.Softmax(dim=-1),
                "sparsemax": Sparsemax(dim=-1)
            }
            assert attn_func in str2func, "Please select a valid attn function"
            self.attn_func = str2func[attn_func]

        out_bias = attn_type == "mlp"  # should be done differently
        linear_out = nn.Linear(dim * 2, dim, bias=out_bias)
        out_activation = nn.Tanh() if attn_type != "mlp" else None

        if out_activation is not None:
            self.output_layer = nn.Sequential(linear_out, out_activation)
        else:
            self.output_layer = linear_out

    def forward(self, query, memory_bank, memory_lengths=None, coverage=None):
        """
        Args:
          query (`FloatTensor`): query vectors `[batch x tgt_len x dim]`
          memory_bank (`FloatTensor`): source vectors `[batch x src_len x dim]`
          memory_lengths (`LongTensor`): the source context lengths `[batch]`
          coverage: NOT IMPLEMENTED

        Returns:
          (`FloatTensor`, `FloatTensor`):

          * Computed vector `[tgt_len x batch x dim]`
          * Attention distribtutions for each query
             `[tgt_len x batch x src_len]`
        """
        one_step = query.dim() == 2
        if one_step:
            query = query.unsqueeze(1)

        src_batch, src_len, src_dim = memory_bank.size()
        tgt_batch, tgt_len, tgt_dim = query.size()
        aeq(src_batch, tgt_batch)
        aeq(src_dim, tgt_dim)

        if coverage is not None:
            cov_batch, cov_len = coverage.size()
            aeq(src_batch, cov_batch)
            aeq(src_len, cov_len)
            cover = coverage.view(-1).unsqueeze(1)
            memory_bank += self.linear_cover(cover).view_as(memory_bank)
            memory_bank = torch.tanh(memory_bank)

        align = self.score(query, memory_bank)

        if memory_lengths is not None:
            mask = sequence_mask(memory_lengths, max_len=align.size(-1))
            align.masked_fill_(1 - mask.unsqueeze(1), -float('inf'))

        # it should not be necessary to view align as a 2d tensor, but
        # something is broken with sparsemax and it cannot handle a 3d tensor
        align_vectors = self.attn_func(align.view(-1, src_len)).view_as(align)

        # each context vector c_t is the weighted average over source states
        c = torch.bmm(align_vectors, memory_bank)

        # concatenate
        concat_c = torch.cat([c, query], 2)
        attn_h = self.output_layer(concat_c)

        if one_step:
            attn_h = attn_h.squeeze(1)
            align_vectors = align_vectors.squeeze(1)
        else:
            attn_h = attn_h.transpose(0, 1).contiguous()
            align_vectors = align_vectors.transpose(0, 1).contiguous()

        return attn_h, align_vectors
