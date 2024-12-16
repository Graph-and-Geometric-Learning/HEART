import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import softmax
from torch_scatter import scatter_mean, scatter_sum


class DotAttnConv(nn.Module):
    def __init__(self, in_channels, out_channels, n_heads=1, n_max_visits=15, temp=1.):
        super(DotAttnConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_heads, self.temp = n_heads, temp

        self.pos_encoding = nn.Embedding(n_max_visits, in_channels)
        self.W_q = nn.Linear(in_channels, out_channels, bias=False)
        self.W_k = nn.Linear(in_channels, out_channels, bias=False)
        self.W_v = nn.Linear(in_channels, out_channels, bias=False)
        self.W_out = nn.Linear(out_channels, out_channels, bias=False)
        self.ln = nn.LayerNorm(out_channels)

    def forward(self, x, edge_index, visit_pos):
        # x: [N, in_channels], edge_index: [2, E]
        N, device = x.size(0), x.device
        isolated_nodes_mask = ~torch.isin(torch.arange(N).to(x.device), edge_index[1].unique())
        isolated_nodes = isolated_nodes_mask.nonzero(as_tuple=False).squeeze()

        pos_encoding = self.pos_encoding(visit_pos)
        h_q, h_k, h_v = self.W_q(x + pos_encoding), self.W_k(x + pos_encoding), self.W_k(x)
        h_q, h_k, h_v = h_q.reshape(N, self.n_heads, -1), h_k.reshape(N, self.n_heads, -1), h_v.reshape(N, self.n_heads, -1)
        
        attn_scores = torch.sum(h_q[edge_index[0]] * h_k[edge_index[1]], dim=-1) / self.temp  # [N_edges, n_heads]
        dst_nodes = torch.cat([edge_index[1] + N*i for i in range(self.n_heads)], dim=0).to(device)
        attn_scores = softmax(attn_scores.reshape(-1), dst_nodes, num_nodes=N * self.n_heads).unsqueeze(dim=-1)  # [N_edges * n_heads, 1]

        # aggregation
        h_v = h_v.permute(1, 0, 2).reshape(N*self.n_heads, -1)
        src_nodes = torch.cat([edge_index[0] + N*i for i in range(self.n_heads)], dim=0).to(device)
        out = scatter_sum(src=h_v[src_nodes] * attn_scores, index=dst_nodes, dim_size=N * self.n_heads, dim=0)
        out = out.reshape(self.n_heads, N, -1).permute(1, 0, 2).reshape(N, -1)

        out = self.W_out(self.ln(out)) + x
        out[isolated_nodes] = x[isolated_nodes]
        return out

