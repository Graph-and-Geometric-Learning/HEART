import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum

from heart.models.transformer import MultiHeadAttention, TransformerBlock


class EdgeModule(nn.Module):
    def __init__(self, config):
        super(EdgeModule, self).__init__()

        self.n_types = 4 + 1  # +1 for [CLS]

        self.left_transform = nn.Parameter(torch.zeros(self.n_types, config.hidden_size, config.hidden_size))
        self.right_transform = nn.Parameter(torch.zeros(self.n_types, config.hidden_size, config.hidden_size))
        self.output = nn.Linear(config.hidden_size * 2, config.edge_hidden_size)

        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_uniform_(self.left_transform)
        nn.init.xavier_uniform_(self.right_transform)

    def forward(self, token_embs, token_types):
        # token_embs: [batch_size, seq_len, hidden_size], token_types: [batch_size, seq_len]
        batch_size, seq_len, hidden_size = token_embs.size()

        # encode token according to its type
        left_trans = self.left_transform[token_types]  # [batch_size, seq_len, hidden_size, hidden_size]
        right_trans = self.right_transform[token_types]  # [batch_size, seq_len, hidden_size, hidden_size]

        left_embs = einsum(token_embs, left_trans, 'b l d, b l m d -> b l m')  # [batch_size, seq_len, hidden_size]
        right_embs = einsum(token_embs, right_trans, 'b l d, b l m d -> b l m')  # [batch_size, seq_len, hidden_size]

        edge_embs = torch.cat((left_embs.unsqueeze(dim=2).repeat(1, 1, seq_len, 1), 
                               right_embs.unsqueeze(dim=1).repeat(1, seq_len, 1, 1)), dim=-1)
        return self.output(edge_embs)


class MultiHeadEdgeAttention(MultiHeadAttention):
    def __init__(self, config):
        super().__init__(config)

        self.d_edge = config.edge_hidden_size
        self.W_output = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.W_K_edge = nn.Linear(config.edge_hidden_size, config.edge_hidden_size)
        self.W_edge = nn.Linear(config.edge_hidden_size, 1)
        self.W_edge_output = nn.Linear(self.d_edge * self.n_heads, config.hidden_size)

    def forward(self, Q, K, V, attn_mask, edge_embs):
        batch_size, n_tokens = Q.size(0), Q.size(1)

        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]

        k_s_edge = self.W_K_edge(edge_embs).view(batch_size, n_tokens, n_tokens, -1)
        edge_bias = self.W_edge(edge_embs).view(batch_size, 1, n_tokens, n_tokens)
        edge_bias = edge_bias * (2 ** -0.5)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) # attn_mask : [batch_size x n_heads x len_q x len_k]

        # bias attention scores with edge representation
        scores = torch.matmul(q_s, k_s.transpose(-1, -2)) * ((2 * q_s.size(-1)) ** -0.5)
        scores = scores + edge_bias
        scores.masked_fill_(attn_mask, -1e9)  # [batch_size, n_heads, n_tokens, n_tokens]
        attn = self.dropout(nn.Softmax(dim=-1)(scores))
        
        # add edge context into aggregated context
        context = torch.matmul(attn, v_s)
        edge_context = einsum(attn, k_s_edge, 'b h n m, b n m d -> b h n d')  # [batch_size, n_tokens, n_heads, d_k]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k) # context: [batch_size x len_q x n_heads * d_v]
        edge_context = edge_context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_edge)
        edge_context = self.W_edge_output(edge_context)
        return self.W_output(torch.cat([context, edge_context], dim=-1))


class EdgeTransformerBlock(TransformerBlock):
    def __init__(self, config):
        super().__init__(config)

        self.self_attn = MultiHeadEdgeAttention(config)
        self.norm_edge = nn.LayerNorm(config.edge_hidden_size)

    def forward(self, x, edge_embs, self_attn_mask):
        norm_x = self.norm_attn(x)
        norm_edge_embs = self.norm_edge(edge_embs)
        x = x + self.dropout(self.self_attn(norm_x, norm_x, norm_x, self_attn_mask, norm_edge_embs))

        norm_x = self.norm_ffn(x)
        x = x + self.dropout(self.pos_ffn(norm_x))
        return x