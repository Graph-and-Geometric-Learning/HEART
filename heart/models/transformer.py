import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, config):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        return self.fc2(self.dropout(gelu(self.fc1(x))))


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()

        self.d_k, self.n_heads = config.hidden_size // config.num_attention_heads, config.num_attention_heads

        self.W_Q = nn.Linear(config.hidden_size, config.hidden_size)
        self.W_K = nn.Linear(config.hidden_size, config.hidden_size)
        self.W_V = nn.Linear(config.hidden_size, config.hidden_size)
        self.W_output = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
    
    def ScaledDotProductAttention(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(Q.size(-1)) # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
        attn = self.dropout(nn.Softmax(dim=-1)(scores))
        context = torch.matmul(attn, V)
        return context, attn

    def forward(self, Q, K, V, attn_mask):
        batch_size = Q.size(0)

        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) # attn_mask : [batch_size x n_heads x len_q x len_k]

        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context, attn = self.ScaledDotProductAttention(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k) # context: [batch_size x len_q x n_heads * d_v]
        return self.W_output(context)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super(TransformerBlock, self).__init__()
        self.self_attn = MultiHeadAttention(config)
        self.pos_ffn = PoswiseFeedForwardNet(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.norm_attn = nn.LayerNorm(config.hidden_size)
        self.norm_ffn = nn.LayerNorm(config.hidden_size)

    def forward(self, x, self_attn_mask):
        norm_x = self.norm_attn(x)
        x = x + self.dropout(self.self_attn(norm_x, norm_x, norm_x, self_attn_mask))

        norm_x = self.norm_ffn(x)
        x = x + self.dropout(self.pos_ffn(norm_x))
        return x