import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention
    """

    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn


class MultiHeadAttentionOne(nn.Module):
    """
    Multi-Head Attention module with shared projection
    """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.5):
        super(MultiHeadAttentionOne, self).__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qkvs = nn.Linear(d_model, n_head * d_k, bias=False)
        nn.init.normal_(self.w_qkvs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k=None, v=None, query_input=False):
        H, W = q.size()[-2:]
        if not k and not v:
            q = q.view(q.size()[0], q.size()[1], -1)  # [bz, c, h, w]
            q = q.permute(0, 2, 1).contiguous()  # [bz, hw, c]
            
            d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
            sz_b, len_q, _ = q.size()

            residual = q
            q = self.w_qkvs(q).view(sz_b, len_q, n_head, d_k)

            q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # [(n*b), lq, dk]

            output, attn, log_attn = self.attention(q, q, q)

        else:
            k = k.view(k.size()[0], k.size()[1], -1)  # [bz, c, h, w]
            v = v.view(v.size()[0], v.size()[1], -1)  # [bz, c, h, w]
            q = q.view(q.size()[0], q.size()[1], -1)  # [bz, c, h, w]

            k = k.permute(0, 2, 1).contiguous()  # [bz, hw, c]
            v = v.permute(0, 2, 1).contiguous()  # [bz, hw, c]
            q = q.permute(0, 2, 1).contiguous()  # [bz, hw, c]


            d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
            sz_b, len_q, _ = q.size()
            sz_b, len_k, _ = k.size()
            sz_b, len_v, _ = v.size()

            residual = q
            q = self.w_qkvs(q).view(sz_b, len_q, n_head, d_k)
            k = self.w_qkvs(k).view(sz_b, len_k, n_head, d_k)
            v = self.w_qkvs(v).view(sz_b, len_v, n_head, d_v)

            q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # [(n*b), lq, dk]
            k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # [(n*b), lk, dk]
            v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # [(n*b), lv, dv]

            output, attn, log_attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # [b, lq, (n*dv)]

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        output = output.view(sz_b, -1, H, W) # B, C (d_model), H, W

        return output



class SelfAttention(nn.Module):
    def __init__(self, embed_dim=[64,64,128,256,512], num_heads=1):
        super().__init__()

        # self.att0 = MultiHeadAttentionOne(num_heads, embed_dim[0], embed_dim[0], embed_dim[0])
        # self.att1 = MultiHeadAttentionOne(num_heads, embed_dim[1], embed_dim[1], embed_dim[1])
        # self.att2 = MultiHeadAttentionOne(num_heads, embed_dim[2], embed_dim[2], embed_dim[2])
        # self.att3 = MultiHeadAttentionOne(num_heads, embed_dim[3], embed_dim[3], embed_dim[3])
        self.att4 = MultiHeadAttentionOne(num_heads, embed_dim[4], embed_dim[4], embed_dim[4])

        self.att = [
                    None, # self.att0, 
                    None, # self.att1, 
                    None, # self.att2, 
                    None, # self.att3, 
                    self.att4
                    ]

    def forward(self, x):
        # x in the shape of {0:[*5], 1:[*5], -1:[*5], s_0:[*5], ..., ...}

        for frame_id in x.keys():
            if not isinstance(frame_id, str): # not right view
                for idx in range(5):
                    if idx in [4]:
                        x[frame_id][idx] = self.att[idx](x[frame_id][idx])
        return x
