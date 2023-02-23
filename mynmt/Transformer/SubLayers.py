"""多头注意力层（自注意力层）、FFN、位置编码层"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head: int, d_model: int, d_k: int, d_v: int, dropout=0):
        """
        Multi-Head Attention module\n
        :returns: output, attn：加上残差的计算结果、注意力矩阵
        """
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        # 仅在最后一个维度（embedding size）上映射
        self.W_Q = nn.Sequential(nn.LayerNorm(d_model, eps=1e-6),
                                 nn.Linear(d_model, n_head * d_k), )
        self.W_K = nn.Sequential(nn.LayerNorm(d_model, eps=1e-6),
                                 nn.Linear(d_model, n_head * d_k), )
        self.W_V = nn.Sequential(nn.LayerNorm(d_model, eps=1e-6),
                                 nn.Linear(d_model, n_head * d_v))
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)
        # 降回d_model
        self.W = nn.Sequential(nn.Linear(n_head * d_v, d_model),
                               nn.Dropout(dropout), )

    def forward(self, Q, K, V, mask=None):
        residual, batch_size = Q, Q.size(0)  # 残差、batch_size
        # [b, lq, dv] -> [b, lq, n_head*dv] -> [b, lq, n_head, dv] -> [b, n_head, lq, dv]
        Q = self.W_Q(Q).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(batch_size, -1, self.n_head, self.d_v).transpose(1, 2)

        if mask is not None: mask = mask.unsqueeze(1)  # For head axis broadcasting.

        # output: [b, n_heads, len_q, d_v], attn: [b, n_heads, len_q(=len_k), len_k(=len_q)]
        output, attn = self.attention(Q, K, V, mask=mask)
        # [b, lq, n, dv] -> [b, lq, n*dv]。contiguous().view()与reshape()等价
        output = output.transpose(1, 2).reshape(batch_size, -1, self.n_head * self.d_v)
        # [b, len_q, d_model]
        output = self.W(output) + residual
        return output, attn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature):
        """
        Scaled Dot-Product Attention\n
        :returns: output, attn：自注意力计算结果、注意力矩阵
        """
        super().__init__()
        self.scale = temperature

    def forward(self, Q, K, V, mask=None):
        """(Q*K^T) / square(d)=> mask => softmax => *V"""
        attn = torch.matmul(Q / self.scale, K.transpose(-2, -1))  # [b, n_heads, len_q(=len_k), len_k(=len_q)]
        if mask is not None: attn = attn.masked_fill(mask == 0, -1e9)  # 找到mask中=1位置，并将attn对应位置用value填充
        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, V)
        return output, attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0):
        """
        A feed-forward-layer module\n
        :returns: result：加上残差的计算结果
        """
        super().__init__()
        self.ffn = nn.Sequential(nn.LayerNorm(d_in, eps=1e-6),
                                 nn.Linear(d_in, d_hid), nn.ReLU(inplace=True),
                                 nn.Linear(d_hid, d_in),
                                 nn.Dropout(dropout), )

    def forward(self, x):
        result = self.ffn(x) + x
        return result

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class PositionalEncoding(nn.Module):
    def __init__(self, n_position, d_hid):
        super().__init__()
        # register_buffer相当于创建常量参数，每次step的时候不会更新，其他都相同
        self.register_buffer('pos_table', _get_sinusoid_encoding_table(n_position, d_hid))

    def forward(self, x):
        # 位置编码表[1, n_position取句子长度个, d_hid=embedding_size]
        return x + self.pos_table[:, :x.size(1)].clone().detach().to(device)


def _get_sinusoid_encoding_table(n_position, d_hid):
    """
    构造一个位置编码表（静态的）\n
    :param n_position: 该编码表能使用的最大句长 -> position：单词位置
    :param d_hid: 单词最大表示维度(=embedding_size) -> i：位置编码向量中的第几维
    :return: sinusoid_table：位置编码表[1, n_position, d_hid]
    """

    def get_position_angle_vec(position):
        return [position / numpy.power(10000, 2 * (i // 2) / d_hid)
                for i in range(d_hid)]

    sinusoid_table = numpy.array([get_position_angle_vec(position)
                                  for position in range(n_position)])
    sinusoid_table[:, 0::2] = numpy.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = numpy.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table).unsqueeze(0)
