"""编码器（编码器层）、解码器（解码器层）"""
import torch.nn as nn
from Transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward, PositionalEncoding


class Encoder(nn.Module):
    def __init__(self, n_layer, dict_size, embedding_size, padding_idx,
                 d_model, d_hid, n_head, d_k, d_v, n_position=200, dropout=0, scale_emb=False):
        """
        编码器\n
        :param n_layer: 编码器层数
        :param dict_size: 字典长度（word2index字典长度）
        :param embedding_size: embedding size
        :param padding_idx: 指定哪个值（即<pad>对应的序号）映射为全0
        :param d_model: Q、K、V矩阵最深一维长度
        :param d_hid: 前馈层中间宽度
        :param n_head: 多头注意力机制头的数量
        :param d_k: K的宽度（Q的和K一样）
        :param d_v: V的宽度
        :param n_position: 构造的位置编码表，需要能使用的最大句长
        :param scale_emb: 是否将embedding得到的矩阵除以sqrt(d_model)
        """
        super().__init__()
        self.d_model = d_model
        self.scale_emb = scale_emb
        # embedding + 位置编码 + Norm
        # padding_idx指定哪个值（即<pad>对应的序号）映射为全0
        self.embedding = nn.Embedding(dict_size, embedding_size, padding_idx=padding_idx)
        self.in_layers = nn.Sequential(
            PositionalEncoding(n_position, embedding_size),  # 加入位置编码
            nn.Dropout(dropout),
            nn.LayerNorm(d_model, eps=1e-6), )
        # n_layers层编码器
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, d_hid, n_head, d_k, d_v, dropout)
                                             for _ in range(n_layer)])

    def forward(self, src_seq, src_mask, return_attention=False):
        enc_output = self.embedding(src_seq)
        if self.scale_emb: enc_output *= self.d_model ** 0.5
        enc_output = self.in_layers(enc_output)
        # 进入编码器层
        enc_slf_attn_list = []
        for enc_layer in self.encoder_layers:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            if return_attention: enc_slf_attn_list.append(enc_slf_attn)
        if return_attention: return enc_output, enc_slf_attn_list
        return enc_output


class Decoder(nn.Module):
    def __init__(self, n_layer, dict_size, embedding_size, padding_idx,
                 d_model, d_hid, n_head, d_k, d_v, n_position=200, dropout=0, scale_emb=False):
        """
        编码器\n
        :param n_layer: 编码器层数
        :param dict_size: 字典长度（word2index字典长度）
        :param embedding_size: embedding size
        :param padding_idx: 指定哪个值（即<pad>对应的序号）映射为全0
        :param d_model: Q、K、V矩阵最深一维长度
        :param d_hid: 前馈层中间宽度
        :param n_head: 多头注意力机制头的数量
        :param d_k: K的宽度（Q的和K一样）
        :param d_v: V的宽度
        :param n_position: 构造的位置编码表，需要能使用的最大句长
        :param scale_emb: 是否将embedding得到的矩阵除以sqrt(d_model)
        """
        super().__init__()
        self.d_model = d_model
        self.scale_emb = scale_emb
        # embedding + 位置编码 + Norm
        # padding_idx指定哪个值（即<pad>对应的序号）映射为全0
        self.embedding = nn.Embedding(dict_size, embedding_size, padding_idx=padding_idx)
        self.in_layers = nn.Sequential(
            PositionalEncoding(n_position, embedding_size),  # 加入位置编码
            nn.Dropout(dropout),
            nn.LayerNorm(d_model, eps=1e-6), )
        # n_layers层解码器
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, d_hid, n_head, d_k, d_v, dropout)
                                             for _ in range(n_layer)])

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attention=False):
        dec_output = self.embedding(trg_seq)
        if self.scale_emb: dec_output *= self.d_model ** 0.5
        dec_output = self.in_layers(dec_output)
        # 进入解码器层
        dec_slf_attn_list, dec_enc_attn_list = [], []
        for dec_layer in self.decoder_layers:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            if return_attention:
                dec_slf_attn_list.append(dec_slf_attn)
                dec_enc_attn_list.append(dec_enc_attn)
        if return_attention: return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_hid, n_head, d_k, d_v, dropout=0):
        """
        编码器层 = 多头注意力 + FFN\n
        :param d_hid: 前馈层中间宽度
        :param n_head: 多头自注意力的头的数量
        :param d_k: K的宽度（Q的和K一样）
        :param d_v: V的宽度
        :returns: enc_output, enc_slf_attn：计算结果、注意力矩阵
        """
        super().__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_hid, dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.ffn(enc_output)
        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_hid, n_head, d_k, d_v, dropout=0):
        """
        解码器层 = 多头注意力（自） + 多头注意力（与编码器） + FFN\n
        :param d_hid: 前馈层中间宽度
        :param n_head: 多头自注意力的头的数量
        :param d_k: K的宽度（Q的和K一样）
        :param d_v: V的宽度
        :returns: dec_output, dec_slf_attn, dec_enc_attn：计算结果、第1层的注意力矩阵、第2层的注意力矩阵
        """
        super().__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_hid, dropout)

    def forward(self, dec_input, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output, dec_enc_attn = self.enc_attn(dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output = self.ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn
