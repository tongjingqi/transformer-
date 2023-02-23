"""Transformer模型、构建pad_mask、构建下三角mask"""
import torch
import torch.nn as nn

from Transformer.Layers import Encoder, Decoder


class Transformer(nn.Module):
    def __init__(self, src_dict_size, trg_dict_size, src_padding_idx, trg_padding_idx,
                 embedding_size=512, d_model=512, d_hid=2048,
                 n_layer=6, n_head=8, d_k=64, d_v=64, n_position=200, dropout=0,
                 trg_prj_weight_share=True, src_trg_weight_share=False,
                 scale_emb_or_prj='prj', return_attention=False):
        """
        :param src_dict_size: 字典长度（word2index字典长度）
        :param src_padding_idx: 指定哪个值（即<pad>对应的序号）映射为全0
        :param embedding_size: embedding size
        :param d_model: Q、K、V矩阵最深一维长度（一般等于embedding_size）
        :param d_hid: 前馈层中间宽度
        :param n_layer: 编码器层数
        :param n_head: 多头注意力机制头的数量
        :param d_k: K的宽度（Q的和K一样）
        :param d_v: V的宽度
        :param n_position: 构造的位置编码表，需要能使用的最大句长
        :param trg_prj_weight_share: 分类器、解码器的embedding共享权重
        :param src_trg_weight_share: 编码器的embedding、解码器的embedding共享权重
        :param scale_emb_or_prj: 对embedding、还是分类器输出进行缩放
        :param return_attention: 是否返回attention矩阵
        """
        super().__init__()
        self.d_model = d_model
        self.src_padding_idx, self.trg_padding_idx = src_padding_idx, trg_padding_idx
        self.return_attention = return_attention
        # 'emb': multiply \sqrt{d_model} to embedding output
        # 'prj': multiply (\sqrt{d_model} ^ -1) to linear projection output
        # 'none': no multiplication
        assert scale_emb_or_prj in ['emb', 'prj', 'none']
        scale_emb = (scale_emb_or_prj == 'emb') if trg_prj_weight_share else False
        self.scale_prj = (scale_emb_or_prj == 'prj') if trg_prj_weight_share else False

        # 编码器、解码器、分类器（全连接映射到目标语字典）
        self.encoder = Encoder(n_layer, src_dict_size, embedding_size, src_padding_idx,
                               d_model, d_hid, n_head, d_k, d_v, n_position, dropout, scale_emb)
        self.decoder = Decoder(n_layer, trg_dict_size, embedding_size, trg_padding_idx,
                               d_model, d_hid, n_head, d_k, d_v, n_position, dropout, scale_emb)
        self.projection = nn.Linear(d_model, trg_dict_size, bias=False)

        # 参数初始化
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)

        assert d_model == embedding_size, \
            'To facilitate the residual connections, \
             the dimensions of all module outputs shall be the same.'

        # 分类器、解码器的embedding共享权重
        if trg_prj_weight_share: self.projection.weight = self.decoder.embedding.weight
        # 编码器的embedding、解码器的embedding共享权重
        if src_trg_weight_share: self.encoder.embedding.weight = self.decoder.embedding.weight

    def forward(self, src_input, trg_input):
        # mask：<pab>处=0、上三角处=0
        src_mask = get_pad_mask(src_input, self.src_padding_idx)
        trg_mask = get_pad_mask(trg_input, self.trg_padding_idx) & get_subsequent_mask(trg_input)

        enc_output = self.encoder(src_input, src_mask, False)
        dec_output = self.decoder(trg_input, trg_mask, enc_output, src_mask, False)  # [b, length_sentence, d_model]
        logit = self.projection(dec_output)  # [b, length_sentence, trg_dict_size]
        if self.scale_prj: logit *= self.d_model ** -0.5
        return logit.view(-1, logit.size(2))  # [b*length_sentence, trg_dict_size]


def get_pad_mask(seq, pad_idx):
    """<pad>处=0"""
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    """构建下三角矩阵，对角线上移为1"""
    len_sentence = seq.size()[1]
    # tril下三角矩阵，triu上三角矩阵。diagonal：+对角线上移个数，-对角线下移个数
    subsequent_mask = torch.tril(torch.ones((1, len_sentence, len_sentence), device=seq.device),
                                 diagonal=1).bool()
    return subsequent_mask
