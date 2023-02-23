# %%
# code by Tae Hwan Jung(Jeff Jung) @graykode, Derek Miller @dmmiller612
# Reference : https://github.com/jadore801120/attention-is-all-you-need-pytorch
#           https://github.com/JayParks/transformer
import math
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from Transformer.SubLayers import PositionalEncoding
from Transformer.Optim import ScheduledOptim
from Load import Make_Dicts, Train_Dataset, MAX_SENTENCE_LENGTH
from _score import Test


def get_sinusoid_encoding_table(n_position, d_model):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table)


def get_attn_pad_mask(seq_q, seq_k):
    # print(seq_q)
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size, 1, len_k(=len_q), one is masking
    return pad_attn_mask.expand(batch_size, len_q, len_k).to(device)  # batch_size, len_q, len_k


def get_attn_subsequent_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    return subsequent_mask.to(device)


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(
            d_k)  # scores : [batch_size, n_head, len_q(=len_k), len_k(=len_q)]
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        # W矩阵仅在最后一个维度（embedding size）上映射
        self.W_Q = nn.Linear(d_model, d_k * n_head)
        self.W_K = nn.Linear(d_model, d_k * n_head)
        self.W_V = nn.Linear(d_model, d_v * n_head)
        self.linear = nn.Linear(n_head * d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        # Q: [batch_size, len_q, d_model], K: [batch_size, len_k, d_model], V: [batch_size, len_k, d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, n_head, d_k).transpose(1, 2)  # q_s: [batch_size, n_head, len_q, d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_head, d_k).transpose(1, 2)  # k_s: [batch_size, n_head, len_k, d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_head, d_v).transpose(1, 2)  # v_s: [batch_size, n_head, len_k, d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_head, 1, 1)  # attn_mask : [batch_size, n_head, len_q, len_k]

        # context: [batch_size, n_head, len_q, d_v], attn: [batch_size, n_head, len_q(=len_k), len_k(=len_q)]
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        # contiguous().view()与reshape()等价
        context = context.transpose(1, 2).reshape(batch_size, -1, n_head * d_v)  # context: [batch_size, len_q, n_head * d_v]
        output = self.linear(context)
        return self.layer_norm(output + residual), attn  # output: [batch_size, len_q, d_model]


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_hid, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_hid, out_channels=d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs  # inputs : [batch_size, len_q, d_model]
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)  # inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, len_q, d_model]
        return enc_outputs, attn


class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_dict_size, d_model)
        self.pos_emb = PositionalEncoding(MAX_SENTENCE_LENGTH, d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layer)])

    def forward(self, enc_inputs):  # inputs : [batch_size, source_len]
        enc_outputs = self.src_emb(enc_inputs).to(device)
        enc_outputs = self.pos_emb(enc_outputs).to(device)
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(trg_dict_size, d_model)
        self.pos_emb = PositionalEncoding(MAX_SENTENCE_LENGTH, d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layer)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):  # target_inputs : [batch_size, target_len]
        dec_outputs = self.tgt_emb(dec_inputs)
        dec_outputs = self.pos_emb(dec_outputs)
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)

        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.projection = nn.Linear(d_model, trg_dict_size, bias=False)

    def forward(self, enc_inputs, dec_inputs):
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs)  # dec_logits : [batch_size, src_dict_size, trg_dict_size]
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns


def  greedy_decoder(model, enc_input, start_symbol):
    enc_outputs = model.encoder(enc_input)[0].to(device)
    dec_input = torch.zeros(1, 11).type_as(enc_input.data)
    next_symbol = start_symbol
    for i in range(0, 11):
        dec_input[0][i] = next_symbol
        dec_outputs = model.decoder(dec_input[:,:i+1], enc_input, enc_outputs)[0]
        projected = model.projection(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[i]
        next_symbol = next_word.item()
    return dec_input


train_path_src = 'data/train.new.en'
train_path_trg = 'data/train.new.de'
val_path_src = 'data/valid.new.en'
val_path_trg = 'data/valid.new.de'
# 训练相关参数，及默认值
epochs = 40  # 总训练轮数
checkpoint = 5  # 保存点（间隔checkpoirnt保存一次）
batch_words_approximately = 4600  # 一个batch大约包含的单词数量（上限），至少为MAX_SENTENCE_LENGTH+1. 4096
lr_mul = 3.0  # 学习率的缩放倍数？，2.0。Transformer学习率=lr_mul * [d^-0.5*min(step^-0.5, step*warmup_steps^-1.5)]
warmup_steps = 6000  # 预热步数，4000
label_smoothing = 0.1  # 标签平滑，0.1
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

# 模型相关参数，及默认值
d_model = 512  # Embedding Size，512
d_hid = 2048  # FeedForward dimension，2048
d_k = d_v = 64  # dimension of K(=Q), V，64
n_layer = 6  # number of Encoder of Decoder Layer，6
n_head = 8  # number of heads in Multi-Head Attention，8

if __name__ == '__main__':
    # 训练集
    word2index_src, index2word_src, X = Make_Dicts(train_path_src)  # 字典构造
    word2index_trg, index2word_trg, Y = Make_Dicts(train_path_trg)
    src_dict_size = len(word2index_src)  # 字典大小
    trg_dict_size = len(word2index_trg)
    train_dataset = Train_Dataset(X, word2index_src, Y, word2index_trg, batch_words_approximately)
    # 验证集
    val_dataset = Train_Dataset(val_path_src, word2index_src, val_path_trg, word2index_trg, batch_words_approximately)
    print(f'字典大小：原语{src_dict_size}，目标语{trg_dict_size}\n'
          f'训练集长度：{len(train_dataset)}')

    # 网络
    model = Transformer().to(device)
    # 损失函数：ignore_index忽略的标签、label_smoothing标签平滑
    loss_fn = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=label_smoothing)
    # 优化器、调度器
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09)  # Transformer默认参数
    optimizer = ScheduledOptim(optimizer, lr_mul, d_model, warmup_steps)  # 调度器：预热->降低


# 训练
def Train():
    loss_min_train = float('inf')  # 保存train损失最低的模型
    loss_min_val = float('inf')  # 保存val损失最低的模型
    for epoch in tqdm(range(epochs), '训练中', mininterval=5):
        loss_total_train = 0
        batch_count = 0
        for inputs, target_inputs, target_outputs in train_dataset:
            inputs, target_inputs, target_outputs = inputs.to(device), target_inputs.to(device), target_outputs.to(device)
            output = model(inputs, target_inputs)[0]  # [b*length_sentence, trg_dict_size]
            optimizer.zero_grad()
            loss = loss_fn(output, target_outputs.view(-1))
            loss.backward()
            optimizer.step_and_update_lr()

            # 损失输出与存储画图
            loss_total_train += loss.data
            batch_count += 1
            # if batch_count > 1: break
        loss_total_train /= batch_count
        print(f' train：loss={loss_total_train}, ppl={math.exp(loss_total_train)}')
        train_loss_list.append(loss_total_train.to(device))  # 训练集损失(chage cpu to device)
        # loss_total_val = Test(model, val_dataset, device).to('cpu')  # 验证集损失
        # val_loss_list.append(loss_total_val)
        model.train()
        lr_list.append(optimizer.optimizer.state_dict()['param_groups'][0]['lr'])  # 学习率

        # 间隔固定轮数模型保存
        if not (epoch + 1) % checkpoint:
            if not os.path.exists('data/outputs'): os.makedirs('data/outputs')
            torch.save(model, f'data/outputs/transformer_epoch{epoch + 1}.ckpt')
        # 保存train损失最小的网络
        if loss_total_train < loss_min_train:
            if os.path.exists(f'data/outputs/transformer_epoch{epochs}_train_loss{loss_min_train:0.6f}.ckpt'):
                os.remove(f'data/outputs/transformer_epoch{epochs}_train_loss{loss_min_train:0.6f}.ckpt')  # 只保存loss最低的1个
            loss_min_train = loss_total_train
            torch.save(model, f'data/outputs/transformer_epoch{epochs}_train_loss{loss_min_train:0.6f}.ckpt')
        # # 保存val损失最小的网络
        # if loss_total_val < loss_min_val:
        #     if os.path.exists(f'data/outputs/transformer_epoch{epochs}_val_loss{loss_min_val:0.6f}.ckpt'):
        #         os.remove(f'data/outputs/transformer_epoch{epochs}_val_loss{loss_min_val:0.6f}.ckpt')  # 只保存loss最低的1个
        #     loss_min_val = loss_total_val
        #     torch.save(model, f'data/outputs/transformer_epoch{epochs}_val_loss{loss_min_val:0.6f}.ckpt')


if __name__ == '__main__':
    train_loss_list = []  # 在训练集损失
    # val_loss_list = []  # 在验证集损失
    lr_list = []  # 学习率
    Train()

    # 画图
    import matplotlib.pyplot as plt

    pickle.dump(train_loss_list, open(f'data/outputs/train_loss_list_epoch{epochs}.pickle', 'wb'))
    # pickle.dump(val_loss_list, open(f'data/outputs/val_loss_list_epoch{epochs}.pickle', 'wb'))
    pickle.dump(lr_list, open(f'data/outputs/lr_list_epoch{epochs}.pickle', 'wb'))

    plt.plot(train_loss_list, label='train loss')
    # plt.plot(val_loss_list, label='val loss')
    # plt.plot(lr_list, label='learning rate')
    plt.title('Change of loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.tight_layout()
    plt.legend()
    plt.savefig(fname=f'data/outputs/损失变化_训练集min{min(train_loss_list):.6f}.png')
    plt.show()
    # traindict_src = open('data/traindict_src.pkl', 'wb')
    # pickle.dump((word2index_src, index2word_src), traindict_src, True);
    # traindict_src.close();
    # traindict_trg = open('data/traindict_trg.pkl', 'wb')
    # pickle.dump((word2index_trg, index2word_trg), traindict_trg, True)
    # traindict_trg.close()
    #
    # model=torch.load('./transformer_epoch10.ckpt',map_location=torch.device('cpu'))
    #
    # model.eval()
    # # 数据集
    # src_input = ['this is a shot of can@@ nery row in 1932 .']  # 11,11， das ist ein bild der can@@ nery row von 1932 .
    # src_input = [[word2index_src.get(i, 1) for i in sentence.split(' ')] for sentence in src_input]  # 原语
    # src_input = torch.tensor(src_input, dtype=torch.long, device=device)  # [batch size, ls]
    # greedy_dec_input = greedy_decoder(model, src_input, start_symbol=word2index_trg.get('<bos>', 1))
    # predict = model(src_input, greedy_dec_input)[0]
    # predict = predict.data.max(1, keepdim=True)[1]
    # print('das ist ein bild der can@@ nery row von 1932 .', '->', [index2word_trg.get(n.item()) for n in predict.squeeze()])
    #
    # for batch in val_dataset:
    #     src_input = batch[0]
    #     print(src_input.shape)
    #     # greedy_dec_input = greedy_decoder(model, src_input, start_symbol=word2index_trg.get('<bos>', 1))
    #     break