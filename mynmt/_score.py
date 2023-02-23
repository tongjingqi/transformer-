import torch
from torch import nn
from torch.nn import functional as F

import math


# 测试
def Test(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        loss_total = 0
        batch_count = 0
        for inputs, target_inputs, target_outputs in dataloader:
            inputs, target_inputs, target_outputs = inputs.to(device), target_inputs.to(device), target_outputs.to(device)
            output = model(inputs, target_inputs)  # [b*length_sentence, trg_dict_size]
            loss = F.cross_entropy(output, target_outputs.view(-1), ignore_index=0, label_smoothing=0.1)
            # 损失输出与存储画图
            loss_total += loss.data
            batch_count += 1
        loss_total /= batch_count
        print(f'val/test：loss={loss_total}, ppl={math.exp(loss_total)}')
    return loss_total

#
# def Score(pred, gold, trg_pad_idx, smoothing=False):
#     loss = cal_loss(pred, gold, trg_pad_idx, smoothing=smoothing)
#
#     pred = pred.max(1)[1]
#     gold = gold.contiguous().view(-1)
#     non_pad_mask = gold.ne(trg_pad_idx)
#     n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
#     n_word = non_pad_mask.sum().item()
#
#     return loss, n_correct, n_word
#
#
# def cal_loss(pred, gold, trg_pad_idx, smoothing=False):
#     """ Calculate cross entropy loss, apply label smoothing if needed. """
#     gold = gold.contiguous().view(-1)
#
#     if smoothing:
#         eps = 0.1
#         n_class = pred.size(1)
#
#         one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
#         one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
#         log_prb = F.log_softmax(pred, dim=1)
#
#         non_pad_mask = gold.ne(trg_pad_idx)
#         loss = -(one_hot * log_prb).sum(dim=1)
#         loss = loss.masked_select(non_pad_mask).sum()  # average later
#     else:
#         loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='sum')
#     return loss
