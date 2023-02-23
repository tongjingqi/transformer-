import torch
from torch.utils.data import Dataset

MAX_SENTENCE_LENGTH = 500  # 句子最大长度
extra_character_list = ['<pad>', '<unk_word>', '<bos>', '<eos>']  # 填充、不在字典、开头、结尾


class Train_Dataset(Dataset):
    def __init__(self, source_X: dir or list, word2index_X: dict,
                 source_Y: dir or list, word2index_Y: dict, batch_words_approximately=10000):
        """
        加载数据\n
        :param source_X: dir时：读取该文件；list时：sentences
        :param word2index_X: word2index_src：{单词->序号} dict
        :param batch_words_approximately: 一个batch大约包含的单词数量（上限）
        :returns: inputs, target_inputs, target_outputs：Decoder原语输入，Encoder目标语输入，Encoder目标语输出
        """
        X = list(map(str.strip, open(source_X, 'r', encoding='utf-8').readlines())) \
            if isinstance(source_X, str) else source_X
        Y = list(map(str.strip, open(source_Y, 'r', encoding='utf-8').readlines())) \
            if isinstance(source_Y, str) else source_Y
        # 将数据集用词典映射，并按照原语的长度，从小到大排序
        self.inputs, self.target_inputs, self.target_outputs = _Make_Train_Dataset(X, word2index_X, Y, word2index_Y,
                                                                                   batch_words_approximately)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx], dtype=torch.long), \
               torch.tensor(self.target_inputs[idx], dtype=torch.long), \
               torch.tensor(self.target_outputs[idx], dtype=torch.long)


def _Make_Train_Dataset(sentences_X: list, word2index_X: dict, sentences_Y: list, word2index_Y: dict, batch_words_approximately):
    """
    将数据集用词典映射，并按照原语的长度，从小到大排序\n
    :param sentences_X: 原语所有句子
    :param word2index_X: 原语词典
    :param sentences_Y: 目标语所有句子
    :param word2index_Y: 目标与词典
    :return: inputs, target_inputs, target_outputs
    """
    # 对每个句子，映射为序列
    inputs = [[word2index_X.get(i, 1) for i in sentence.split(' ')] for sentence in sentences_X]  # 原语
    target_inputs = [[2] + [word2index_Y.get(i, 1) for i in sentence.split(' ')] for sentence in sentences_Y]  # 目标语正确答案：前加<bos>
    target_outputs = [[word2index_Y.get(i, 1) for i in sentence.split(' ')] + [3] for sentence in sentences_Y]  # 目标语正确答案：后加<eos>
    # 按照原语句长，由小到大排序
    temp = sorted(zip(inputs, target_inputs, target_outputs), key=lambda i: len(i[0]))
    # 将排好序的数据解包
    inputs, target_inputs, target_outputs = [], [], []
    _input, _target_input, _target_output = [], [], []
    for x, y, z in temp:
        if len(x) > MAX_SENTENCE_LENGTH or len(y) > MAX_SENTENCE_LENGTH: continue  # 过滤调长度超出范围的句子
        # 如果一个batch所需单词数量未满
        if (len(_input) + 1) * len(x) < batch_words_approximately:  # 预估单词总数量 = 句子数 * 最长句子长度
            _input.append(x)
            _target_input.append(y)
            _target_output.append(z)
        else:
            # 确定该batch最长句子
            _max_length_input = len(_input[-1])  # 输入：该batch中最长的句子就是最后一个
            _max_length_target = max(map(len, _target_input))  # 输出：找到最长句子的长度
            # 在末尾补0（<pad>）至该batch最大长度
            inputs.append([sentence + [0] * (_max_length_input - len(sentence)) for sentence in _input])
            target_inputs.append([sentence + [0] * (_max_length_target - len(sentence)) for sentence in _target_input])
            target_outputs.append([sentence + [0] * (_max_length_target - len(sentence)) for sentence in _target_output])
            _input, _target_input, _target_output = [], [], []
    return inputs, target_inputs, target_outputs


def Make_Dicts(source: dir):
    """
    根据source构造索引字典（按字母顺序排列）\n
    :param source: 读取文件路径
    :returns: word2index, index2word, sentences：字典（单词->序号）、字典（序号->单词）、所有句子
    """
    # 获取所有词
    sentences = open(source, 'r', encoding='utf-8').readlines()
    sentences = [i.strip() for i in sentences]  # 去除开头结尾所有空字符
    word_list = set(' '.join(sentences).split(' '))  # 将所有句子合并、然后分割，就得到了所有单词
    word_list = sorted(list(word_list))  # 将word_list转为有序的列表
    # 构造字典
    word_list = extra_character_list + word_list  # 添加<pad>填充、<unk_word>不在字典、<bos>开头、<eos>结尾
    word2index = dict((word, index) for index, word in enumerate(word_list))  # 词->序列
    index2word = dict((index, word) for index, word in enumerate(word_list))  # 序列->词
    return word2index, index2word, sentences

# if __name__ == '__main__':
#     word2index_X, index2word_X, X = Make_Dicts('data/trainres.de')
#     word2index_Y, index2word_Y, Y = Make_Dicts('data/trainres.en')
#     # print(word2index_Y)
#     # exit()
#     train_dataset = Train_Dataset(X, word2index_X, Y, word2index_Y, 1000)
#     for inputs, target_inputs, target_outputs in train_dataset:
#         print(inputs.shape, target_inputs.shape, target_outputs.shape, sep='，')
