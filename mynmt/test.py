from temp import Make_Dicts,greedy_decoder,Transformer,Encoder,Decoder,EncoderLayer,DecoderLayer,MultiHeadAttention,PoswiseFeedForwardNet
import torch
print(torch.cuda.is_available())
import pickle
from random import choice

def read_dict(src_path,trg_path):
    src_file=open(src_path,'rb')
    trg_file=open(trg_path,'rb')
    word2index_src, index2word_src=pickle.load(src_file)
    word2index_trg, index2word_trg=pickle.load(trg_file)
    return word2index_src,index2word_src,word2index_trg,index2word_trg

# test_path_src='./data/test.en'
# test_path_trg='./data/test.de'
test_path_src='./data/valid.new.en'
test_path_trg='./data/valid.new.de'
model_path='./transformer_epoch10.ckpt'
tests=10

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
from Transformer.SubLayers import _get_sinusoid_encoding_table

slen = 200
d_model = 512
if __name__=='__main__':
    _word2index_src,_index2word_src,X = Make_Dicts(test_path_src)
    _word2index_trg,_index2word_trg,Y = Make_Dicts(test_path_trg)
    word2index_src,index2word_src,word2index_trg,index2word_trg=read_dict('data/traindict_src.pkl','data/traindict_trg.pkl')
    sourcelen=len(X)
    model=torch.load(model_path,map_location=device)
    model.encoder.pos_emb.pos_table = _get_sinusoid_encoding_table(slen, d_model)
    model.decoder.pos_emb.pos_table = _get_sinusoid_encoding_table(slen, d_model)
    res_file = open("./data/result_valid.txt", "w+", encoding="utf-8")
    for i in range(sourcelen):
        src_input=X[i]
        src_in=src_input
        src_input=[[word2index_src.get(i, 1) for i in sentence.split(' ')] for sentence in [src_input]]  # 原语
        # print(src_input)
        src_input = torch.tensor(src_input, dtype=torch.long, device=device)  # [batch size, ls]
        # print(src_input)
        greedy_dec_input = greedy_decoder(model, src_input, start_symbol=word2index_trg.get('<bos>', 1)).to(device)
        predict = model(src_input, greedy_dec_input)[0]
        predict = predict.data.max(1, keepdim=True)[1]
        print(src_in, '->', [index2word_trg.get(n.item()) for n in predict.squeeze()])
        res_file.write(" ".join([index2word_trg.get(n.item()) for n in predict.squeeze()])+'\n')

