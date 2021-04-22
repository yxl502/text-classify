import torch
import torch.nn as nn
from torch import optim
from models import Model
from datasets import data_loader, text_CLS
from configs import Config

cfg = Config()

data_path = 'sources/weibo_senti_100k.csv'
data_stop_path = 'sources/hit_stopword'
dict_path = 'sources/dict'
dataset = text_CLS(dict_path, data_path, data_stop_path)
train_dataloader = data_loader(dataset, cfg)
cfg.pad_size = dataset.max_len_seq

model_text_cls = Model(config=cfg)
model_text_cls.to(cfg.device)
model_text_cls.load_state_dict(torch.load('models/10.pth'))

#########################################################################
# print(train_dataloader)
# for i, batch in enumerate(train_dataloader):
#     label, data = batch
#     data = torch.tensor(data, dtype=torch.int64).to(cfg.device)
#     # print('data:', data)
#     label = torch.tensor(label, dtype=torch.int64).to(cfg.device)
#     # print('label:', label)
#
#     pred_softmax = model_text_cls.forward(data)
#     # print(pred_softmax)
#     # print(label)
#
#     pred = torch.argmax(pred_softmax, dim=1)
#     # print('pred', pred)
#
#     out = torch.eq(pred, label)
#     # print(out)
#     print(out.sum() * 1.0 / pred.size()[0])

#########################################################################

import numpy as np
from datasets import load_data, read_dict

dict_path = 'sources/dict'
data_path = 'sources/weibo_senti_100k.csv'
data_stop_path = 'sources/hit_stopword'

_, max_len_seq = load_data(data_path, data_stop_path)

# word_list = '早上好，周末还要工作的你[晕][晕][晕][晕][晕]'
word_list = '好成绩连连到啊！'
input_idx = []

voc_dict = read_dict(dict_path)

for word in word_list:
    if word in voc_dict.keys():
        input_idx.append(voc_dict[word])
    else:
        input_idx.append(voc_dict['<UNK>'])

if len(input_idx) < max_len_seq:
    input_idx += [voc_dict['<PAD>']
                  for _ in range(max_len_seq - len(input_idx))]

data = np.array(input_idx)
data = np.expand_dims(data, 0)
data = torch.tensor(data, dtype=torch.int64).to(cfg.device)
# print(data)
pred_softmax = model_text_cls.forward(data)
#
#
pred = torch.argmax(pred_softmax, dim=1)
print('pred', pred)
