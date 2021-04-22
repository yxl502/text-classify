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

loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_text_cls.parameters(), lr=cfg.learn_rate)

for epoch in range(cfg.num_epochs):
    for i, batch in enumerate(train_dataloader):
        label, data = batch
        data = torch.tensor(data, dtype=torch.int64).to(cfg.device)
        label = torch.tensor(label, dtype=torch.int64).to(cfg.device)

        optimizer.zero_grad()
        pred = model_text_cls.forward(data)
        loss_val = loss_func(pred, label)

        print('epoch is {}, ite is {}, loss_val is {}'.format(epoch, i, loss_val))
        loss_val.backward()

        optimizer.step()

    if epoch % 10 == 0:
        torch.save(model_text_cls.state_dict(), 'models/{}.pth'.format(epoch))
