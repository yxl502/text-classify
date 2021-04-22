from torch.utils.data import Dataset, DataLoader
import jieba
import numpy as np


def read_dict(voc_dict_path):
    voc_dict = {}
    dict_list = open(voc_dict_path, encoding='utf8').readlines()
    for item in dict_list:
        item = item.split(',')
        voc_dict[item[0]] = int(item[1].strip())
    return voc_dict


def load_data(data_path, data_stop_path):
    data_list = open(data_path, encoding='utf8').readlines()[1:]

    stop_word = open(data_stop_path, encoding='utf8').readlines()
    stop_word = [line.strip() for line in stop_word]
    stop_word.append(' ')
    stop_word.append('\n')

    voc_item = {}

    data = []
    max_len_seq = 0

    for item in data_list[:]:
        label = item[0]
        content = item[2:].strip()
        seg_list = jieba.cut(content)

        seg_res = []

        for seg_item in seg_list:
            # print(seg_item)

            if seg_item in stop_word:
                continue

            seg_res.append(seg_item)

            if seg_item in voc_item.keys():
                voc_item[seg_item] = voc_item[seg_item] + 1
            else:
                voc_item[seg_item] = 1

        # print(content)
        # print(seg_res)
        if len(seg_res) > max_len_seq:
            max_len_seq = len(seg_res)

        data.append([label, seg_res])

    return data, max_len_seq


class text_CLS(Dataset):
    def __init__(self, voc_dict_path, data_path, data_stop_path):
        self.data_path = data_path
        self.data_stop_path = data_stop_path

        self.voc_dict = read_dict(voc_dict_path)
        self.data, self.max_len_seq = load_data(self.data_path, self.data_stop_path)

        np.random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data = self.data[item]
        label = int(data[0])
        word_list = data[1]
        input_idx = []

        for word in word_list:
            if word in self.voc_dict.keys():
                input_idx.append(self.voc_dict[word])
            else:
                input_idx.append(self.voc_dict['<UNK>'])

        if len(input_idx) < self.max_len_seq:
            input_idx += [self.voc_dict['<PAD>']
                          for _ in range(self.max_len_seq - len(input_idx))]

        data = np.array(input_idx)
        return label, data


def data_loader(dataset, config):
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=config.is_shuffle)


if __name__ == '__main__':
    data_path = 'sources/weibo_senti_100k.csv'
    data_stop_path = 'sources/hit_stopword'
    dict_path = 'sources/dict'
    dataset = text_CLS(dict_path, data_path, data_stop_path)
    from configs import Config
    cfg = Config()

    train_dataloader = data_loader(dataset, config=cfg)
    for i, batch in enumerate(train_dataloader):
        print(batch)
