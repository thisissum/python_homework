import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset


def load_json(path, encoding='utf-8'):
    with open(path,'r',encoding=encoding) as f:
        data = json.load(f)
    return data


def save_json(path, data, encoding='utf-8'):
    with open(path,'w', encoding=encoding) as f:
        json.dump(data, f)


def load_npy(path):
    output = np.load(path)
    return output


def save_npy(path, data):
    np.save(path, data)


def save_model(path, epoch, models, loss, device):
    torch.save({
            'epoch': epoch + 1,
            'model': [model.state_dict() for model in models],
            'loss': loss,
            'device': device
        }, os.path.join(path, '{}_{}.tar'.format('checkpoint', epoch+1)))


def load_by_step(loader, steps=1000):
    count = 0
    while count < steps:
        for data in loader:
            count += 1
            yield data
            if count >= steps:
                break



class FlexibleDataset(Dataset):
    """A torch Dataset that support multi input
    """

    def __init__(self, *args):
        super(FlexibleDataset, self).__init__()
        self.args_num = len(args)
        if self.args_num == 1:
            self.inputs = args[0]
        else:
            self.inputs = [i for i in args]

    def __len__(self):
        if self.args_num == 1:
            return len(self.inputs)
        else:
            return len(self.inputs[0])

    def __getitem__(self, index):
        if self.args_num == 1:
            output = self.inputs[index]
        else:
            output = [data[index] for data in self.inputs]
        return output


def build_dataloader(*args, batch_size=64):
    """Using FlexibleDataset to build a DataLoader with specific batch_size
    args:
        *args: multi input tensor, data to load
        batch_size: int.
    """
    fdataset = FlexibleDataset(*args)
    dataloader = DataLoader(fdataset, batch_size=batch_size)
    return dataloader


class SentenceReader(object):
    """A memory saving method using for-loop to read corpus
    args:
        path: str, path of corpus
        func: function, function to preprocess one line.
    """

    def __init__(self, path, func=None, encoding='utf-8'):
        self.path = path
        self.func = func
        self.encoding = encoding

    def __iter__(self):
        if self.func is None:
            with open(self.path, 'r', encoding=self.encoding) as f:
                for sentence in f:
                    yield sentence
        else:
            with open(self.path, 'r', encoding=self.encoding) as f:
                for sentence in f:
                    yield self.func(sentence)


class BatchSentenceReader(object):
    """A memory saving method using for-loop to read corpus with specific batch
    args:
        path: str, path of corpus
        batch_size: int
    """
    def __init__(self, path, batch_size, encoding='utf-8'):
        self.path = path
        self.batch_size = batch_size
        self.encoding = encoding

    def __iter__(self):
        with open(self.path, 'r', encoding=self.encoding) as f:
            i = 0
            sentence_list = []
            for sentence in f:
                if i % self.batch_size == 0 and i != 0:
                    yield sentence_list
                    sentence_list = []
                i += 1
                sentence_list.append(sentence)
            yield sentence_list



