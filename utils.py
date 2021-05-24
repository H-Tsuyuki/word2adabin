# -*- coding: utf-8 -*-

import os
import io
import pickle
import random
import numpy as np
from tqdm import tqdm

from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        wv, loss, depth, wid, loss_new = self.data[idx]
        return (wv, loss, depth, wid, loss_new)


def tally_param(model):
    npa=0
    for name, param in model.named_parameters():
        print(name, param.nelement())
        npa += param.nelement()
    return npa

# Get word vectors from vocabulary (glove, word2vec, fasttext ..)
def get_glove_vec(path_to_vec, vocab_size):
    word_dict = {}
    word_vec = []

    with io.open(path_to_vec, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    pbar = tqdm(lines, ncols=10)
    pbar.set_description("Load GloVe")
    for i, line in enumerate(pbar):
        if i<2: continue
        if i==vocab_size: break
        word, vec = line.split(' ', 1)
        vec = list(map(float,vec.split()))
        word_dict[word] = i 
        word_vec.append(vec)

    #ramdom.shuffle(word_vec)
    return word_dict, word_vec

def get_word_vec(wv_path, vocab_path):
    with io.open(wv_path, 'rb') as f:
        word_vec = pickle.load(f)
    
   # from IPython.core.debugger import Pdb; Pdb().set_trace() 
    word_vec = word_vec[2:]
    with io.open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    del vocab["<pad>"], vocab["<unk>"]
    return vocab, word_vec


def accum_batches(iterator, accum_count):
    batches = []
    for batch in iterator:
        batches.append(batch)
        if len(batches) == accum_count:
            yield batches
            batches = []
