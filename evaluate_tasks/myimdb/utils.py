import os
import csv
import io
import glob
import numpy as np
from tqdm import tqdm
import random

import torch
from torch.utils.data import Dataset
from collections import defaultdict
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sent, label = self.data[idx]
#        sent = torch.cat(sent).view(len(sent), len(sent[0]))
        return (sent, label)

class MyIterator(object):
    def __init__(self, dataset, vocab, batch_size, shuffle):
        self.dataset = dataset
        self.max_length = max([len(line[0]) for line in dataset])
        if len(dataset[0])==3:
            self.max_length1 = max([len(line[1]) for line in dataset])
        if shuffle:
            random.shuffle(self.dataset)
        self.dataset = [self.dataset[i:i+batch_size] for i in range(0,len(self.dataset),batch_size)] 
        #self.dataset = cycle(self.dataset)
        self._i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._i == len(self.dataset):
            raise StopIteration()
# 最大長までpadding
        batch = self.dataset[self._i]
        max_length = max([len(line[0]) for line in batch])
        if len(batch[0])==3:
            max_length1 = max([len(line[1]) for line in batch])
            max_length = max([max_length1,max_length])
        lengths = [len(line[0]) for line in batch]
        pad_sent =  [line[0]+[1]*(max_length-len(line[0])) for line in batch] 
        if len(batch[0])==3:
            lengths1 = [len(line[1]) for line in batch]
            pad_sent1 =  [line[1]+[1]*(max_length-len(line[1])) for line in batch] 
            label =  [ int(line[2]) for line in batch]
            pad_sent = [ [pad_sent[i], pad_sent1[i] ]for i in range(len(batch))]
            lengths = [ [lengths[i], lengths1[i]]for i in range(len(batch))]
        else:
            label =  [ int(line[1]) for line in batch]

        self._i +=1
        return pad_sent, label, lengths

def make_vocab(dataset, max_vocab_size=30000, min_freq=None):
    counts = defaultdict(int)
    if len(dataset[0])==2:
        for tokens, _ in dataset:
            for token in tokens:
                counts[token] += 1
    else:
        for tokens1,tokens2, _ in dataset:
            tokens = tokens1 + tokens2
            for token in tokens:
                counts[token] += 1

    vocab = {'<unk>': 0, '<pad>': 1}
    for w, c in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        #if len(vocab) >= max_vocab_size or c < min_freq:
        if c<=0:
            del counts[w]
        else:
            vocab[w] = len(vocab)
        
    return vocab, counts

def get_glove(vocab, glove_path):
    # create word_vec with glove vectors
    glove_w2vec = {'<unk>': list(np.random.rand(300)), '<pad>': list(np.zeros(300))}
    new_vocab = {}
    word_vec = []
    i=0
    # vocabの中にある単語のglove vectorを抽出 
    with open(glove_path) as f:
        for line in f:
            i+=1
            word, vec = line.split(' ', 1)
            if word in vocab:
                vec = list(map(float, vec.split()))
                glove_w2vec[word] = vec
    
    for w in vocab:
        if not w in glove_w2vec:
            vocab[w] = -1
    
    for w, i in sorted(vocab.items(), key=lambda x: (-x[1], x[0]), reverse=True):
        if i==-1 : continue 
        new_vocab[w] = len(new_vocab)
        word_vec.append(glove_w2vec[w])
    

    print('Found {0}(/{1}) words with glove vectors'.format(
                len(word_vec), len(vocab)))
    return new_vocab, word_vec


def build_vocab(data, glove_path, min_freq):
    vocab, counts = make_vocab(data, min_freq=min_freq)
 #   word_vec = None
    new_vocab = vocab
    new_vocab, word_vec = get_glove(vocab, glove_path)
    print('Vocab size : {0}'.format(len(new_vocab)))
    return new_vocab, word_vec, counts

def get_glove_all(glove_path):
    # create word_vec with glove vectors
    vocab = {'<unk>': 0, '<pad>': 1}
#    word_vec = [list(np.zeros(300)), list(np.zeros(300))]
    word_vec = [list(np.random.rand(300)), list(np.zeros(300))]
    i=1
    with open(glove_path) as f:
        for line in f:
            i+=1
            word, vec = line.split(' ', 1)
            vec = list(map(float, vec.split()))
            word_vec.append(vec)
            vocab[word]=i
    
    return vocab, word_vec


def to_id(data, vocab, max_length=400):
    for i in range(len(data)):
        if len(data[0])==2:
            sent, label = data[i]
            sent = [vocab.get(w,0) for w in sent]
#           sent = sent+[1]*(max_length-len(sent))
            data[i][0] = sent
        else:
            sent1, sent2, label = data[i]
            sent1 = [vocab.get(w,0) for w in sent1]
            sent2 = [vocab.get(w,0) for w in sent2]
            data[i][0] = sent1
            data[i][1] = sent2
    #data = list(zip(*data))
    return data
