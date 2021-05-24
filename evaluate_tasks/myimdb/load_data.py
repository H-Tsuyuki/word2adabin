import os
import csv
import io
import glob
import numpy as np
from tqdm import tqdm
import random

import torch
from nltk.tokenize import word_tokenize

def read_trec(args, data_type):
    tgt2idx = {'ABBR': 0, 'DESC': 1, 'ENTY': 2,
               'HUM': 3, 'LOC': 4, 'NUM': 5}
 #   tgt2idx = {'0': 0, '1': 1, '2': 2,
  #             '3': 3, '4': 4, '5': 5}
    dataset = []
    if data_type=="train":
        path = args.trec_path+"train.bpe"
        path = args.trec_path+"train_5500.label"
    else:
        path = args.trec_path+"test.bpe"
        path = args.trec_path+"TREC_10.label"
    with io.open(path, encoding='utf-8', errors='ignore') as f:
        text = f.readlines()
        for line in text:
            label, tokens = line.strip().split(':', 1)
            tokens = tokens.lower().split(' ', 1)[1].split()
            dataset.append([tokens, tgt2idx[label]])

    return dataset

def read_sst2(args, data_type):
    dataset = []
    path = args.sst2_path+"sentiment-"+data_type
    with io.open(path, encoding='utf-8', errors='ignore') as f:
        text = f.readlines()
        for line in text:
            sample = line.strip().split('\t')
            label = int(sample[1])
            tokens = sample[0].lower().split()[:400]
            dataset.append([tokens, label])
    return dataset


def get_nli(data_path):
    s1 = {}
    s2 = {}
    target = {}

    dico_label = {'entailment': 0,  'neutral': 1, 'contradiction': 2}

    for data_type in ['train', 'dev', 'test']:
        s1[data_type], s2[data_type], target[data_type] = {}, {}, {}
        s1[data_type]['path'] = os.path.join(data_path, 's1.' + data_type)
        s2[data_type]['path'] = os.path.join(data_path, 's2.' + data_type)
        target[data_type]['path'] = os.path.join(data_path,
                                                 'labels.' + data_type)

        s1[data_type]['sent'] = [line.rstrip().split() for line in
                                 open(s1[data_type]['path'], 'r')]
        s2[data_type]['sent'] = [line.rstrip().split() for line in
                                 open(s2[data_type]['path'], 'r')]
        target[data_type]['data'] = np.array([dico_label[line.rstrip('\n')]
                for line in open(target[data_type]['path'], 'r')])

        assert len(s1[data_type]['sent']) == len(s2[data_type]['sent']) == \
            len(target[data_type]['data'])

        print('** {0} DATA : Found {1} pairs of {2} sentences.'.format(
                data_type.upper(), len(s1[data_type]['sent']), data_type))

    train = [s1['train']['sent'], s2['train']['sent'], target['train']['data']]
    dev = [s1['dev']['sent'], s2['dev']['sent'], target['dev']['data']]
    test = [s1['test']['sent'], s2['test']['sent'], target['test']['data']]
    
    train = [list(x) for x in zip(*train)]
    dev = [list(x) for x in zip(*dev)]
    test = [list(x) for x in zip(*test)]
    return train, dev, test

def read_csv(args, data_type):
    if args.task == "dbpedia":
        path = args.dbpedia_path+data_type+".csv"
    elif args.task=="ag":
        path = args.ag_path+data_type+".csv"
    elif args.task=="yelp":
        path = args.yelp_path+data_type+".csv"

    dataset = []
    with io.open(path, encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)            
        for line in tqdm(reader):
            label = str(int(line[0])-1)
            if args.task=="yelp":            
                tokens = [i.lower() for i in word_tokenize(line[1].strip())][:400]
            else:            
                tokens = [i.lower() for i in word_tokenize(line[2].strip())][:400]
            dataset.append([tokens, label])
            #if len(dataset)==100000:
            #    break
    return dataset 

#def read_imdb(args, data_type):
#    path = args.imdb_path+data_type
#    def read_and_label(posneg, label):
#        dataset = []
#        target = os.path.join(path, posneg, '*')
#        for i, f_path in enumerate(tqdm(glob.glob(target))):
#            with io.open(f_path, encoding='utf-8', errors='ignore') as f:
#                text = f.read().strip()
#            tokens = [i.lower() for i in word_tokenize(text)][:400]
#            dataset.append([tokens, label])
##            if len(dataset)>100: break
#        return dataset
#
#    pos_dataset = read_and_label('pos', 0)
#    neg_dataset = read_and_label('neg', 1)
#    dataset = pos_dataset + neg_dataset
#    return dataset 


def read_imdb(args, data_type):
    if args.task == "dbpedia":
        f_path = args.dbpedia_path+data_type+".bpe"
        f_path = args.dbpedia_path+data_type+".txt"
    elif args.task=="ag":
    #    f_path = args.ag_path+data_type+".bpe"
        f_path = args.ag_path+data_type+".txt"
    elif args.task=="yelp":
        f_path = args.yelp_path+data_type+".txt"
    elif args.task=="imdb":
    #    f_path = args.imdb_path+data_type+".bpe"
        f_path = args.imdb_path+data_type+".txt"
    dataset = []
    with io.open(f_path, encoding='utf-8', errors='ignore') as f:
        text = f.readlines()
    for line in tqdm(text):
        label = line.strip().split()[0]
        tokens = line.strip().lower().split()[1:]
        tokens = tokens[:400]
        #from IPython.core.debugger import Pdb; Pdb().set_trace()
        dataset.append([tokens, label])
    return dataset


def read_sst5(args, data_type):
    dataset = []
    path = args.sst5_path+data_type+".bpe"
    path = args.sst5_path+"sentiment-"+data_type
    with io.open(path, encoding='utf-8', errors='ignore') as f:
        text = f.readlines()
    for line in text:
        label = line.strip().split()[0]
        tokens = line.strip().lower().split()[1:]
        dataset.append([tokens, label])
    return dataset 
