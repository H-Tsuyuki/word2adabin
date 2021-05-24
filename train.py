# -*- coding: utf-8 -*-

import os
import pickle
import random
import argparse
import statistics as st
import itertools
import numpy as np
import copy

import torch as t
import torch.nn.functional as F
from bitarray import bitarray
from tqdm import tqdm
from torch.optim import Adam, SGD
from torch.utils.data import Dataset, DataLoader

from adacomp import AdaComp
from emb_ours import CodeAutoEncoder
from utils import tally_param, get_word_vec, get_glove_vec, accum_batches, MyDataset
from config import parse_args


def train(args):
    ## Define Model
    if args.name!="adacomp":
        model = CodeAutoEncoder(args.height, args.n_attr, args.gpu, args.active_depth, args.height, args)
#        model = CodeAutoEncoder(args.height, args.n_attr, args.gpu, args.active_depth, 4, args)
    else:
        model = AdaComp(args.height, args.n_attr, args.gpu, args.active_depth, 4)
    #model.f1.load_state_dict(t.load("./evaluate_tasks/myimdb/f1"))
    #model.f2.load_state_dict(t.load("./evaluate_tasks/myimdb/f2"))
    loss_fn = t.nn.MSELoss()
#    print("Total Params: ",tally_param(model.g))
    model = model.cuda(args.gpu)
    loss_fn = loss_fn.cuda(args.gpu)
    optim = Adam(model.parameters(), lr=args.lr)

   ## Load Data 
    #word_dict, word_vec = get_glove_vec(args.glove_path, args.vocab_size) #GloVeふぁいるのarray
    word_dict, word_vec = get_word_vec(args.word_vec_path, args.vocab_path) #imdbに登場する単語のglove vector
    print(len(word_dict))
    count=0 
    valid_disloss = 0
    valid_conloss = 0
    min_loss = 10**10
    n_iter = 0
    mean_depth, max_depth, min_depth, var_depth = 0, 0, 0, 0
    ## Training 
    for epoch in range(1, args.epoch + 1):
        trainloader = DataLoader(word_vec, batch_size=args.mb, shuffle=True)
     
        pbar = tqdm(trainloader)
        pbar.set_description("[Epoch {}]".format(epoch))
        for i,wv in enumerate(pbar):
            x = model.to_tensor(wv)
            _, con, _ = model(x)
            y, _, ctrl = con
            mseloss = loss_fn(x, y) 
            loss = mseloss 

            if True or args.active_depth:
                #lengthloss = loss_fn(idx.cuda(1).float(), ctrl.sum(1))
                #lengthloss = F.relu(ctrl.sum(1).mean() - args.mean_height)

                #regloss = loss_fn(t.eye(args.n_attr*args.height).float().cuda(args.gpu), t.matmul(model.f.weight, model.f.weight.transpose(0,1)))
                loss = loss# + 0.01*regloss
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            pbar.set_postfix(disloss=10*valid_disloss, conloss=10*valid_conloss, mean=mean_depth, var=var_depth, max=max_depth, min=min_depth)
            n_iter += 1
 
        with t.no_grad():
            validloader = DataLoader(word_vec, batch_size=args.mb, shuffle=False)
            total_disloss = 0
            total_conloss = 0
            mean_depth, max_depth, min_depth, var_depth = 0, 0, 0, 0
            
            for i,wv in enumerate(validloader):
                x = model.to_tensor(wv)
                ctrl_p, con, dis = model(x)
                y, _, _ = con
                yy, _, _ = dis
                
                conloss = loss_fn(x, y) 
                disloss = loss_fn(x, yy) 
                if args.active_depth:
#                    ctrl_ = args.height-ctrl_p.max(1)[1].data.cpu().float()
                    ctrl_ = ctrl_p.max(1)[1].data.cpu().float()
              #      ctrl_ = ctrl_p.sum(1).data.cpu().float()
                    max_depth += ctrl_.max().item()
                    min_depth += ctrl_.min().item()
                    mean_depth += ctrl_.mean().item()
                    var_depth += t.mean((ctrl_ - ctrl_.mean())**2).item()
                 
                total_disloss += disloss.item()
                total_conloss += conloss.item()
            valid_disloss = total_disloss/(i+1)
            valid_conloss = total_conloss/(i+1)
            mean_depth=mean_depth/(i+1)
            min_depth=min_depth/(i+1)
            max_depth=max_depth/(i+1)
            var_depth=var_depth/(i+1)


    model_config_param = {"param":model.state_dict(), "config":args}
    t.save(model_config_param, args.save_dir+"/model.pkl")

      
#        if n_iter>15: break

## Save word embedding
    testloader = DataLoader(word_vec, batch_size=args.mb, shuffle=False)
    pbar = tqdm(testloader)
    word2code = []
    word2ctrl = []
    word2vec = []
    word2vecyy = []
    codes = []
    codes_bin = bitarray()
    with t.no_grad():
       for i, wv in enumerate(pbar):
            x = model.to_tensor(wv)
            ctrl_p, con, dis = model(x)
            yy, code_discrete, ctrl_discrete = dis
            y, _, _ = con
            y=y
            yy=yy
            if args.active_depth:
                #code_bin = ((code_discrete>0).long() - (code_discrete==0).long() ).cpu().data.numpy()
                code_bin = ((code_discrete*ctrl_discrete) - (ctrl_discrete==0).float()).long().cpu().data.numpy()   
                ctrl = ctrl_discrete.cpu().long().data.numpy().astype(str)
                code = code_bin.astype(str)
            else:
                code_bin = (code_discrete>0).long().cpu().data.numpy()
                code = code_bin.astype(str)
            y = y.data.cpu().numpy().astype(str)
            yy = yy.data.cpu().numpy().astype(str)

            ccc = [[j for j in list(code[i]) if j!="-1" ] for i in range(len(x))]
            ccc_bin = [[j for j in list(code_bin[i]) if j!=-1 ] for i in range(len(x))]
            if args.active_depth:
                ccc_ctrl = [[j for j in list(ctrl[i]) if j!=-1 ] for i in range(len(x))]
            
            word_names = list(word_dict.keys())[args.mb*(i):args.mb*(i)+len(x)]
            w2c = [ " ".join([word_names[i]]  + ccc[i]) for i in range(len(x)) ]
            if args.active_depth:
                w2ctrl = [ " ".join([word_names[i]]  + ccc_ctrl[i]) for i in range(len(x)) ]
            w2v = [ " ".join([word_names[i]]  + list(y[i])) for i in range(len(x)) ]
            w2vyy = [ " ".join([word_names[i]]  + list(yy[i])) for i in range(len(x)) ]
            word2code = word2code + w2c
            if args.active_depth:
                word2ctrl = word2ctrl + w2ctrl
            word2vec = word2vec + w2v
            word2vecyy = word2vecyy + w2vyy
            codes += list(ccc)
#            codes_bin += list(ccc_bin)
            for i_bin in ccc_bin:
                codes_bin.extend(i_bin)

    w2c = sorted([[i for i in w.split()] for w in word2code], key=lambda x:[int(k) for k in x[1:]])
    sort_word2code = [" ".join(w) for w in w2c]
    with open(args.save_dir+'/codes.txt', 'w') as f:
        f.write("\n".join(word2code))
    if args.active_depth:
        with open(args.save_dir+'/ctrl.txt', 'w') as f:
            f.write("\n".join(word2ctrl))
    with open(args.save_dir+'/sort_codes.txt', 'w') as f:
        f.write("\n".join(sort_word2code))
    with open(args.save_dir+'/vecs.txt', 'w') as f:
        f.write("\n".join(word2vec))
    with open(args.save_dir+'/vecs_discrete.txt', 'w') as f:
        f.write("\n".join(word2vecyy))
    
    if args.bin:
        codes_ = codes_bin
    else:
        codes_ = bitarray()
        for j in codes:
            aa = [list(format(i,"b").zfill(len(format(args.n_attr-1,"b")))) for i in np.array(j,np.int64)]
            code_ = sum([[int(k) for k in i] for i in aa],[])
            codes_.extend(code_)
    np.savez_compressed(args.save_dir+ "/subcodes", np.packbits(codes_))
#    params = list(itertools.chain.from_iterable([i.data.cpu().numpy().reshape(-1) for i in model.g.parameters()]  ))
#    np.savez_compressed(args.save_dir+ '/codebook', params)


if __name__ == '__main__':
    train(parse_args())
