import os
import pickle
import random
import itertools
import numpy as np
from tqdm import tqdm
from bitarray import bitarray

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from emb_ours import CodeAutoEncoder 
from adacomp import AdaComp

from model import Classifier
from config import config
from utils import get_glove_all, build_vocab, to_id, MyIterator
from load_data import read_imdb, read_sst5, get_nli, read_csv, read_trec, read_sst2


def train_model(epoch, train, optim, model, vocab):
    train_loader = MyIterator(train, vocab, args.batch_size, shuffle=True)

    total_correct = 0
    total_n_sample = 0
    loss_drop = 0
    loss_fn = nn.BCELoss()
    loss_fn = loss_fn.cuda(args.gpu)

    pbar = tqdm(train_loader)
    pbar.set_description("[Epoch {}]".format(epoch))
    loss_mse = nn.MSELoss()
    loss_mse = loss_mse.cuda(args.gpu)
    for idx, (x,y,l) in enumerate(pbar):

        x = torch.tensor(x).cuda(args.gpu)
        y = torch.tensor(y).cuda(args.gpu)
        gold_out = torch.eye(len(y), args.n_class)[y].cuda(args.gpu)
        out, code, ctrl, ctrl_p, emb = model(x, l, train_flg=True)
        loss = loss_fn(out, gold_out)
 #       loss2 = loss_mse(emb, model.embed(x))
        loss = loss #+ 1e-2*loss2
        if False:
            #glloss = 1e-1*(((model.embed.weight**2).cumsum(1)+1e-13).sqrt().mean())
            glloss = 1e-2*((model.embed.weight.norm(p=2,dim=1)**2).mean())
            loss = loss + glloss
        if ctrl is not None:
            #indice = random.sample(range(len(vocab)), 5000)
            #indice = torch.tensor(indice).cuda(args.gpu)
            #aaa = torch.log(F.softplus(model.fc2(torch.tanh(model.fc1(model.embed(indice)))))+1e-13)
            #aaa = F.gumbel_softmax(aaa, tau=0.1, hard=False)
            #aaa = aaa.cumsum(1)
            #loss_L = F.relu(aaa.sum(1).mean() - 100)
            loss = loss #+ 0.*loss_L 
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        pred = out.data.max(1)[1]
        correct = pred.eq(y.data).sum().item()
        n_sample = len(x)
        total_correct += correct
        total_n_sample += n_sample
        acc = 100*correct/float(len(x))
        pbar.set_postfix(acc=acc, loss=loss.item(), dloss=loss_drop)

def test_model(epoch, test, model, vocab, dev, args, best_acc, test_acc, test_epoch):
    correct = 0
    acc = 0
    dev_acc = 0
    test_acc = 0
    mean_depth=0
    min_depth=0
    max_depth=0
    var_depth=0
    if True:
        dev_loader = MyIterator(dev, vocab, args.batch_size, shuffle=False)
    
        with torch.no_grad():
            for idx, (x,y,l) in enumerate(dev_loader):
                x = torch.tensor(x).cuda(args.gpu)
                y = torch.tensor(y).cuda(args.gpu)
                out, code, ctrl, ctrl_p, _ = model(x, l, train_flg=False)
                if args.active_depth:                   
#                    indice = random.sample(range(len(vocab)), 5000)
#                    indice = torch.tensor(indice).cuda(args.gpu)
#                    hh = model.embed_code(indice)
#                    aaa = torch.log(torch.arange(1,301).cuda(args.gpu).float())
#                    ctrl = F.relu((aaa - hh).sign())
                    #aaa = torch.log(F.softplus(model.fc2(torch.tanh(model.fc1(model.embed(torch.unique(x))))))+1e-13)
                    #aaa = F.gumbel_softmax(aaa, tau=0.1, hard=False)
#                    ctrl_ = model.embed.weight[:,-1].abs().data.cpu()*300
#                    aaa = aaa.cumsum(1)
#                if True:                                      
                    #ctrl_ = ctrl_p.max(1)[1].data.cpu().float()            
#                    ctrl_ = aaa.sum(1).data.cpu().float()            
                    ctrl_ = ctrl.sum(1).data.cpu().float()            
                    max_depth += ctrl_.max().item()                        
                    min_depth += ctrl_.min().item()                        
                    mean_depth += ctrl_.mean().item()                      
                    var_depth += torch.mean((ctrl_ - ctrl_.mean())**2).item()  
                                                                           
                pred = out.data.max(1)[1]
                correct += pred.eq(y.data).sum().item()
            
            mean_depth=round(mean_depth/(idx+1),2)                                    
            min_depth=round(min_depth/(idx+1),2)                                    
            max_depth=round(max_depth/(idx+1),2)                                  
            var_depth=round(var_depth/(idx+1),2)                    
            
            dev_acc = 100*correct/float(len(dev))
            print('mean:{} min:{} max:{} var:{}'.format(mean_depth, min_depth, max_depth, var_depth))
            print('Dev acc:{} '.format(round(dev_acc,2) ))
    
#    if dev_acc >= best_acc:
#        best_acc = dev_acc
#        test_epoch = epoch
#    if args.task=="sst5" or args.task=="sst2" and dev_acc >=0: # best_acc:
    
    if dev_acc >=0: # best_acc:
    #if True: 
        test_epoch = epoch
        best_acc = dev_acc
        
        test_epoch = epoch
        
        correct = 0
        test_loader = MyIterator(test, vocab, args.batch_size, shuffle=False)
        with torch.no_grad():
            for idx, (x,y,l) in enumerate(test_loader):
                x = torch.tensor(x).cuda(args.gpu)
                y = torch.tensor(y).cuda(args.gpu)
                out, code, ctrl, ctrl_p, _ = model(x, l, train_flg=False)
                pred = out.data.max(1)[1]
                correct += pred.eq(y.data).sum().item()
            
            test_acc = 100*correct/float(len(test))
            print('Test acc:{} '.format(round(test_acc,2)))
    return best_acc, test_acc, test_epoch
       

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch IMDB Example')
    parser.add_argument('--task', type=str, default="imdb", choices=["sst2", "sst5", "trec", "imdb", "nli", "ag", "dbpedia", "yelp"], help="") 
    parser.add_argument('--method', type=str, default="bin", choices=["glove","bin", "bin_no_the", "adacomp", "ours"], help="") 
    
    parser.add_argument('--sst2_path', type=str, default="/mnt/aoni04/tsuyuki/word2tree/evaluate_tasks/myimdb/data/sst2/")
    parser.add_argument('--imdb_path', type=str, default="/mnt/aoni04/tsuyuki/word2tree/evaluate_tasks/myimdb/data/imdb/aclImdb/")
    parser.add_argument('--trec_path', type=str, default="/mnt/aoni04/tsuyuki/word2tree/evaluate_tasks/myimdb/data/TREC/")
    parser.add_argument('--ag_path', type=str, default="/mnt/aoni04/tsuyuki/word2tree/evaluate_tasks/myimdb/data/ag_news_csv/")
    parser.add_argument('--dbpedia_path', type=str, default="/mnt/aoni04/tsuyuki/word2tree/evaluate_tasks/myimdb/data/dbpedia/")
    parser.add_argument('--yelp_path', type=str, default="/mnt/aoni04/tsuyuki/word2tree/evaluate_tasks/myimdb/data/yelp_review_polarity_csv/")
    parser.add_argument('--nli_path', type=str, default="/mnt/aoni04/tsuyuki/InferSent/dataset/SNLI/")
    parser.add_argument('--sst5_path', type=str, default="/mnt/aoni04/tsuyuki/word2tree/evaluate_tasks/myimdb/data/sst5/")

    parser.add_argument('--save_dir', type=str, default="/mnt/aoni04/tsuyuki/word2tree/evaluate_tasks/myimdb/results_bpe/")
    parser.add_argument('--save_dir2', type=str, default="/mnt/aoni04/tsuyuki/word2tree/evaluate_tasks/myimdb/results_nbook4/")
    parser.add_argument('--save_dir3', type=str, default="/mnt/aoni04/tsuyuki/word2tree/evaluate_tasks/myimdb/results_pretrain/")

#    parser.add_argument('--wv_path', type=str, default="/mnt/aoni04/tsuyuki/data/glove.42B.300d.txt",
    parser.add_argument('--wv_path', type=str, default="/mnt/aoni04/tsuyuki/data/glove.6B.300d.txt",
                        help='glove path')
    parser.add_argument('--model_path', type=str, default="/mnt/aoni04/tsuyuki/word2tree/results_new/")

    parser.add_argument('--min_freq', type=int, default=0, metavar='N')
    parser.add_argument('--height', type=int, default=64, metavar='N')
    parser.add_argument('--n_book', type=int, default=0, metavar='N', help="nunber of block")
    parser.add_argument('--h_dim', type=int, default=150, metavar='N', help='hidden state dim')
    parser.add_argument('--emb_dim', type=int, default=300, metavar='N', help="word_emb_dim")
    parser.add_argument('--batch_size', type=int, default=64, metavar='N')
    parser.add_argument('--epochs', type=int, default=10, metavar='N')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR')
    parser.add_argument('--gpu', type=int, default=0, metavar='N',
                        help='gpu id')
    parser.add_argument('--active_depth', action="store_true")
    parser.add_argument('--pretrained', action="store_true")

    args = parser.parse_args()

    args = config(args)
    if args.method=="ours":
        args.model_path += args.task+"/bin/"+str(args.height)+"/model.pkl"
        #args.model_path += "_glove/"+args.task+"/ours/"+str(args.height)+"/model.pkl"
#        args.model_path += args.task+"/bin_no_the/"+str(args.height)+"/model.pkl"
    elif args.method!="bin_no_the":
  #      args.model_path += "_glove/"+args.task+"/"+args.method+"/"+str(args.height)+"/model.pkl"
        args.model_path += args.task+"/"+args.method+"/"+str(args.height)+"/model.pkl"
    else:
        args.model_path += args.task+"/"+args.method+"/"+str(args.height)+"/model.pkl"

    if args.n_book==4:
        args.save_dir = args.save_dir2
    if args.pretrained:
        args.save_dir = args.save_dir3

    args.save_dir = args.save_dir+args.task+"/"
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
    args.save_dir = args.save_dir+args.method+"/"
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
    args.save_dir = args.save_dir+str(args.height)+"/"
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)

    """ Seed """
    seed = 1234
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True  
    torch.cuda.manual_seed(seed)

    """ Prepare Data Iterator """
    # Load Data
    if args.task=="nli":
        train, dev, test = get_nli(args.nli_path)
        vocab, word_vec, counts = build_vocab(train, args.wv_path, args.min_freq)
    elif args.task=="dbpedia" or args.task=="ag" or args.task=="yelp":
        train = read_imdb(args, "train")
        test = read_imdb(args, "test")
        #train = read_csv(args, "train")
        #test = read_csv(args, "test")
        vocab, word_vec, counts = build_vocab(train, args.wv_path, args.min_freq)
    elif args.task=="sst2":
        train = read_sst2(args, "train")
        dev = read_sst2(args, "dev")
        test = read_sst2(args, "test")
        vocab, word_vec, counts = build_vocab(train, args.wv_path, args.min_freq)
    elif args.task=="sst5":
        train = read_sst5(args, "train")
        dev = read_sst5(args, "dev")
        test = read_sst5(args, "test")
        vocab, word_vec, counts = build_vocab(train, args.wv_path, args.min_freq)
    elif args.task=="trec":
        train = read_trec(args, "train")
        test = read_trec(args, "test")
        vocab, word_vec, counts = build_vocab(train, args.wv_path, args.min_freq)
    elif args.task=="imdb":
        train = read_imdb(args, "train")
        test = read_imdb(args, "test")
        vocab, word_vec, counts = build_vocab(train, args.wv_path, args.min_freq)
    # build the vocabulary & load GloVe
#    vocab, word_vec = get_glove_all(args.wv_path)
#    pickle.dump(counts, open("results_dict/"+args.task+"/full_counts.pkl","wb"))
#    pickle.dump(vocab, open("results_dict/"+args.task+"/full_vocab.pkl","wb"))
#    pickle.dump(word_vec, open("results_dict/"+args.task+"/word_vec.pkl","wb"))
#    np.savez_compressed("results_dict/"+args.task+"/comp_wvec", np.array(word_vec, dtype=np.float32))
    
    #if args.method!="glove" args.method=="bin_no_the" or args.method=="bin":
    if args.method=="bin_no_the": #or args.method=="bin":
        with open("results_dict/"+args.task+"/glovegtune_vec.pkl","rb") as f:
            word_vec = pickle.load(f)
    word_vec = torch.tensor(word_vec).float().cuda(args.gpu)
    train = to_id(train, vocab)
    test = to_id(test, vocab)
    if args.task=="sst5" or args.task=="sst2" or args.task=="nli":
        dev = to_id(dev, vocab)
    else:
        dev=test
     
    """ Model """
    if args.pretrained:
        word_models = torch.load(args.model_path, map_location="cuda:"+str(args.gpu))
        word_model_args = word_models["config"]
        if args.method=="adacomp":
            word_model = AdaComp(args.height, 1, args.gpu, args.active_depth, 4)
        else:
            word_model = CodeAutoEncoder(word_model_args.height, word_model_args.n_attr, args.gpu, args.active_depth, args.n_book, args)
        word_model.load_state_dict(word_models["param"])
    else:
        if args.method=="adacomp":
            word_model = AdaComp(args.height, 1, args.gpu, args.active_depth, 4)
        elif args.method=="glove":
            word_model=None
        else:
            word_model = CodeAutoEncoder(args.height, 1, args.gpu, args.active_depth, args.n_book, args)
    model = Classifier(args.emb_dim, args.h_dim, len(vocab), args.n_class, args.n_layer, word_vec, args, word_model)
    model = model.cuda(args.gpu)

    """ Optim """
#    optim = optim.Adam(model.parameters(), lr=args.lr)
    optim = optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    """ Train model """ 
    best_acc = 0
    test_acc = 0
    test_epoch = 0
    for epoch in range(1,args.epochs+1):
        train_model(epoch, train, optim, model, vocab)
        best_acc, test_acc, test_epoch = test_model(epoch, test, model, vocab, dev, args, best_acc, test_acc, test_epoch)
    vec = model.embed.weight.data.cpu().tolist()
    #np.savez_compressed("results_dict/"+args.task+"/comp_bpevec_scratch", np.array(vec, dtype=np.float32))
#    pickle.dump(vec, open("results_dict/"+args.task+"/bpe_vec.pkl","wb"))
#    pickle.dump(vocab, open("results_dict/"+args.task+"/bpe_vocab.pkl","wb"))
#    pickle.dump(vec, open("results_dict/"+args.task+"/glovegtune_vec.pkl","wb"))
#    pickle.dump(vocab, open("results_dict/"+args.task+"/glovegtune_vocab.pkl","wb"))
    #print('Test epoch:{} acc:{}'.format(test_epoch, round(test_acc,2)))
    with open(args.save_dir+'best_acc.txt', 'w') as f:                                                           
        print('Best epoch:{} acc:{}'.format(test_epoch, round(best_acc,2)), file=f)
    print('Best epoch:{} acc:{}'.format(test_epoch, round(best_acc,2)))
## Save word embedding
    testloader = DataLoader(word_vec, batch_size=args.batch_size, shuffle=False)                                            
    pbar = tqdm(testloader)                                                                                         
    word2code = []                                                                                                  
    word2ctrl = []                                                                                                  
    word2vec = []                                                                                                  
    word2vecyy = []                                                                                                  
    codes = []                                                                                                      
    codes_bin = bitarray()                                                                                          
    with torch.no_grad():                                                                                               
       for i, x in enumerate(pbar):                                                                                
            ctrl_p, con, dis = word_model(x)                                                                             
            yy, code_discrete, ctrl_discrete = dis                                                                  
            y, _, _ = con                                                                                           
            if args.active_depth:                                                                                   
                #code_bin = ((code_discrete>0).long() - (code_discrete==0).long() ).cpu().data.numpy()               
#                code_bin = (code_discrete>0).long().cpu().data.numpy()                                              
                code_bin = ((code_discrete*ctrl_discrete) - (ctrl_discrete==0).float()).long().cpu().data.numpy()

                ctrl_n = ctrl_discrete.sum(1).cpu().long().data.numpy().astype(str)
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
                                                                                                                    
            word_names = list(vocab.keys())[args.batch_size*(i):args.batch_size*(i)+len(x)]                                     
            w2c = [ " ".join([word_names[i]]  + ccc[i]) for i in range(len(x)) ]
                                                
            if args.active_depth:                                                                                   
                w2ctrl = [ " ".join([word_names[i]]  + ccc_ctrl[i]) for i in range(len(x)) ]  
            
                w2v = [ " ".join([word_names[i]]  + list(y[i]) + [ctrl_n[i]]) for i in range(len(x)) ]
                w2vyy = [ " ".join([word_names[i]]  + list(yy[i]) + [ctrl_n[i]]) for i in range(len(x)) ] 
            else:
                w2v = [ " ".join([word_names[i]]  + list(y[i])) for i in range(len(x)) ]
                w2vyy = [ " ".join([word_names[i]]  + list(yy[i])) for i in range(len(x)) ] 

            word2code = word2code + w2c                                                                             
            if args.active_depth:
                word2ctrl = word2ctrl + w2ctrl 
            word2vec = word2vec + w2v 
            word2vecyy = word2vecyy + w2vyy 
            codes += list(ccc)                                                                                      
            for i_bin in ccc_bin:                                                                                   
                codes_bin.extend(i_bin)                                                                             
                                                                                                                    
    w2c = sorted([[i for i in w.split()] for w in word2code], key=lambda x:[int(k) for k in x[1:]])                 
    w2ctrl = sorted([[i for i in w.split()] for w in word2ctrl], key=lambda x:[int(k) for k in x[1:]])                 
    sort_word2code = [" ".join(w) for w in w2c]                                                                     
    sort_word2ctrl = [" ".join(w) for w in w2ctrl]                                                                     
    with open(args.save_dir+'codes.txt', 'w') as f:                                                                
        f.write("\n".join(word2code))                                                                               
    if args.active_depth:
        with open(args.save_dir+'ctrl.txt', 'w') as f:                                                                
            f.write("\n".join(word2ctrl))                                                                               
        with open(args.save_dir+'sort_ctrl.txt', 'w') as f:                                                           
            f.write("\n".join(sort_word2ctrl))                                                                               
    with open(args.save_dir+'sort_codes.txt', 'w') as f:                                                           
        f.write("\n".join(sort_word2code))                                                                          
    with open(args.save_dir+'vecs.txt', 'w') as f:                                                           
        f.write("\n".join(word2vec))                                                                          
    with open(args.save_dir+'vecs_discrete.txt', 'w') as f:                                                           
        f.write("\n".join(word2vecyy))                                                                          
                                                                                                                    
    codes_ = codes_bin                                                                                          
    np.savez_compressed(args.save_dir+ "subcodes", np.packbits(codes_))
    if args.method=="adacomp":
        params1 = [i.data.cpu().numpy().reshape(-1) for i in word_model.g1.parameters()] 
        params2 = [i.data.cpu().numpy().reshape(-1) for i in word_model.g2.parameters()] 
        params3 = [i.data.cpu().numpy().reshape(-1) for i in word_model.g3.parameters()] 
        params4 = [i.data.cpu().numpy().reshape(-1) for i in word_model.g4.parameters()]
        params = params1+params2+params3+params4
        params = list(itertools.chain.from_iterable(params))
    else:
        params = list(itertools.chain.from_iterable([i.data.cpu().numpy().reshape(-1) for i in word_model.g.parameters()]  ))
    np.savez_compressed(args.save_dir+ 'codebook', np.array(params,dtype=np.float32))

