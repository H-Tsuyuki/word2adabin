import os
import json
import random
import argparse
import torch as t


def config(args):
    if args.name == "bin":
        args.bin = True
        args.active_depth = False
    elif args.name == "adacomp":
        args.bin = True
        args.active_depth = True
    elif args.name == "NC":
        args.bin = False
        args.active_depth = False
    elif args.name == "ours":
        args.bin = True
        args.active_depth = True
    
    args.multi_codebook = not args.bin
    return args

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default="bin", choices=["bin", "bin_no_the", "adacomp", "NC", "ours"], help="binはnearlossbinarization法．bin_no_theはstraight through estimatorを使わないbin．oursが提案法．")
#    parser.add_argument('--glove_path', type=str, default='../data/glove.42B.300d.txt', help="data directory path")
    parser.add_argument('--glove_path', type=str, default='../data/glove.6B.300d.txt', help="data directory path")
    parser.add_argument('--task', type=str, default='ag',choices=["imdb", "trec", "ag", "dbpedia","yelp", "sst5"], help="data directory path")
    parser.add_argument('--load_dir', type=str, default='evaluate_tasks/myimdb/results_dict/')
    parser.add_argument('--vocab_size', type=int, default='400000', help="data directory path")
    parser.add_argument('--save_dir', type=str, default='./results_new', help="model directory path")
    parser.add_argument('--height', type=int, default=64, help="Codes dimension")
    parser.add_argument('--n_attr', type=int, default=1, help="Code vector dimension")
    parser.add_argument('--attr_dim', type=int, default=300, help="Code vector dimension")
    parser.add_argument('--z_dim', type=int, default=300, help="Code vector dimension")
    parser.add_argument('--epoch', type=int, default=10, help="number of epochs")
    parser.add_argument('--mb', type=int, default=128, help="mini-batch size")
    parser.add_argument('--lr', type=float, default=1e-3, help="mini-batch size")
    parser.add_argument('--accum_count', type=int, default=0, help="mini-batch size")
    parser.add_argument('--gpu', type=int, default=1, help="gpu id")
    parser.add_argument('--mean_height', type=int, default=64, help="Codes dimension")
    parser.add_argument('--bin', action="store_true", help="bin or multi_codebook")
    parser.add_argument('--active_depth', action="store_true", help="gpu id")
    parser.add_argument('--ctrl_gate', action="store_true", help="gpu id")
    parser.add_argument('--ternary', action="store_true", help="gpu id")
    args = parser.parse_args()
    args = config(args) 
    
    args.word_vec_path = args.load_dir+args.task+"/word_vec.pkl"
#    args.word_vec_path = args.load_dir+args.task+"/glovegtune_vec.pkl"
  #  args.word_vec_path = args.load_dir+args.task+"/glovegtune_mean_vec.pkl"
    #args.vocab_path = args.load_dir+args.task+"/3ika_vocab.pkl"
    args.vocab_path = args.load_dir+args.task+"/vocab.pkl"
#    args.word_vec_path = args.load_dir+args.task+"/bpe_vec.pkl"
#    args.vocab_path = args.load_dir+args.task+"/bpe_vocab.pkl"

    if args.name=="ours" or args.name=="adacomp":
        args.active_depth=True
    
    args.save_dir += "/"+args.task
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
        
    args.save_dir += "/"+args.name
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
    
    args.save_dir += "/"+str(args.height)
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
 
    args.save_dir=args.save_dir+"/"
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)

    log_file = args.save_dir+"/config"
    with open(log_file, 'w', encoding='utf-8') as f:
        print(json.dumps(args.__dict__, indent=4), file=f) 
    
    seed = 1
    t.manual_seed(seed)
    random.seed(seed)
    t.backends.cudnn.deterministic = True
    t.cuda.manual_seed(seed)
    return args
