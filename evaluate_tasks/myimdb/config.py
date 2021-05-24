def config(args):
    if args.task == "sst5":
        args.h_dim = 300
        args.n_layer = 1
        args.n_class = 5
    elif args.task == "sst2":
        args.h_dim = 300 
        args.n_layer = 1
        args.n_class = 2 
    elif args.task == "trec":
        args.h_dim = 300
        args.n_layer = 1 
        args.n_class = 6 
    elif args.task == "imdb":
        args.h_dim = 300
        args.n_layer = 1 
        args.n_class = 2
    elif args.task == "ag":
        args.h_dim = 300
        args.n_layer = 2 
        args.n_class = 4
    elif args.task == "dbpedia":
        args.h_dim = 300
        args.n_layer = 1 
        args.n_class = 14
    elif args.task == "yelp":
        args.h_dim = 300
        args.n_layer = 1
        args.n_class = 2 
    elif args.task == "nli":
        args.h_dim = 300
        args.n_layer = 1 
        args.n_class = 3
    
    if args.method=="bin" or args.method=="glove" or args.method=="bin_no_the":
        args.active_depth=False
    else:
        args.active_depth=True
    return args
