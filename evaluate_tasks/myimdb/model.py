import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderCNN(nn.Module):
    def __init__(self, emb_dim, h_dim, n_layer, n_class, args):
        super(EncoderCNN, self).__init__()
        self.h_dim = h_dim
        self.args = args
        input_channel = 1
        output_channel = h_dim//2
        self.conv1 = nn.Conv2d(input_channel, output_channel, (3, h_dim), stride=1, padding=(2,0))
        self.conv2 = nn.Conv2d(input_channel, output_channel, (4, h_dim), stride=1, padding=(3,0))
        self.conv3 = nn.Conv2d(input_channel, output_channel, (5, h_dim), stride=1, padding=(4,0))
        self.fc = nn.Linear(output_channel*3, n_class)
        self.dropout = nn.Dropout(0.1)

    def conv_block(self, input, conv_layer):
        conv_out = conv_layer(input)# conv_out.size() = (batch_size, out_channels, dim, 1)
        activation = F.relu(conv_out.squeeze(3))# activation.size() = (batch_size, out_channels, dim1)
        max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2)# maxpool_out.size() = (batch_size, out_channels)	
        return max_out 

    def forward(self, emb, lengths=None):
        emb = emb.unsqueeze(1)
        max_out1 = self.conv_block(emb, self.conv1)
        max_out2 = self.conv_block(emb, self.conv2)
        max_out3 = self.conv_block(emb, self.conv3)
		
        h = torch.cat((max_out1, max_out2, max_out3), 1)
        # all_out.size() = (batch_size, num_kernels*out_channels)
  #      h = self.dropout(h)
        output = self.fc(h)
        return output
 
class EncoderRNN(nn.Module):
    def __init__(self, emb_dim, h_dim, n_layer, n_class, args):
        super(EncoderRNN, self).__init__()
        self.h_dim = h_dim
        self.lstm1 = nn.LSTM(emb_dim, h_dim, n_layer, batch_first=True)
        if args.task =="nli":
            self.fc = nn.Linear(2*h_dim, n_class)
            self.lstm2 = nn.LSTM(emb_dim, h_dim, n_layer, batch_first=True)
        else:
            self.fc = nn.Linear(h_dim, n_class)
        self.args = args

    def forward(self, emb, lengths=None):
        if len(emb.shape)==4:
            if False:
                lengths = torch.tensor(lengths).cuda(self.args.gpu)
                h1 = emb[:,0].sum(1)/lengths[:,0].repeat(emb[:,0].shape[2],1).transpose(0,1).float()
                h2 = emb[:,1].sum(1)/lengths[:,1].repeat(emb[:,1].shape[2],1).transpose(0,1).float()
            else: 
                o1, (h1, c1) = self.lstm1(emb[:,0])
                o2, (h2, c2) = self.lstm2(emb[:,1])
                h1 = o1.mean(dim=1)
                h2 = o2.mean(dim=1)
            h = torch.cat([h1,h2],1)
        else:
            if  self.args.task=="dbpedia":
       #         from IPython.core.debugger import Pdb; Pdb().set_trace()

                lengths = torch.tensor(lengths).cuda(self.args.gpu)
                h = emb.sum(1)/lengths.repeat(emb.shape[2],1).transpose(0,1).float()
            else: 
                o, (h, c) = self.lstm1(emb)
                h = o.mean(dim=1)
                #h = h[-1]
        output = self.fc(h)
        return output


class Classifier(nn.Module):
    def __init__(self, emb_dim, h_dim, vocab_size, n_class, n_layer, pretrained_vector, args, word_model=None):
        super(Classifier, self).__init__()
        self.my_embed = word_model
        self.args = args
        self.emb_dim = emb_dim
        if pretrained_vector is None:
            self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=1)
        else:
            #self.embed = nn.Embedding.from_pretrained(pretrained_vector, padding_idx=1)
            self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=1)
            self.embed.weight = nn.Parameter(pretrained_vector)
     #   self.rnn = EncoderRNN(emb_dim, h_dim, n_layer, n_class, args)
        self.rnn = EncoderRNN(emb_dim, h_dim, n_layer, n_class, args)
        self.m = nn.Softmax(dim=1)
        self.fc = nn.Linear(300, 300)

    
    def forward(self, x, lengths=None, train_flg=None, epoch=None):
        ctrl, ctrl_p, code = None, None, None 
        #with torch.no_grad():
        emb = self.embed(x)
   #     emb = self.fc(emb)
        
        if len(emb.shape)==4:
            emb1 = emb[:,1]
            emb = emb[:,0]
            if self.my_embed is not None:
                emb = emb.view(-1, self.emb_dim)
                emb, code, ctrl, ctrl_p = self.to_bin(emb, train_flg, epoch)
                emb1 = emb1.view(-1, self.emb_dim)
                emb1, code1, ctrl1, ctrl_p1 = self.to_bin(emb1, train_flg, epoch)
            emb = emb.view(x.shape[0],1, -1, self.emb_dim)
            emb1 = emb1.view(x.shape[0],1, -1, self.emb_dim)
            emb = torch.cat([emb,emb1],1) 
        else:
            if self.my_embed is not None:
                emb = emb.view(-1, self.emb_dim)
                emb, code, ctrl, ctrl_p = self.to_bin(emb, train_flg, epoch)
                emb = emb.view(x.shape[0], -1, self.emb_dim)

        output = self.rnn(emb, lengths)
        output = self.m(output)
        return output, code, ctrl, ctrl_p, emb 

    def to_bin(self, emb, train_flg=None, epoch=None):
        ctrl_p, (emb, code, ctrl), (emb_discrete, code_discrete, ctrl_discrete) = self.my_embed(emb)

        if train_flg:
            return emb, code, ctrl, ctrl_p
        else:
            return emb_discrete, code_discrete, ctrl_discrete, ctrl_p
        
