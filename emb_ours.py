# -*- coding: utf-8 -*-
import copy
import sys

sys.setrecursionlimit(2000)
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F

from torch import LongTensor as LT
from torch import FloatTensor as FT


###############################################################################################################
class CodeAutoEncoder(nn.Module):
    def __init__(self, height, n_attr, gpu, active_depth, n_book, args):
        super(CodeAutoEncoder, self).__init__()
       	self.gpu =  gpu
       	self.height = height 
       	self.n_attr = n_attr
       	self.active_depth = active_depth
        self.n_book = n_book
        
        if args.name=="bin_no_the": 
            self.f = nn.Linear(300, height, bias=False)
        else:
            self.f = nn.Linear(300, height, bias=True)
        self.ctrlf = nn.Linear(300, n_book)
        self.g = nn.Linear(height, 300)
        
        self.f1 = nn.Linear(300, 300)
        self.f2 = nn.Linear(300, n_book)
        
        self.args = args

    def encode(self, x):
        ctrl_p = 0
        
        h = self.f(x).view(x.shape[0],-1)
        if self.args.name=="bin_no_the": 
            #subcode = h + (F.relu(h.sign()) - h ).detach()
            subcode = F.relu(h.sign())
        else:
            subcode = h + (F.relu(h.sign()) - h ).detach()
            #subcode = F.relu(h.sign())
        if self.active_depth:
            hh = t.tanh(self.f1(x))
            hh = t.log(F.softplus(self.f2(hh))+1e-8)
            ctrl_p = F.gumbel_softmax(hh, tau=0.1, hard=False)
        return subcode, ctrl_p

    def decode(self, z, ctrl_p):
        ctrl = 0
        new_ctrl = 0
        height = self.height
        if self.active_depth:
            #from IPython.core.debugger import Pdb; Pdb().set_trace()
            ctrl = t.cumsum(t.flip(ctrl_p,[1]),1)
            ctrl = t.flip(ctrl,[1])
#            ctrl = t.cumsum(ctrl_p,1)
            new_ctrl=ctrl
            if self.n_book != self.height: 
                for i in range(ctrl.shape[1]):
                    if i+1!=ctrl.shape[1]:
                        height = height//2
                    if i==0:
                        new_ctrl = ctrl[:,i].repeat(height,1).transpose(0,1)
                    else:
                        new_ctrl = t.cat([new_ctrl,ctrl[:,i].repeat(height,1).transpose(0,1)],1)
#                new_ctrl=ctrl.repeat_interleave(self.height//self.n_book,1)
            z = z*new_ctrl
        
        if self.args.name=="bin_no_the": 
            y = t.matmul(z,self.f.weight)# + self.g.bias
        else:
#            y = t.matmul(z,self.f.weight) + self.g.bias
            y = self.g(z)
        
        return y, z, new_ctrl
 
    def forward(self, x):
        code, ctrl_p = self.encode(x)
        code_discrete, ctrl_p_discrete = code, ctrl_p

        if self.active_depth:
            ctrl_p_discrete = F.one_hot(ctrl_p.max(1)[1],num_classes=self.n_book).cuda(self.gpu).float()
        yy, code_discrete, ctrl_discrete = self.decode(code, ctrl_p_discrete)
        y, code, ctrl = self.decode(code, ctrl_p)
                
        return ctrl_p, (y, code, ctrl), (yy, code_discrete, ctrl_discrete)
   
    def to_tensor(self, x):
        x = t.cat(x).reshape(len(x), x[0].shape[0]).transpose(0,1)
        x = x.cuda(self.gpu).float()
        return x
