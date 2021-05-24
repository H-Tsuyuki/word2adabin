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
class AdaComp(nn.Module):
    def __init__(self, height, n_attr, gpu, active_depth, n_book):
        super(AdaComp, self).__init__()
       	self.gpu =  gpu
       	self.height = height 
       	self.n_attr = n_attr
       	self.active_depth = active_depth
        self.n_book = n_book
#        height2 = height-height//n_book 
#        height3 = height2-height//n_book 
#        height4 = height3-height//n_book 
        height2 = height//2
        height3 = height2//2 
        height4 = height3//2 
        self.f1 = nn.Linear(300, height, bias=False)
        self.f2 = nn.Linear(300, height2, bias=False)
        self.f3 = nn.Linear(300, height3, bias=False)
        self.f4 = nn.Linear(300, height4, bias=False)
        self.ctrlf = nn.Linear(300, n_book)
        self.f_1 = nn.Linear(300, 300)
        self.f_2 = nn.Linear(300, n_book)
        self.g1 = nn.Linear(height, 300, bias=True)
        self.g2 = nn.Linear(height2, 300, bias=True)
        self.g3 = nn.Linear(height3, 300, bias=True)
        self.g4 = nn.Linear(height4, 300, bias=True)
        self.height2 = height2  
        self.height3 = height3  
        self.height4 = height4  
 
    def encode(self, x):
        ctrl_p = 0
        
#        with t.no_grad():
        h1 = self.f1(x)
        h2 = self.f2(x)
        h3 = self.f3(x)
        h4 = self.f4(x)
         
        subcode1 = h1 + (F.relu(h1.sign()) - h1 ).detach()
        subcode2 = h2 + (F.relu(h2.sign()) - h2 ).detach()
        subcode3 = h3 + (F.relu(h3.sign()) - h3 ).detach()
        subcode4 = h4 + (F.relu(h4.sign()) - h4 ).detach()
        subcode = [subcode1,subcode2, subcode3,subcode4]
#        hh = self.ctrlf(x)
        hh = t.tanh(self.f_1(x))
        hh = t.log(F.softplus(self.f_2(hh)+1e-8))
        ctrl_p = F.gumbel_softmax(hh, tau=0.1, hard=False)
        return subcode, ctrl_p

    def decode(self, z, ctrl_p):
        
#        with t.no_grad():
#        y1 = t.matmul(z[0],self.f1.weight).unsqueeze(1) #+ self.g1.bias
#        y2 = t.matmul(z[1],self.f2.weight).unsqueeze(1) #+ self.g2.bias
#        y3 = t.matmul(z[2],self.f3.weight).unsqueeze(1) #+ self.g3.bias
#        y4 = t.matmul(z[3],self.f4.weight).unsqueeze(1) #+ self.g4.bias
        y1 = self.g1(z[0]).unsqueeze(1) 
        y2 = self.g2(z[1]).unsqueeze(1) 
        y3 = self.g3(z[2]).unsqueeze(1) 
        y4 = self.g4(z[3]).unsqueeze(1) 
        y = t.cat([y1,y2,y3,y4], 1)
        ctrl = ctrl_p.unsqueeze(2).repeat(1,1,300)
        y = y*ctrl
        y = y.sum(1)
            #y = t.matmul(z,self.encoder.f.f.weight)
        return y, z, ctrl_p
 
    def forward(self, x):
        code, ctrl_p = self.encode(x)

        ctrl_p_discrete = F.one_hot(ctrl_p.max(1)[1],num_classes=self.n_book).cuda(self.gpu).float()
        yy, code_discrete, ctrl_discrete = self.decode(code, ctrl_p_discrete)
        y, code, ctrl = self.decode(code, ctrl_p)
        

#        new_ctrl = ctrl_discrete[:,0].repeat(self.height,1).transpose(0,1).unsqueeze(1)
#        tmp = t.cat([ctrl_discrete[:,1].repeat(self.height2,1).transpose(0,1), t.zeros(code[0].shape[0],self.height2).cuda(self.gpu)],1)
#        new_ctrl = t.cat([new_ctrl, tmp.unsqueeze(1)],1)
#        tmp = t.cat([ctrl_discrete[:,2].repeat(self.height3,1).transpose(0,1), t.zeros(code[0].shape[0],self.height3*3).cuda(self.gpu)],1)
#        new_ctrl = t.cat([new_ctrl, tmp.unsqueeze(1)],1)
#        tmp = t.cat([ctrl_discrete[:,3].repeat(self.height4,1).transpose(0,1), t.zeros(code[0].shape[0],self.height4*8).cuda(self.gpu)],1)
#        new_ctrl = t.cat([new_ctrl, tmp.unsqueeze(1)],1)
        new_ctrl = ctrl_discrete[:,0].repeat(self.height,1).transpose(0,1)
        new_ctrl = t.cat([new_ctrl, ctrl_discrete[:,1].repeat(self.height2,1).transpose(0,1)],1)
        new_ctrl = t.cat([new_ctrl, ctrl_discrete[:,2].repeat(self.height3,1).transpose(0,1)],1)
        new_ctrl = t.cat([new_ctrl, ctrl_discrete[:,3].repeat(self.height4,1).transpose(0,1)],1)
#            new_ctrl = t.flip(t.flip(t.eye(code[0].shape[0], self.height)[[new_ctrl.sum(1).long()-1]], [1]).cumsum(1), [1]).cuda(self.gpu)
        new_ctrl = t.flip(t.flip(F.one_hot(new_ctrl.sum(1).long()-1,num_classes=self.height), [1]).cumsum(1), [1]).cuda(self.gpu).float() 
        code_discrete = code_discrete[0] ##どうしたらいいかわからんからとりあえずの処置
        #return ctrl_p, (y, code, ctrl), (yy, code_discrete, ctrl_discrete)
        return ctrl_p, (y, code, ctrl), (yy, code_discrete, new_ctrl)
   
    def to_tensor(self, x):
        x = t.cat(x).reshape(len(x), x[0].shape[0]).transpose(0,1)
        x = x.cuda(self.gpu).float()
        return x
