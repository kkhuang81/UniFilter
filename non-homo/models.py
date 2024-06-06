import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import numpy as np

class Dense(nn.Module):

    def __init__(self, in_features, out_features, bias='none'):
        super(Dense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias == 'bn':
            self.bias = nn.BatchNorm1d(out_features)
        else:
            self.bias = lambda x: x
            
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))        
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input):                        
        output = torch.mm(input, self.weight)
        output = self.bias(output)
        if self.in_features == self.out_features:
            output = output + input
        return output

class MLP(nn.Module):
    def __init__(self, nfeat, nlayers,nhidden, nclass, dropout, bias):
        super(MLP, self).__init__()
        self.fcs = nn.ModuleList()
        self.fcs.append(Dense(nfeat, nhidden, bias))
        for _ in range(nlayers-2):
            self.fcs.append(Dense(nhidden, nhidden, bias))
        self.fcs.append(Dense(nhidden, nclass))
        self.act_fn = nn.ReLU()
        self.dropout = dropout

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)        
        x = self.act_fn(self.fcs[0](x))
        for fc in self.fcs[1:-1]:
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.act_fn(fc(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fcs[-1](x)
        return x

class Combination(nn.Module):
    '''
    A mod combination the bases of polynomial filters.
    Args:
        channels (int): number of feature channels.
        level (int): number of bases to combine.
        sole (bool): whether or not use the same filter for all output channels.
    '''
    def __init__(self, channels, level, dropout, sole=False):
        super().__init__()
        self.dropout = dropout
        self.K=level
        self.comb_weight = nn.Parameter(torch.ones((1, level, 1)))
        self.reset_parameters()            

    def reset_parameters(self):
        bound = 1.0/self.K
        TEMP = np.random.uniform(bound, bound, self.K)       
        self.comb_weight=nn.Parameter(torch.FloatTensor(TEMP).view(-1,self.K, 1))

    def forward(self, x):
        '''
        x: node features filtered by bases, of shape (number of nodes, level, channels).
        '''
        x = F.dropout(x, self.dropout, training=self.training)
        x = x * self.comb_weight
        x = torch.sum(x, dim=1)
        return x

class GFK(nn.Module):
    def __init__(self, level, nfeat, nlayers, nhidden, nclass, dropoutC, dropoutM, bias, sole=True):
        super(GFK, self).__init__()
        self.nfeat = nfeat
        self.level = level+1
        self.comb = Combination(nfeat, self.level, dropoutC, sole)
        self.mlp = MLP(nfeat, nlayers,nhidden, nclass, dropoutM, bias)


    def forward(self, x):
        x = self.comb(x)                      
        x = self.mlp(x)
        return x
       
