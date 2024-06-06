from __future__ import division
from __future__ import print_function
from utils import load_dataset, data_split, muticlass_f1, edgeindex_construct
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from models import *
import uuid
import os

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="cora",help='Dataset to use.')
parser.add_argument('--seed', type=int, default=51290, help='Random seed.')
parser.add_argument('--type', type=int, default=0, help='the type of the split')

parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--patience', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')

parser.add_argument('--model', type=str, choices=['mlp', 'gfk'], default='gfk')
parser.add_argument('--lr1', type=float, default=0.01, help='Initial learning rate of MLP.')
parser.add_argument('--lr2', type=float, default=0.01, help='Initial learning rate of Combination.')
parser.add_argument('--wd1', type=float, default=5e-4, help='Weight decay of MLP.')
parser.add_argument('--wd2', type=float, default=5e-4, help='Weight decay of Combination.')
parser.add_argument('--sole', action="store_true", help='if one paramter for one level feature')
parser.add_argument('--dpC', type=float, default=0.5, help='Dropout rate of Combination.')
parser.add_argument('--dpM', type=float, default=0.5, help='Dropout rate of MLP.')

parser.add_argument('--plain', action="store_true", help='if plain basis')

parser.add_argument('--dev', type=int, default=1, help='device id')
parser.add_argument('--hid', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--nlayers', type=int, default=2, help='Number of hidden layers.')
parser.add_argument('--bias', default='none', help='bias.')
parser.add_argument('--batch', type=int, default=64, help='batch size')
parser.add_argument('--K', type=int, default=10, help='the maximum level')
parser.add_argument('--tau', type=float, default=0.5, help='homo/heterophily trade-off')


parser.add_argument('--train_rate', type=float, default=0.60, help='train set rate.')
parser.add_argument('--val_rate', type=float, default=0.20, help='val set rate.')


args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

print("--------------------------")
print(args)

def train():
    model.train()
    time_epoch = 0

    t1 = time.time()
    optimizer.zero_grad()
    output = model(features[train_idx])
    loss_train = loss_fn(output, labels[train_idx])
    loss_train.backward()
    optimizer.step()
    return loss_train, time.time()-t1

def validate():
    model.eval()
    with torch.no_grad():
        output = model(features[val_idx])
        micro_val = muticlass_f1(output, labels[val_idx])
        return micro_val.item()

def test():
    model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    with torch.no_grad():
        output = model(features[test_idx])
        micro_test = muticlass_f1(output, labels[test_idx])
        return micro_test.item()

def GraphConstruct(edge_index, n):
    graph = []
    for i in range(n):
        edge = []
        graph.append(edge)
    m = edge_index.shape[1]
    for i in range(m):
        u,v=edge_index[0][i], edge_index[1][i]
        graph[u].append(v)
    return graph

def homocal(graph, train_idx, labels):
    n = labels.shape[0]
    train = np.array([False]*n)
    train[train_idx]=True
    edge = 0.0
    cnt = 0.0
    for node in train_idx:
        for nei in graph[node]:
            if train[nei]:
                edge += 1.0
                if labels[node]==labels[nei]:
                    cnt += 1.0
    return cnt/edge

SEEDS=[1941488137,25190,983997847,12591,4019585660,2108550661,1648766618,329014539,3212139042,2424918363]
training_time=[]
test_f1score=[]


dataset_str = 'data/' + args.dataset +'/'+args.dataset+'.npz'
data = np.load(dataset_str)
edge_index, feat, label=data['edge_index'], data['feats'], data['labels'] 
num_nodes = label.shape[0]
graph = GraphConstruct(edge_index, num_nodes) 

labels = torch.LongTensor(label) 


LP, _,_ = edgeindex_construct(edge_index, num_nodes)    
feat=torch.FloatTensor(feat)

run = 10

for idx in range(run):           
    train_idx, val_idx, test_idx = data_split(label, args.train_rate, args.val_rate, SEEDS[idx%10])
    homoratio = homocal(graph, train_idx, label)
    print(idx, '-homoration: ', homoratio)     
    features, dim = load_dataset(LP, feat, args.K, args.tau, homoratio, args.plain)
    checkpt_file = 'pretrained/'+uuid.uuid4().hex+'.pt'

    # Model and optimizer
    if args.model =='mlp':
        model = MLP(nfeat=features.shape[1],
            nlayers=args.nlayers,
            nhidden=args.hid,
            nclass=labels.max().item() + 1,
            dropout=args.dropout,
            bias = args.bias).to(args.dev)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.model =='gfk':        
        model = GFK(level=args.K,
            nfeat=dim,
            nlayers=args.nlayers,
            nhidden=args.hid,
            nclass=labels.max().item() + 1,
            dropoutC=args.dpC,
            dropoutM=args.dpM,
            bias = args.bias,
            sole = args.sole).to(args.dev)
        optimizer = optim.Adam([{
            'params': model.mlp.parameters(),
            'weight_decay': args.wd1,
            'lr': args.lr1
        }, {
            'params':model.comb.parameters(),
            'weight_decay': args.wd2,
            'lr': args.lr2
        }])
        features=features.view(-1, args.K+1, dim)
    else:
        raise ValueError('wrong model para')

    loss_fn = nn.CrossEntropyLoss()

        
    features = features.cuda(args.dev)
    labels = labels.cuda(args.dev)

    train_time = 0
    bad_counter = 0
    best = 0
    best_epoch = 0
    
    for epoch in range(args.epochs):
        loss_tra,train_ep = train()
        f1_val = validate()
        train_time+=train_ep
        if(epoch+1)%100 == 0:
            print('Epoch:{:04d}'.format(epoch+1),
                'train',
                'loss:{:.3f}'.format(loss_tra),
                '| val',
                'acc:{:.3f}'.format(f1_val),
                '| cost:{:.3f}'.format(train_time))
        if f1_val > best:
            best = f1_val
            best_epoch = epoch
            torch.save(model.state_dict(), checkpt_file)
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break

    if args.model == 'gfk':
        theta=model.comb.comb_weight.clone()
        theta=theta.detach().cpu().numpy().reshape(-1)    
        print('Theta:', [float('{:.4f}'.format(i)) for i in theta])
    f1_test = test()
    print("Train cost: {:.4f}s".format(train_time))
    print('Load {}th epoch'.format(best_epoch))
    print("Test f1:{:.3f}".format(f1_test))
    print("Current epoch: ", epoch)
    print("--------------------------")

    training_time.append(train_time)
    test_f1score.append(f1_test)
    os.remove(checkpt_file)

print("avg_train_time: {:.4f} s".format(np.mean(training_time)))
print("std_train_time: {:.4f} s".format(np.std(training_time)))
print("avg_f1_score: {:.4f}".format(np.mean(test_f1score)))
print("std_f1_score: {:.4f}".format(np.std(test_f1score)))
