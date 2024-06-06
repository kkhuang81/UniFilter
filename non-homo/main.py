import argparse
import sys
import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import seaborn as sns
from torch_geometric.utils import to_undirected
from data_utils import eval_acc, eval_rocauc, load_fixed_splits
from utils import load_dataset, edgeindex_construct
from models import *
import uuid
import torch.optim as optim

import optuna

@torch.no_grad()
def evaluate(model, feature, label, index, eval_func, criterion):
    model.eval()
    out = model(feature[index])
    acc = eval_func(label[index], out)
    return acc

def parse_args():
    parser = argparse.ArgumentParser(description='General Training Pipeline')
    parser.add_argument('--seed',type=int, default=51290)
    parser.add_argument('--dev',type=int, default=0)
    parser.add_argument('--dataset', type=str, default='genius')

    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--hid', type=int, default=128)
    parser.add_argument('--nlayers', type=int, default=3, help='number of layers for MLP')

    parser.add_argument('--model', type=str, choices=['mlp', 'gfk'], default='gfk')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--runs', type=int, default=5, help='number of distinct runs')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--dprate', type=float, default=0.5)

    parser.add_argument('--plain', action="store_true", help='if plain basis')

    parser.add_argument('--patience', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--lr1', type=float, default=0.01, help='Initial learning rate of MLP.')
    parser.add_argument('--lr2', type=float, default=0.01, help='Initial learning rate of Combination.')
    parser.add_argument('--wd1', type=float, default=5e-4, help='Weight decay of MLP.')
    parser.add_argument('--wd2', type=float, default=5e-4, help='Weight decay of Combination.')
    parser.add_argument('--sole', action="store_true", help='if one paramter for one level feature')
    parser.add_argument('--dpC', type=float, default=0.5, help='Dropout rate of Combination.')
    parser.add_argument('--dpM', type=float, default=0.5, help='Dropout rate of MLP.')
    parser.add_argument('--tau', type=float, default=0.5, help='tau.')
    parser.add_argument('--to_undirected', action="store_true", help='if to_undirected')
    parser.add_argument('--self_loop', action="store_true", default=False, help='if self_loop')
    parser.add_argument('--bias', default='none', help='bias.')
    parser.add_argument('--log_dir', type=str,  default='./report')

    # Optuna Settings
    parser.add_argument('--optruns', type=int, default=100)
    parser.add_argument('--path', type=str, default="")
    parser.add_argument('--name', type=str, default="opt")

    # Train settings
    args = parser.parse_args()
    print(args)
    print("---------------------------------------------")
    return args

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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

if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)
    device = f'cuda:{args.dev}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    #load fixed dataset split
    split_idx_lst = load_fixed_splits(args.dataset)
    dataset_str = 'data/' + args.dataset +'/'+args.dataset+'.npz'
    data = np.load(dataset_str)
    edge_index, feat, labels=data['edge_index'], data['feats'], data['labels'] 
    num_nodes = labels.shape[0]
    graph = GraphConstruct(edge_index, num_nodes) 

    label = torch.LongTensor(labels)
    if len(label.shape) == 1:
        label = label.unsqueeze(1)
    label = label.to(device)   
    

    LP, _,_ = edgeindex_construct(edge_index, num_nodes, args.self_loop)    
    feat=torch.FloatTensor(feat)

    if args.dataset == 'genius':
        criterion = nn.BCEWithLogitsLoss()
        eval_func = eval_rocauc
    else:
        criterion = nn.NLLLoss()
        eval_func = eval_acc

    ### Training loop ###
    results = []

    checkpt_file = 'pretrained/'+uuid.uuid4().hex+'.pt'

    for run in range(args.runs):
        split_idx = split_idx_lst[run]
        train_idx = split_idx['train']
        homoratio = homocal(graph, train_idx, labels)
        print(run, '-homoration: ', homoratio)     
        features, dim = load_dataset(LP, feat, args.K, args.tau, homoratio, args.plain)
        train_idx=train_idx.to(device)
        # Model and optimizer
        if args.model =='mlp':
            model = MLP(nfeat=features.shape[1],
                nlayers=args.nlayers,
                nhidden=args.hid,
                nclass=label.max().item() + 1,
                dropout=args.dropout,
                bias = args.bias).to(device)
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        elif args.model =='gfk':    
            model = GFK(level=args.K,
                nfeat=dim,
                nlayers=args.nlayers,
                nhidden=args.hid,
                nclass=label.max().item() + 1,
                dropoutC=args.dpC,
                dropoutM=args.dpM,
                bias = args.bias,
                sole = args.sole).to(device)            

            optimizer = optim.AdamW([{
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
        
        features = features.to(device)
        
        model.train()

        best_val_acc = best_test_acc = 0
        best_val_loss = float('inf')
        val_loss_history = []
        val_acc_history = []
        bad_counter = 0
        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()
            out = model(features[train_idx])

            if args.dataset =='genius':
                if label.shape[1] == 1:
                    true_label = F.one_hot(label, label.max() + 1).squeeze(1)
                else:
                    true_label = label
                loss = criterion(out, true_label.squeeze(1)[train_idx].to(torch.float))
            else:
                out = F.log_softmax(out, dim=1)
                loss = criterion(out, label.squeeze(1)[train_idx]) 
            loss.backward()
            optimizer.step()
            val_acc = evaluate(model, features, label, split_idx['valid'], eval_func, criterion)
            if epoch % 100 == 0:
                print(f'Epoch: {epoch:02d}, 'f'Loss: {loss:.4f}, 'f'Valid: {100 * val_acc:.2f}%, ')
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), checkpt_file)
        model.load_state_dict(torch.load(checkpt_file))
        test_acc = evaluate(model, features, label, split_idx['test'], eval_func, criterion)
        print(f'best_val_acc:{100*best_val_acc:.2f}%, test acc {100*test_acc:.2f}')
        results.append([test_acc,best_val_acc])
        os.remove(checkpt_file)
    test_acc_mean, val_acc_mean= np.mean(results, axis=0) * 100
    test_acc_std = np.sqrt(np.var(results, axis=0)[0]) * 100
    values=np.asarray(results)[:,0]
    uncertainty=np.max(np.abs(sns.utils.ci(sns.algorithms.bootstrap(values,func=np.mean,n_boot=1000),95)-values.mean()))
    print(f'Dataset {args.dataset}, in {args.runs} repeated experiment:')
    print(f'test acc mean = {test_acc_mean:.4f} ± {uncertainty*100:.4f}  \t val acc mean = {val_acc_mean:.4f}')

   
