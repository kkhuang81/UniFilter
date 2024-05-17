import torch
import math
import numpy as np
import random
from dataset import load_nc_dataset
from collections import OrderedDict
import csv
import os
import sys
import time
from time import perf_counter

import scipy.sparse as sp
import torch
import pickle as pkl
import struct
import gc
import scipy.special as ss
from scipy.sparse import csr_matrix, coo_matrix

from torch_geometric.utils import remove_self_loops, add_self_loops, to_undirected, is_undirected,dropout_adj

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def random_splits(labels, num_classes, percls_trn=20, val_lb=500, seed=12591):   
    
    num_nodes=labels.shape[0]
    index=[i for i in range(0,num_nodes)]
    train_idx=[]
    rnd_state = np.random.RandomState(seed)
    for c in range(num_classes):
        class_idx = np.where(labels == c)[0]
        if len(class_idx)<percls_trn:
            train_idx.extend(class_idx)
        else:
            train_idx.extend(rnd_state.choice(class_idx, percls_trn,replace=False))
    train_idx=np.array(train_idx)              
    rest_index = [i for i in index if i not in train_idx]
    val_idx=np.array(rnd_state.choice(rest_index,val_lb,replace=False))
    test_idx=np.array([i for i in rest_index if i not in val_idx])    

    return train_idx, val_idx, test_idx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def PropMatrix(adj):    
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    del row_sum
    gc.collect()    
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.

    t=time.time()
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)    
    adj=d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt) # optimize this step
    print('matrix multiplication time: ', time.time()-t) 
    return adj

def edgeindex_construct(edge_index, num_nodes, self_loop=True):     
    edge_index=torch.LongTensor(edge_index)       
    if not is_undirected(edge_index):
        edge_index = to_undirected(edge_index)           
    if self_loop:
        print('add self loop: ', edge_index.shape)
        edge_index=remove_self_loops(edge_index)[0]
        edge_index=add_self_loops(edge_index)[0]
        print('after add self loop: ', edge_index.shape)        
    edge_index=edge_index.numpy()
    
    num_edges=edge_index[0].shape[0]
    data=np.array([1]*num_edges)
    adj=sp.coo_matrix((data, (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes)).tocsr()
    
    t = time.time()
    adj=PropMatrix(adj)
    Propagate_matrix_time = time.time()-t
    print('propagate matrix time: ', Propagate_matrix_time)
    

    t=time.time()
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    sparse_mx_time = time.time()-t
    print('sparse_mx: ', sparse_mx_time)

    return adj, Propagate_matrix_time, sparse_mx_time

def load_dataset(LP, feat, K=6, tau = 1.0, homo_ratio=0.6, plain=False):          
    
    num_nodes, dim=feat.shape
    
    cosval = math.cos(math.pi*(1.0-homo_ratio)/2.0)
    print('cosval: ', cosval) 


    if not plain:
        print('Adaptive Basis')
        t1 = time.time()                
        norm = torch.norm(feat, dim=0)
        norm = torch.clamp(norm, 1e-8)
        last = feat/norm
        second = torch.zeros_like(last)
        basis_sum = torch.zeros_like(last)
        HM = torch.zeros_like(last)
        HM += feat
        basis_sum +=  last
        # features = [last]
        features = [feat]
        for k in range(1, K+1):
            V_k = torch.spmm(LP, last)
            HM = torch.spmm(LP, HM)   
            project_1 = torch.einsum('nd,nd->d', V_k, last)
            project_2 = torch.einsum('nd,nd->d', V_k, second)
            V_k -= (project_1 * last + project_2 * second)
            norm = torch.norm(V_k,dim = 0)
            norm = torch.clamp(norm, 1e-8)
            V_k /= norm 
            H_k = basis_sum / k
            Tf = torch.sqrt(torch.square(torch.einsum('nd,nd->d', H_k, features[-1])/cosval) - ((k-1)*cosval+1)/k)
            torch.nan_to_num_(Tf, nan=0.0)
            H_k += torch.mul(Tf, V_k)
            norm = torch.norm(H_k,dim = 0)
            norm=torch.clamp(norm, 1e-8)
            H_k /= norm   
            norm = torch.norm(HM, dim = 0)
            norm = torch.clamp(norm, 1e-8)        
            features.append(HM*tau + H_k*(1.0-tau))   
            basis_sum += H_k
            second = last 
            last = V_k      
        #features=basis
        features_time = time.time()-t1
        print('feat diffusion time plus: ', features_time)
        del last, second, LP
        gc.collect()
        features = torch.cat(features, 1)   
        print(features.shape)        
    else:
        print('Non-orthogonalization')  
        t1 = time.time()            
        features=[feat]
        basis=feat    
        for i in range(1,K+1):
            basis=torch.spmm(LP, basis)   
            features.append(basis)
        features_time = time.time()-t1
        print('feat diffusion time: ', features_time)      
        del basis, LP
        gc.collect()
        features = torch.cat(features,1)
        print(features.shape)     
    return features, dim
