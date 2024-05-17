## How Universal Polynomial Bases Enhance Spectral Graph Neural Networks: Heterophily, Over-smoothing, and Over-squashing

**UniFilter** is a polynomial graph filter by using a novel universal polynomial basis called **UniBasis**. This repository contains the source codes of UniFilter, data process, and split generation codes. For detailed information about NIGCN, please refer to our paper in ICML 2024. If any issues are observed, please contact Keke Huang, thanks!

## Environment Settings    

- pytorch 1.7.0
- torch-geometric 1.6.1
- scipy 1.9.3
- seaborn 0.12.0
- scikit-learn 1.1.3
- ogb 1.3.1
- gdown

## Datasets

Please acquire all the data from ChebNet II and put the data in the subfolder './data'. 
The ogb datasets (ogbn-arxiv and ogbn-papers100M) and non-homophilous datasets (from [LINKX](https://arxiv.org/abs/2110.14446) ) can be downloaded automatically.

