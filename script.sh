python3 training.py --dataset cora --epochs 1000 --seed 51290 --hid 256 --nlayers 3 --K 10 --patience 200 --lr1 0.15 --lr2 0.005 --wd1 0.0005 --wd2 0 --dpC 0.3 --dpM 0.4 --tau 1.0  

python3 training.py --dataset citeseer --epochs 1000 --seed 51290 --hid 128 --nlayers 4 --K 10  --patience 200 --lr1 0.01 --lr2 0.05 --wd1 5e-5 --wd2 0.0001 --dpC 0.9 --dpM 0.8 --tau 0.9  

python3 training.py --dataset pubmed --epochs 1000 --seed 51290 --hid 128 --nlayers 4 --K 10 --patience 200 --lr1 0.1 --lr2 0.05 --wd1 0.0005 --wd2 0.0001 --dpC 0.4 --dpM 0 --tau 0.8 


python3 training.py --dataset actor --epochs 1000 --seed 51290 --hid 256 --nlayers 4 --K 10 --patience 200 --lr1 0.05 --lr2 0.005 --wd1 0.0005 --wd2 0.0005 --dpC 0.2 --dpM 0  --tau 0.1 

python3 training.py --dataset squirrel --epochs 1000 --seed 51290 --hid 256 --nlayers 3 --K 10 --patience 200 --lr1 0.01 --lr2 0.05 --wd1 0.5e-5 --wd2 0.0005 --dpC 0 --dpM 0.6 --bias 'bn' --tau 0.7 

python3 training.py --dataset chameleon --epochs 1000 --seed 51290 --hid 256 --nlayers 4 --K 10 --patience 200 --lr1 0.01 --lr2 0.005 --wd1 0 --wd2 0.0005 --dpC 0 --dpM 0.7  --tau 0.7 

python3 training.py --dataset ogbnarxiv --K 10 --dpC 0 --dpM 0.1 --lr1 0.01 --lr2 0.005 --hid 2048 --wd1 0.0005 --wd2 0.0005 --nlayers 2 --model gfk --patience 300 --tau 0.5



 
