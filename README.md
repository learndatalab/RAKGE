# RAKGE
This repository provides PyTorch implementations of **RAKGE** as described in the paper: **Exploiting Relation-aware Attribute Representation Learning in Knowledge Graph Embedding for Numerical Reasoning** (KDD 2023).


<img width="602" alt="RAKGE" src="https://github.com/learndatalab/RAKGE/assets/116534675/a94d2b69-4c32-48be-aff9-d75c25f5073e">

## Citing
If you want to mention RAKGE for your research, please consider citing the following paper:

    @inproceedings{RAKGE,
    author = {Kim, Gayeong and Kim, Sookyung and Kim, Ko Keun and Park, Suchan and Jung, Heesoo and Park, Hogun},
    title = {Exploiting Relation-aware Attribute Representation Learning in Knowledge Graph Embedding for Numerical Reasoning},
    booktitle = {Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
    year = {2023}
    }



## Experiment Environment
- python 3.7+
- torch 1.9+
- dgl 0.7+ 


## Basic Usage

### Preprocess the datasets
To preprocess their numeric attributes, please execute the following command:

    chmod +x ./preprocess.sh && ./preprocess.sh

### Reproduce the results
Now you are ready to train and evaluate RAKGE and other baselines. To reproduce the results provided in the paper, please execute the corresponding command for each model as follows:

#### RAKGE
    python run.py --gpu 0 --n_layer 0  --literal --init_dim 200 --att_dim 200 --head_num 5 --name RAKGE --scale 0.25 --order 0.25 --data {credit, spotify} --drop 0.7 

#### TransE
    python run.py --gpu 0 --n_layer 0 --init_dim 200 --name lte --score_func transe --opn mult --x_ops "d" --hid_drop 0.7  --data {credit,spotify}
    
#### LiteralE
    python run.py --gpu 0 --n_layer 0 --literal --init_dim 200 --name TransELiteral_gate --data {credit, spotify} --input_drop 0.7 

#### R-GCN
    python run.py --gpu 0 --n_layer 1 --score_func transe --opn mult --gcn_dim 150 --init_dim 150 --num_base 5 --encoder rgcn --name repro --data {credit, spotify} --hid_drop 0.7







## Miscellaneous
Please send any questions you might have about the code and/or the algorithm to gayeongkim@o365.skku.edu



## Acknowledgement
We refer to the code of [LTE-KGC](https://github.com/MIRALab-USTC/GCN4KGC) and [LiteralE](https://github.com/SmartDataAnalytics/LiteralE). Thanks for their contributions.
