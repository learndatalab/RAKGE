#!/bin/sh

cd datasets/
unzip credit.zip
unzip spotify.zip
unzip FB15k-237.zip
cd ../RAKGE/
python preprocess_kg_num_lit.py --dataset credit
python preprocess_kg_num_lit.py --dataset spotify
python preprocess_kg_num_lit.py --dataset FB15k-237

