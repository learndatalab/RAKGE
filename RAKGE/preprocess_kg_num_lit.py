import numpy as np
import argparse
import pickle
import pandas as pd

parser = argparse.ArgumentParser(description='Create literals')
parser.add_argument('--dataset', default='credit', metavar='',
                    help='which dataset in {`credit`, `spotify`, `FB15k-237`} to be used?')
args = parser.parse_args()

train=pd.read_csv(f'../datasets/{args.dataset}/train.txt', sep="\t",header=None,names=['head','relation','tail'])
valid=pd.read_csv(f'../datasets/{args.dataset}/valid.txt', sep="\t",header=None,names=['head','relation','tail'])
test=pd.read_csv(f'../datasets/{args.dataset}/test.txt', sep="\t",header=None,names=['head','relation','tail'])
neg=pd.read_csv(f'../datasets/{args.dataset}/neg.txt', sep="\t",header=None,names=['head','relation','tail'])

all_df=pd.concat([train, valid, test], ignore_index=True)

print("# of Triplets", len(all_df))
print('# of Triplets (train)', len(train))
print('# of Triplets (valid)', len(valid))
print('# of Triplets (test)', len(test))

# Entity dictionary
ent=set(all_df['head'].unique()) | set(all_df['tail'].unique())
entities = {str(e) for e in ent}
entity_dict = {str(v): k for k, v in enumerate(entities)}

print("# of Entites:", len(entity_dict))

with open(f'../datasets/{args.dataset}/entities.dict','wb') as fw:
    pickle.dump(entity_dict, fw)

# Relation dictionary
relations=all_df['relation'].unique()
relation_dict = {v: k for k, v in enumerate(relations)}


print("# of Entity Relations:",len(relation_dict))
with open(f'../datasets/{args.dataset}/relations.dict','wb') as fw:
    pickle.dump(relation_dict, fw)

# Load raw literals
df = pd.read_csv(f'../datasets/{args.dataset}/literals/numerical_literals.txt', header=None, sep='\t')

numrel_dict = {v: k for k, v in enumerate(df[1].unique())}

print("# of Attribute Triples: ", len(df))

# Resulting file
num_lit = np.zeros([len(entity_dict), len(numrel_dict)], dtype=np.float32)

for i, (s, p, lit) in enumerate(df.values):
    try:
        if "id" in p:
            num_lit[entity_dict[str(s).lower()], numrel_dict[p]] = 1.0
        else:
            num_lit[entity_dict[str(s).lower()], numrel_dict[p]] = lit

    except KeyError:
        continue

np.save(f'../datasets/{args.dataset}/literals/numerical_literals.npy', num_lit)


M = train.shape[0]
X = np.zeros([M, 3], dtype=int)
for i, row in train.iterrows():
    X[i, 0] = entity_dict[str(row[0])]
    X[i, 1] = relation_dict[row[1]]
    X[i, 2] = entity_dict[str(row[2])]
# X_train = X.astype(np.int32)
# train_npy = os.path.join(f'../datasets/{args.dataset}'train/.npy')
np.save(f'../datasets/{args.dataset}/train.npy', X.astype(np.int32))
# np.save(train_npy, X_train)



M = neg.shape[0]
X_neg = np.zeros([M, 3], dtype=int)
for i, row in neg.iterrows():
    X_neg[i, 0] = entity_dict[str(row[0])]
    X_neg[i, 1] = relation_dict[row[1]]
    X_neg[i, 2] = entity_dict[str(row[2])]
X_neg = X_neg.astype(np.int32)
self.negative = np.asarray(_read_triplets_as_list(
    neg_path, entity_dict, relation_dict))
neg_npy = os.path.join(self.dir, 'neg.npy')
np.save(neg_npy, X_neg)



