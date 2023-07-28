from collections import defaultdict as ddict

def process(dataset):
    """
    pre-process dataset
    :param dataset: a dictionary containing 'train', 'valid' and 'test' data.
    :param num_rel: relation number
    :return:
    """
    sr2o = ddict(set)
    sr2o_neg = ddict(set)
    for subj, rel, obj in dataset['train']:
        sr2o[(subj, rel)].add(obj)
    sr2o_train = {k: list(v) for k, v in sr2o.items()}

    for subj, rel, obj in dataset['neg']:
        sr2o_neg[(subj, rel)].add(obj)
    sr2o_negative = {k: list(v) for k, v in sr2o_neg.items()}

    for split in ['valid', 'test']:
        for subj, rel, obj in dataset[split]:
            sr2o[(subj, rel)].add(obj)

    sr2o_all = {k: list(v) for k, v in sr2o.items()}
    triplets = ddict(list)

    for subj, rel, obj in dataset['train']:
        triplets['train'].append({'triple': (subj, rel, obj), 'label': sr2o_train[(subj, rel)], 'neg_label': sr2o_negative.get((subj,rel))})

    for split in ['valid', 'test']:
        for i, (subj, rel, obj) in enumerate(dataset[split]):
            triplets[f"{split}_tail"].append({'triple': (subj, rel, obj), 'label': sr2o_all[(subj, rel)]})
            if i > len(dataset['train']):
                break
    triplets = dict(triplets)

    return triplets



