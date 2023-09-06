from torch.utils.data import Dataset
import numpy as np
import torch


class TrainDataset(Dataset):
    def __init__(self, triplets, num_ent, params):
        super(TrainDataset, self).__init__()
        self.p = params
        self.triplets = triplets
        self.label_smooth = params.lbl_smooth
        self.num_ent = num_ent


    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, item):
        ele = self.triplets[item]

        triple, label, neg_label = torch.tensor(ele['triple'], dtype=torch.long), np.int32(ele['label']), ele['neg_label']

        n_label = np.zeros([self.num_ent], dtype=np.float32)
        if neg_label is not None:
            neg=torch.tensor(1, dtype=torch.float32)
            n_label = self.get_label(neg_label)
        else:
            neg=torch.tensor(0, dtype=torch.float32)
            n_label = torch.tensor(n_label, dtype=torch.float32)

        label = self.get_label(label)

        if self.label_smooth != 0.0:
            label = (1.0 - self.label_smooth) * label + (1.0 / self.num_ent)

        return triple, label, neg, n_label

    def get_label(self, label):
        """
        get label corresponding to a (sub, rel) pair
        :param label: a list containing indices of objects corresponding to a (sub, rel) pair
        :return: a tensor of shape [nun_ent]
        """
        y = np.zeros([self.num_ent], dtype=np.float32)
        y[label] = 1
        return torch.tensor(y, dtype=torch.float32)

class TestDataset(Dataset):
    def __init__(self, triplets, num_ent, params):
        super(TestDataset, self).__init__()
        self.triplets = triplets
        self.num_ent = num_ent


    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, item):
        ele = self.triplets[item]
        triple, label = torch.tensor(ele['triple'], dtype=torch.long), np.int32(ele['label'])

        label = self.get_label(label)
        return triple, label,  triple[2], label

    def get_label(self, label):
        """
        get label corresponding to a (sub, rel) pair
        :param label: a list containing indices of objects corresponding to a (sub, rel) pair
        :return: a tensor of shape [nun_ent]
        """
        y = np.zeros([self.num_ent], dtype=np.float32)
        y[label] = 1
        return torch.tensor(y, dtype=torch.float32)


