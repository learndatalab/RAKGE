import os
import argparse
import time
import logging
from pprint import pprint
import numpy as np
import random
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import dgl
from data.knowledge_graph import load_data

from model import GCN_TransE, GCN_DistMult, GCN_ConvE
from model.kge_models import TransE, DistMult, ConvE, HAKE
from model.rakge_model import RAKGE
from model.literal_models import TransELiteral_gate, KBLN
from utils import process, TrainDataset, TestDataset




class Runner(object):
    def __init__(self, params):
        self.p = params
        self.prj_path = Path(__file__).parent.resolve()
        self.data = load_data(self.p.dataset)
        self.num_ent, self.train_data, self.valid_data, self.test_data, self.num_rels, self.entity_dict, self.relation_dict, self.negative_data= self.data.num_nodes, self.data.train, self.data.valid, self.data.test, self.data.num_rels, self.data.entity_dict, self.data.relation_dict, self.data.negative
        self.triplets = process({'data': self.data, 'train': self.train_data, 'valid': self.valid_data, 'test': self.test_data,'neg': self.negative_data})
        self.p.embed_dim = self.p.k_w * self.p.k_h if self.p.embed_dim is None else self.p.embed_dim  # output dim of gnn
        self.data_iter = self.get_data_iter()
        if self.p.gpu >= 0:
            self.g = self.build_graph().to("cuda")
        else:
            self.g = self.build_graph()
        self.edge_type, self.edge_norm = self.get_edge_dir_and_norm()
        self.model = self.get_model()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.p.lr, weight_decay=self.p.l2)
        self.best_val_mrr, self.best_epoch, self.best_val_results = 0., 0., {}
        os.makedirs('./logs', exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.prj_path / 'logs' / self.p.name),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        pprint(vars(self.p))

    def fit(self):
        save_root = self.prj_path / 'checkpoints'

        if not save_root.exists():
            save_root.mkdir()
        save_path = save_root / (self.p.name+ str(self.p.gpu) + '.pt')

        if self.p.restore:
            self.load_model(save_path)
            self.logger.info('Successfully Loaded previous model')

        tolerance = 0
        for epoch in range(1, self.p.max_epochs+1):
            start_time = time.time()
            train_loss = self.train()

            self.logger.info(
                f"[Epoch {epoch}]: Training Loss: {train_loss:.5}, Cost: {time.time() - start_time:.2f}s")

            if epoch % 10 == 0:
                val_results = self.evaluate('valid')
                if val_results['mrr'] > self.best_val_mrr:
                    tolerance = 0
                    self.best_val_results = val_results
                    self.best_val_mrr = val_results['mrr']
                    self.best_epoch = epoch
                    self.save_model(save_path)
                else:
                    if tolerance < self.p.tolerance:
                        tolerance += 10

                        if tolerance % 25 == 0:
                            self.load_model(save_path)
                            test_results = self.evaluate('test')
                            self.logger.info(
                                f"MRR: Avg {test_results['mrr']:.5}")
                            self.logger.info(
                                f"MR:  Avg {test_results['mr']:.5}")
                            self.logger.info(
                                f"hits_left@1 = {test_results['hits@1']}")
                            self.logger.info(
                                f"hits_left@3 = {test_results['hits@3']}")
                            self.logger.info(
                                f"hits_left@10 = {test_results['hits@10']}")
                    else:
                        break


                self.logger.info(
                    f"Valid MRR: {val_results['mrr']:.5}, Best Valid MRR: {self.best_val_mrr:.5}")

        self.logger.info(vars(self.p))
        self.load_model(save_path)
        self.logger.info(
            f'Loading best model in {self.best_epoch} epoch, Evaluating on Test data')
        start = time.time()
        test_results = self.evaluate('test')
        end = time.time()
        self.logger.info(f"MRR:  {test_results['mrr']:.5}")
        self.logger.info(f"MR:  {test_results['mr']:.5}")
        self.logger.info(f"hits@1 = {test_results['hits@1']}")
        self.logger.info(f"hits@3 = {test_results['hits@3']}")
        self.logger.info(f"hits@10 = {test_results['hits@10']}")
        self.logger.info("time ={}".format(end-start))

    def train(self):
        self.model.train()
        losses = []
        train_iter = self.data_iter['train']
        for step, (triplets, labels, neg, n_label) in enumerate(train_iter):
            if self.p.gpu >= 0:
                triplets, labels, neg, n_label = triplets.to("cuda"), labels.to("cuda"), neg.to("cuda"), n_label.to("cuda")
            subj, rel, obj  = triplets[:, 0], triplets[:, 1], triplets[:, 2]




            # elif self.p.encoder == 'rgcn':
            if self.p.n_layer > 0 :
                pred = self.model(self.g, subj, rel)
                loss = self.model.calc_loss(pred, labels)
            elif self.p.literal:
                if self.p.name == 'RAKGE':
                    loss = self.model(self.g, subj, rel, labels, neg, n_label)
            else:
                loss = self.model(self.g, subj, rel, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

        loss = np.mean(losses)
        return loss


    def evaluate(self, split):
        """
        Function to evaluate the model on validation or test set
        :param split: valid or test, set which data-set to evaluate on
        :return: results['mr']: Average of ranks_left and ranks_right
                 results['mrr']: Mean Reciprocal Rank
                 results['hits@k']: Probability of getting the correct prediction in top-k ranks based on predicted score
        """

        def get_combined_results(left):
            results = dict()
            count_left = float(left['count'])
            results['mr'] = round((left['mr']) / (count_left), 5)
            results['mrr'] = round((left['mrr']) / (count_left), 5)
            for k in [1, 3, 10]:
                results[f'hits@{k}'] = round(left[f'hits@{k}'] / count_left, 5)
            return results

        self.model.eval()
        left_result = self.predict(split, 'tail')
        res = get_combined_results(left_result)
        return res

    def predict(self, split='valid', mode='tail'):
        """
        Function to run model evaluation for a given mode
        :param split: valid or test, set which data-set to evaluate on
        :param mode: head or tail
        :return: results['mr']: Sum of ranks
                 results['mrr']: Sum of Reciprocal Rank
                 results['hits@k']: counts of getting the correct prediction in top-k ranks based on predicted score
                 results['count']: number of total predictions
        """
        with torch.no_grad():
            results = dict()
            test_iter = self.data_iter[f'{split}_{mode}']
            for step, (triplets, labels, neg, n_label) in enumerate(test_iter):
                triplets, labels, neg, n_label = triplets.to("cuda"), labels.to("cuda"), neg.to("cuda"), n_label.to("cuda")
                subj, rel, obj = triplets[:, 0], triplets[:, 1], triplets[:, 2]


                if self.p.n_layer > 0 :
                    pred = self.model(self.g, subj, rel)

                elif self.p.literal:
                    if self.p.name == 'RAKGE':
                        pred = self.model(self.g, subj, rel, labels, neg, n_label)

                    '''elif self.p.name == 'KBLN':
                        pred = self.model(self.g, subj, rel, labels)
                        

                    elif 'Literal' in self.p.name or 'lte' in self.p.name or 'KBLN' in self.p.name:
                        pred = self.model(self.g, subj, rel, labels)

                else:
                    pred = self.model.forward(self.g, subj, rel, labels)'''
                else:
                    pred = self.model(self.g, subj, rel, labels)

                b_range = torch.arange(pred.shape[0], device="cuda")
                # [batch_size, 1], get the predictive score of obj

                target_pred = pred[b_range, obj]
                # label=>-1000000, not label=>pred, filter out other objects with same sub&rel pair

                pred = torch.where(
                    labels.bool(), -torch.ones_like(pred) * 10000000, pred)
                # copy predictive score of obj to new pred
                pred[b_range, obj] = target_pred
                ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[
                    b_range, obj]  # get the rank of each (sub, rel, obj)

                ranks = ranks.float()
                results['count'] = torch.numel(
                    ranks) + results.get('count', 0)  # number of predictions

                results['mr'] = torch.sum(ranks).item() + results.get('mr', 0)
                results['mrr'] = torch.sum(
                    1.0 / ranks).item() + results.get('mrr', 0)

                for k in [1, 3, 10]:
                    results[f'hits@{k}'] = torch.numel(
                        ranks[ranks <= k]) + results.get(f'hits@{k}', 0)

        return results

    def save_model(self, path):
        """
        Function to save a model. It saves the model parameters, best validation scores,
        best epoch corresponding to best validation, state of the optimizer and all arguments for the run.
        :param path: path where the model is saved
        :return:
        """
        state = {
            'model': self.model.state_dict(),
            'best_val': self.best_val_results,
            'best_epoch': self.best_epoch,
            'optimizer': self.optimizer.state_dict(),
            'args': vars(self.p)
        }

        torch.save(state, path)

    def load_model(self, path):
        """
        Function to load a saved model
        :param path: path where model is loaded
        :return:
        """
        state = torch.load(path)
        self.best_val_results = state['best_val']
        self.best_val_mrr = self.best_val_results['mrr']
        self.best_epoch = state['best_epoch']
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])

    def build_graph(self):
        g = dgl.DGLGraph()
        g.add_nodes(self.num_ent)

        if not self.p.rat:
            g.add_edges(self.train_data[:, 0], self.train_data[:, 2])
            g.add_edges(self.train_data[:, 2], self.train_data[:, 0])
        else:
            if self.p.ss > 0:
                sampleSize = self.p.ss
            else:
                sampleSize = self.num_ent - 1
            g.add_edges(self.train_data[:, 0], np.random.randint(
                low=0, high=sampleSize, size=self.train_data[:, 2].shape))
            g.add_edges(self.train_data[:, 2], np.random.randint(
                low=0, high=sampleSize, size=self.train_data[:, 0].shape))
        return g

    def get_data_iter(self):
        """
        get data loader for train, valid and test section
        :return: dict
        """
        def get_data_loader(dataset_class, split):
            return DataLoader(
                dataset_class(self.triplets[split], self.num_ent, self.p),
                batch_size=self.p.batch_size,
                shuffle=True,
                num_workers=self.p.num_workers,
                drop_last=False
            )
        return {
            'train': get_data_loader(TrainDataset, 'train'),

            'valid_tail': get_data_loader(TestDataset, 'valid_tail'),

            'test_tail': get_data_loader(TestDataset, 'test_tail')
        }
    def get_edge_dir_and_norm(self):
        """
        :return: edge_type: indicates type of each edge: [E]
        """
        in_deg = self.g.in_degrees(range(self.g.number_of_nodes())).float()
        norm = in_deg ** -0.5
        norm[torch.isinf(norm).bool()] = 0
        self.g.ndata['xxx'] = norm
        self.g.apply_edges(
            lambda edges: {'xxx': edges.dst['xxx'] * edges.src['xxx']})
        if self.p.gpu >= 0:
            norm = self.g.edata.pop('xxx').squeeze().to("cuda")
            edge_type = torch.tensor(np.concatenate(
                [self.train_data[:, 1], self.train_data[:, 1] + self.num_rels])).to("cuda")
        else:
            norm = self.g.edata.pop('xxx').squeeze()
            edge_type = torch.tensor(np.concatenate(
                [self.train_data[:, 1], self.train_data[:, 1] + self.num_rels]))
        return edge_type, norm

    def get_model(self):
        if self.p.n_layer > 0:
            if self.p.score_func.lower() == 'transe':
                model = GCN_TransE(num_ent=self.num_ent, num_rel=self.num_rels, num_base=self.p.num_bases,
                                   init_dim=self.p.init_dim, gcn_dim=self.p.gcn_dim, embed_dim=self.p.embed_dim,
                                   n_layer=self.p.n_layer, edge_type=self.edge_type, edge_norm=self.edge_norm,
                                   bias=self.p.bias, gcn_drop=self.p.gcn_drop, opn=self.p.opn,
                                   hid_drop=self.p.hid_drop, gamma=self.p.gamma, wni=self.p.wni, wsi=self.p.wsi, encoder=self.p.encoder, use_bn=(not self.p.nobn), ltr=(not self.p.noltr))
            elif self.p.score_func.lower() == 'distmult':
                model = GCN_DistMult(num_ent=self.num_ent, num_rel=self.num_rels, num_base=self.p.num_bases,
                                     init_dim=self.p.init_dim, gcn_dim=self.p.gcn_dim, embed_dim=self.p.embed_dim,
                                     n_layer=self.p.n_layer, edge_type=self.edge_type, edge_norm=self.edge_norm,
                                     bias=self.p.bias, gcn_drop=self.p.gcn_drop, opn=self.p.opn,
                                     hid_drop=self.p.hid_drop, wni=self.p.wni, wsi=self.p.wsi, encoder=self.p.encoder, use_bn=(not self.p.nobn), ltr=(not self.p.noltr))
            elif self.p.score_func.lower() == 'conve':
                model = GCN_ConvE(num_ent=self.num_ent, num_rel=self.num_rels, num_base=self.p.num_bases,
                                  init_dim=self.p.init_dim, gcn_dim=self.p.gcn_dim, embed_dim=self.p.embed_dim,
                                  n_layer=self.p.n_layer, edge_type=self.edge_type, edge_norm=self.edge_norm,
                                  bias=self.p.bias, gcn_drop=self.p.gcn_drop, opn=self.p.opn,
                                  hid_drop=self.p.hid_drop, input_drop=self.p.input_drop,
                                  conve_hid_drop=self.p.conve_hid_drop, feat_drop=self.p.feat_drop,
                                  num_filt=self.p.num_filt, ker_sz=self.p.ker_sz, k_h=self.p.k_h, k_w=self.p.k_w, wni=self.p.wni, wsi=self.p.wsi, encoder=self.p.encoder, use_bn=(not self.p.nobn), ltr=(not self.p.noltr))

            else:
                raise KeyError(
                    f'score function {self.p.score_func} not recognized.')

        elif self.p.literal:
            # Load literals
            numerical_literals = np.load(f'../datasets/{args.dataset}/literals/numerical_literals.npy', allow_pickle=True)
            missing = np.where(np.random.uniform(0, 1, size=[numerical_literals.shape[0],numerical_literals.shape[1]]) <= self.p.numeric_density, 1, 0)
            # Initialize KBLN RBF parameters
            if self.p.name == 'KBLN':
                X_train = np.load(f'../datasets/{args.dataset}/train.npy')
                h = X_train[:, 0].astype('int')
                t = X_train[:, 1].astype('int')

                n = numerical_literals[h, :] - numerical_literals[t, :]
                c = np.mean(n, axis=0).astype('float32')  # size: (n_literals)
                var = np.var(n, axis=0) + 1e-6  # size: (n_literals), added eps to avoid degenerate case
                max_lit, min_lit = np.max(numerical_literals, axis=0), np.min(numerical_literals, axis=0)
                numerical_literals = (numerical_literals - min_lit) / (max_lit - min_lit + 1e-8)
                numerical_literals = numerical_literals * missing
                model = KBLN(self.num_ent, self.num_rels, numerical_literals.astype('float32'), c, var, params=self.p)

            # Normalize numerical literals

            max_lit, min_lit = np.max(numerical_literals, axis=0), np.min(numerical_literals, axis=0)
            numerical_literals = (numerical_literals - min_lit) / (max_lit - min_lit + 1e-8)
            numerical_literals = numerical_literals * missing

            # Appendix C.1 Feasibility Test
            if self.p.ft == 'binary':
                numerical_literals = np.float32(np.where(numerical_literals>0, 1, 0))

            if self.p.name == 'RAKGE':
                model = RAKGE(self.num_ent, self.num_rels, numerical_literals, params=self.p)

            # LiteralE + TransE
            elif self.p.name == 'TransELiteral_gate':
                model = TransELiteral_gate(self.num_ent, self.num_rels, numerical_literals, params=self.p)

        else:
            if self.p.score_func.lower() == 'transe':
                model = TransE(self.num_ent, self.num_rels, params=self.p)
            elif self.p.score_func.lower() == 'distmult':
                model = DistMult(self.num_ent, self.num_rels, params=self.p)
            elif self.p.score_func.lower() == 'conve':
                model = ConvE(self.num_ent, self.num_rels, params=self.p)
            elif self.p.score_func.lower() == 'hake':
                model = HAKE(self.num_ent, self.num_rels, params=self.p)

            else:
                raise NotImplementedError

        if self.p.gpu >= 0:
            model.to("cuda")
        return model



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser For Arguments',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', default='test_run',
                        help='Set run name for saving/restoring models')
    parser.add_argument('--data', dest='dataset', default='credit',
                        help='Dataset to use, default: credit')
    parser.add_argument('--score_func', dest='score_func', default='conve',
                        help='Score Function for Link prediction')
    parser.add_argument('--opn', dest='opn', default='corr',
                        help='Composition Operation to be used in CompGCN')
    parser.add_argument('--batch', dest='batch_size',
                        default=256, type=int, help='Batch size')
    parser.add_argument('--gpu', type=int, default=0,
                        help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')
    parser.add_argument('--epoch', dest='max_epochs',
                        type=int, default=1000, help='Number of epochs')
    parser.add_argument('--l2', type=float, default=0.0,
                        help='L2 Regularization for Optimizer')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Starting Learning Rate')
    parser.add_argument('--lbl_smooth', dest='lbl_smooth',
                        type=float, default=0.0, help='Label Smoothing')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of processes to construct batches')
    parser.add_argument('--seed', dest='seed', default=12345,
                        type=int, help='Seed for randomization')
    parser.add_argument('--restore', dest='restore', action='store_true',
                        help='Restore from the previously saved model')
    parser.add_argument('--bias', dest='bias', action='store_true',
                        help='Whether to use bias in the model')
    parser.add_argument('--num_bases', dest='num_bases', default=-1, type=int,
                        help='Number of basis relation vectors to use')
    parser.add_argument('--init_dim', dest='init_dim', default=100, type=int,
                        help='Initial dimension size for entities and relations')
    parser.add_argument('--gcn_dim', dest='gcn_dim', default=200,
                        type=int, help='Number of hidden units in GCN')
    parser.add_argument('--embed_dim', dest='embed_dim', default=None, type=int,
                        help='Embedding dimension to give as input to score function')
    parser.add_argument('--n_layer', dest='n_layer', default=0,
                        type=int, help='Number of GCN Layers to use')
    parser.add_argument('--gcn_drop', dest='gcn_drop', default=0.1,
                        type=float, help='Dropout to use in GCN Layer')
    parser.add_argument('--hid_drop', dest='hid_drop',
                        default=0.7, type=float, help='Dropout after GCN')

    parser.add_argument('--gamma', dest='gamma', default=9.0,
                        type=float, help='TransE: Gamma to use')

    # ConvE specific hyperparameters
    parser.add_argument('--conve_hid_drop', dest='conve_hid_drop', default=0.3, type=float,
                        help='ConvE: Hidden dropout')
    parser.add_argument('--feat_drop', dest='feat_drop',
                        default=0.2, type=float, help='ConvE: Feature Dropout')
    parser.add_argument('--input_drop', dest='input_drop', default=0.2,
                        type=float, help='ConvE: Stacked Input Dropout')
    parser.add_argument('--k_w', dest='k_w', default=20,
                        type=int, help='ConvE: k_w')
    parser.add_argument('--k_h', dest='k_h', default=10,
                        type=int, help='ConvE: k_h')
    parser.add_argument('--num_filt', dest='num_filt', default=200, type=int,
                        help='ConvE: Number of filters in convolution')
    parser.add_argument('--ker_sz', dest='ker_sz', default=7,
                        type=int, help='ConvE: Kernel size to use')


    # HAKE specific hyperparameters
    parser.add_argument('--modulus_weight', dest='modulus_weight', default=1.0,
                        type=float, help='HAKE: modulus weight to use')
    parser.add_argument('--phase_weight', dest='phase_weight', default=3.0,
                        type=float, help='HAKE: phase_weight to use')

    parser.add_argument('--rat', action='store_true',
                        default=False, help='random adacency tensors')
    parser.add_argument('--wni', action='store_true',
                        default=False, help='without neighbor information')
    parser.add_argument('--wsi', action='store_true',
                        default=False, help='without self-loop information')
    parser.add_argument('--ss', dest='ss', default=-1,
                        type=int, help='sample size (sample neighbors)')
    parser.add_argument('--nobn', action='store_true',
                        default=False, help='no use of batch normalization in aggregation')
    parser.add_argument('--noltr', action='store_true',
                        default=False, help='no use of linear transformations for relation embeddings')

    parser.add_argument('--encoder', dest='encoder',
                        default='compgcn', type=str, help='which encoder to use')

    # for KGE models
    parser.add_argument('--x_ops', dest='x_ops', default="")
    parser.add_argument('--r_ops', dest='r_ops', default="")

    # for literal models
    parser.add_argument('--literal', action='store_true', default=False)
    parser.add_argument('--tolerance', default=100, type=int)

    # for RAKGE model
    parser.add_argument('--att_dim', dest='att_dim', default=200, type=int)
    parser.add_argument('--head_num', dest='head_num', default=5, type=int)
    parser.add_argument('--drop', dest='drop',
                        default=0.7, type=float, help='Dropout RAKGE')
    ## order score
    parser.add_argument('--order', dest='order', default=0.25, type=float)

    ## contrastive learning
    parser.add_argument('--scale', dest='scale', default=0.25, type=float,
                        help='coefficient of contrastive loss in RAKGE')

    ## ratio of numeric value
    ### If you wanna reproduce the results of Figure 4,
    parser.add_argument('--numeric_density', dest='numeric_density', default=0.8, type=float, help='For the real-world setting, we dropped numeric values')

    ## Appendix C.1 Feasibility Test
    parser.add_argument('--ft', default='numeric', help='binary or numeric')


    args = parser.parse_args()
    if not args.restore and args.literal == False:
        args.name = args.encoder.lower() + '-' + args.score_func.lower() + \
            '-' + args.opn + args.name
    elif args.literal == True:
        args.name = args.name

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    runner = Runner(args)
    runner.fit()
