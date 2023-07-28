import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np
def get_param(shape):
    param = nn.Parameter(torch.Tensor(*shape))
    nn.init.xavier_normal_(param.data)
    return param


class Gate(nn.Module):

    def __init__(self,
                 input_size,
                 output_size,
                 gate_activation=torch.sigmoid):

        super(Gate, self).__init__()
        self.output_size = output_size

        self.gate_activation = gate_activation
        self.g = nn.Linear(input_size, output_size)
        self.g1 = nn.Linear(output_size, output_size, bias=False)
        self.g2 = nn.Linear(input_size-output_size, output_size, bias=False)
        self.gate_bias = nn.Parameter(torch.zeros(output_size))

    def forward(self, x_ent, x_lit):
        x = torch.cat([x_ent, x_lit], x_lit.ndimension()-1)
        g_embedded = torch.tanh(self.g(x))
        gate = self.gate_activation(self.g1(x_ent) + self.g2(x_lit) + self.gate_bias)
        output = (1-gate) * x_ent + gate * g_embedded

        return output


class TransELiteral_gate(nn.Module):
    def __init__(self, num_ents, num_rels, numerical_literals, missing, params=None):
        super(TransELiteral_gate, self).__init__()

        self.bceloss = torch.nn.BCELoss()
        self.p = params

        self.device = "cuda"
        self.emb_dim = self.p.init_dim
        self.att_dim = self.p.att_dim
        self.head_num = self.p.head_num

        # entity, relation embedding table
        self.emb_e = get_param((num_ents, self.p.init_dim))
        self.emb_rel = get_param((num_rels , self.p.init_dim))

        self.numerical_literals = torch.from_numpy(numerical_literals).cuda()
        self.n_num_lit = self.numerical_literals.size(1)
        # gating
        self.emb_num_lit = Gate(self.emb_dim+self.n_num_lit, self.emb_dim)
        self.bias = nn.Parameter(torch.zeros(num_ents))
        self.inp_drop = nn.Dropout(p=self.p.input_drop)

        self.missing = torch.from_numpy(missing).cuda()
        self.numerical_literals1 = torch.from_numpy(numerical_literals).cuda()
        self.numerical_literals = self.numerical_literals1 * self.missing

    def forward(self, g, e1, rel, e2_multi):

        e1_emb = torch.index_select(self.emb_e, 0, e1)
        rel_emb = torch.index_select(self.emb_rel, 0, rel)
        e2_multi_emb = self.emb_e

        e1_num_lit = torch.index_select(self.numerical_literals, 0, e1)
        e2_num_lit = self.numerical_literals


        e1_emb = self.emb_num_lit(e1_emb, e1_num_lit)
        e2_multi_emb = self.emb_num_lit(e2_multi_emb, e2_num_lit)

        e1_emb = self.inp_drop(e1_emb)

        e2_multi_emb = self.inp_drop(e2_multi_emb)

        score = self.p.gamma - \
            torch.norm(((e1_emb + rel_emb).unsqueeze(1) - e2_multi_emb.unsqueeze(0)), p=1, dim=2)

        pred = torch.sigmoid(score)

        if self.training:
            return self.bceloss(pred,e2_multi)
        else:
            return pred



class KBLN(nn.Module):
    def __init__(self, num_ents, num_rels, numerical_literals, c, var, params=None):
        super(KBLN, self).__init__()

        self.bceloss = torch.nn.BCELoss()
        self.p = params
        self.num_ents = num_ents

        self.device = "cuda"
        self.emb_dim = self.p.init_dim
        self.att_dim = self.p.att_dim
        self.head_num = self.p.head_num

        # entity, relation embedding table
        self.emb_e = get_param((num_ents, self.p.init_dim))
        self.emb_rel = get_param((num_rels , self.p.init_dim))

        self.numerical_literals = torch.from_numpy(numerical_literals).cuda()
        self.n_num_lit = self.numerical_literals.size(1)

        self.c = Variable(torch.FloatTensor(c)).cuda()
        self.var = Variable(torch.FloatTensor(var)).cuda()

        self.nf_weights = get_param((num_rels, self.n_num_lit))

        # gating
        self.emb_num_lit = Gate(self.emb_dim+self.n_num_lit, self.emb_dim)
        self.bias = nn.Parameter(torch.zeros(num_ents))
        self.inp_drop = nn.Dropout(p=self.p.hid_drop)

    def forward(self, g, e1, rel, e2_multi):
        e1_emb = torch.index_select(self.emb_e, 0, e1)
        rel_emb = torch.index_select(self.emb_rel, 0, rel)
        e1_emb = self.inp_drop(e1_emb)
        rel_emb = self.inp_drop(rel_emb)

        e2_multi_emb = self.emb_e

        score_l = torch.mm(e1_emb * rel_emb, e2_multi_emb.transpose(1,0))

        n_h = torch.index_select(self.numerical_literals, 0, e1) #(batch, n_lits)

        n_t = self.numerical_literals #(n_ents, n_lits)

        # Features (batch_size x num_ents x n_lit)
        n = n_h.unsqueeze(1).repeat(1, self.num_ents, 1) - n_t
        phi = self.rbf(n) #(batch, num_ents, n_lits)
        # Weights (batch_size, 1, n_lits)
        w_nf = torch.index_select(self.nf_weights, 0, rel)

        # (batch_size, num_ents)
        score_n = torch.bmm(phi, w_nf.unsqueeze(2)).squeeze(2)
        """ End numerical literals """

        score = F.sigmoid(score_l + score_n)
        if self.training:
            return self.calc_loss(score, e2_multi)
        else:
            return score

    def rbf(self, n):
        """
        Apply RBF kernel parameterized by (fixed) c and var, pointwise.
        n: (batch_size, num_ents, n_lit)
        """
        return torch.exp(-(n - self.c)**2 / self.var)

    def calc_loss(self, pred, label):
        return self.loss(pred, label)

    def loss(self, pred, true_label):
        return self.bceloss(pred, true_label)


