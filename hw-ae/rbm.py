import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class RBM(nn.Module):
    def __init__(self, in_features, out_features, k=2):
        super(RBM, self).__init__()
        self.fc = "TODO"
        self.bias_v = "TODO"
        self.bias_h = "TODO"
        self.k = k

    def sample_p(self, p):
       return "TODO"

    def v2h(self, v):
        p_h = F.sigmoid(v @ self.fc + self.bias_h)
        return p_h, self.sample_p(p_h)

    def h2v(self, h):
        p_v = "TODO"
        return p_v, self.sample_p(p_v)

    def gibbs_h2v2h(self, h):
        p_v, a_v = self.h2v(h)
        p_h, a_h = self.v2h(p_v)
        return p_v, a_v, p_h, a_h

    def contrastive_divergence(self, x, lr):
        pos_p_h, pos_a_h = self.v2h(x)

        a_h = pos_a_h
        for _ in range(self.k):
            p_v, a_v, p_h, a_h = self.gibbs_h2v2h(a_h)

        self.fc += "TODO"
        self.bias_v += "TODO"
        self.bias_h += "TODO"

    def v2h2v(self, x):
        h, _ = self.v2h(x)
        v, _ = self.h2v(h)
        return v

class RBMHandle():
    def __init__(self):
        self.models = []

    def v2h(self, x):
        for prev_m in self.models:
            x, _ = prev_m.v2h(x)
        return x

    def h2v(self, h):
        for prev_m in self.models[::-1]:
            h, _ = prev_m.h2v(h)
        return h

    def v2h2v(self, x):
        return self.h2v(self.v2h(x))

    def append(self, m):
        self.models.append(m)

    def __len__(self):
        return len(self.models)

    def __getitem__(self, i):
        return self.models[i]