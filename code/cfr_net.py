import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


def get_linear_layers(in_dim, layer_sizes, bn = False, activation = None):
    linear_layers = map(nn.Linear, [in_dim] + layer_sizes, layer_sizes)
    tmp = [linear_layers]
    if bn:
        bns = [nn.BatchNorm1d(dim) for dim in layer_sizes]
        tmp += [bns]
    if activation is not None:
        activations = [activation() for _ in range(len(layer_sizes))]
        tmp += [activations]
    tmp = zip(*tmp)
    return [module for pair in tmp for module in pair]


class CFRNet(nn.Module):
    def __init__(self, in_dim, repre_layer_sizes, pred_layer_sizes, bn = False):
        super(CFRNet, self).__init__()
        self.repre_layers = nn.Sequential(*(([nn.BatchNorm1d(in_dim)] if bn else [])
                                             + get_linear_layers(in_dim,repre_layer_sizes,bn,nn.ReLU)))
        self.pred_layers_treated = nn.Sequential(*get_linear_layers(repre_layer_sizes[-1], pred_layer_sizes, False, nn.ReLU))
        self.pred_layers_treated.add_module('out1',nn.Linear(pred_layer_sizes[-1],1))
        self.pred_layers_control = nn.Sequential(*get_linear_layers(repre_layer_sizes[-1], pred_layer_sizes, False, nn.ReLU))
        self.pred_layers_control.add_module('out0', nn.Linear(pred_layer_sizes[-1],1))

    def forward(self, x, t):
        self.repre = self.repre_layers(x)
        y = torch.where(t == 1, self.pred_layers_treated(self.repre), self.pred_layers_control(self.repre))
        return y

    def get_repre(self, x, device):
        self.eval()
        with torch.no_grad():
            return self.repre_layers.to(device)(x.to(device))


class WrappedDataLoader():
    def __init__(self, X, t, y, batch_size, shuffle=True, n_covered = None):
        if n_covered is None:
            self.ds = TensorDataset(torch.from_numpy(X).float(),
                                    torch.from_numpy(t.reshape(-1, 1)).int(),
                                    torch.from_numpy(y.reshape(-1, 1)).float())
        else:
            self.ds = TensorDataset(torch.from_numpy(X).float(),
                                    torch.from_numpy(t.reshape(-1, 1)).int(),
                                    torch.from_numpy(y.reshape(-1, 1)).float(),
                                    torch.from_numpy(n_covered).float())
        self.dl = DataLoader(self.ds, batch_size=batch_size, shuffle=shuffle)
        self.size = X.shape

    def get_X_size(self):
        return self.size

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        for b in iter(self.dl):
            yield b

