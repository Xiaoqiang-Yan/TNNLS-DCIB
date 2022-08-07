import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal, Independent
from torch.nn.functional import softplus


class Model_1(nn.Module):
    def __init__(self, cluster_num, data_dim, feature_dim):
        super(Model_1, self).__init__()
        self.data_dim = data_dim
        self.feature_dim = feature_dim
        self.cluster_num = cluster_num
        self.name = 'Model_1'
        self.cluster = nn.Sequential(
            nn.Linear(self.feature_dim, self.cluster_num)
        )

        self.net = nn.Sequential(
            nn.Linear(self.data_dim, self.data_dim),
            nn.BatchNorm1d(self.data_dim),
            nn.ReLU(),
            nn.Linear(self.data_dim, self.feature_dim),
            nn.BatchNorm1d(self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim * 2),
            nn.BatchNorm1d(self.feature_dim * 2),
            nn.ReLU(),
        )
        _initialize_weights(self)

    def forward(self, input, flag=False, fea=None):
        params = self.net(input)
        mu, sigma = params[:, :self.feature_dim], params[:, self.feature_dim:]
        sigma = softplus(sigma) + 1e-7  # Make sigma always positive
        x_P_F = Independent(Normal(loc=mu, scale=sigma), 1)
        x_feature = x_P_F.rsample()
        if not flag:
            x_cluster = self.cluster(x_feature)
            x_cluster = torch.softmax(x_cluster, dim=1)
        else:
            x_feature = x_feature + fea
            x_cluster = self.cluster(x_feature)
            x_cluster = torch.softmax(x_cluster, dim=1)
        return x_feature, x_cluster, x_P_F


class Model_2(nn.Module):
    def __init__(self, cluster_num, data_dim, feature_dim):
        super(Model_2, self).__init__()
        self.data_dim = data_dim
        self.feature_dim = feature_dim
        self.cluster_num = cluster_num
        self.name = 'Model_2'
        self.cluster = nn.Sequential(
            nn.Linear(self.feature_dim, self.cluster_num),
        )
        self.net = nn.Sequential(
            nn.Linear(self.data_dim, self.data_dim),
            nn.BatchNorm1d(self.data_dim),
            nn.ReLU(),
            nn.Linear(self.data_dim, self.feature_dim),
            nn.BatchNorm1d(self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim * 2),
            nn.BatchNorm1d(self.feature_dim * 2),
            nn.ReLU(),

        )
        _initialize_weights(self)

    def forward(self, input, flag=False, fea=None):
        params = self.net(input)
        mu, sigma = params[:, :self.feature_dim], params[:, self.feature_dim:]
        sigma = softplus(sigma) + 1e-7  # Make sigma always positive
        x_P_F = Independent(Normal(loc=mu, scale=sigma), 1)
        x_feature = x_P_F.rsample()
        if not flag:
            x_cluster = self.cluster(x_feature)
            x_cluster = torch.softmax(x_cluster, dim=1)
        else:
            x_feature = x_feature + fea
            x_cluster = self.cluster(x_feature)
            x_cluster = torch.softmax(x_cluster, dim=1)
        return x_feature, x_cluster, x_P_F


class Model_fusion(nn.Module):
    def __init__(self, feature_dim, cluster_num):
        super(Model_fusion, self).__init__()
        self.feature_dim = feature_dim
        self.cluster_num = cluster_num
        self.name = 'Model_f'
        self.net = nn.Sequential(
            nn.Linear(self.feature_dim * 2, self.feature_dim * 2),
            nn.BatchNorm1d(self.feature_dim * 2),
            nn.ReLU(),
            nn.Linear(self.feature_dim * 2, self.feature_dim * 2),
            nn.Dropout(p=0.1),
            nn.ReLU()
        )
        _initialize_weights(self)

    def forward(self, input1, input2):
        input = torch.cat((input1, input2), 1)
        params = self.net(input)
        mu, sigma = params[:, :self.feature_dim], params[:, self.feature_dim:]
        sigma = softplus(sigma) + 1e-7  # Make sigma always positive
        x_P_F = Independent(Normal(loc=mu, scale=sigma), 1)
        x_feature = x_P_F.rsample()
        return x_feature, x_P_F



class MIEstimator(nn.Module):
    def __init__(self, size1, size2):
        super(MIEstimator, self).__init__()

        # Vanilla MLP
        self.net = nn.Sequential(
            nn.Linear(size1 + size2, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 1),
        )

    # Gradient for JSD mutual information estimation and EB-based estimation
    def forward(self, x1, x2):
        pos = self.net(torch.cat([x1, x2], 1))  # Positive Samples
        temp = torch.roll(x1, 1, 0)
        neg = self.net(torch.cat([temp, x2], 1))
        return -softplus(-pos).mean() - softplus(neg).mean(), pos.mean() - neg.exp().mean() + 1


def _initialize_weights(self):
    # print("initialize %s", self.name)
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            assert (m.track_running_stats == self.batchnorm_track)
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()


def UD_constraint_f(classer):
    CL = classer.detach().cpu().numpy()
    N, K = CL.shape
    CL = CL.T
    r = np.ones((K, 1)) / K
    c = np.ones((N, 1)) / N
    CL **= 10
    inv_K = 1. / K
    inv_N = 1. / N
    err = 1e3
    _counter = 0
    while err > 1e-2 and _counter < 150:
        r = inv_K / (CL @ c)
        c_new = inv_N / (r.T @ CL).T
        if _counter % 10 == 0:
            err = np.nansum(np.abs(c / c_new - 1))
        c = c_new
        _counter += 1
    CL *= np.squeeze(c)
    CL = CL.T
    CL *= np.squeeze(r)
    CL = CL.T
    argmaxes = np.nanargmax(CL, 0)
    newL = torch.LongTensor(argmaxes)
    return newL