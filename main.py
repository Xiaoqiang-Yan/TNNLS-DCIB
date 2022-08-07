
import argparse
import torch
import numpy as np
import utils1
from mutual_information import mutual_information
from dataset import Dateset_mat
from tqdm import trange
from model.model import Model_1, Model_2, Model_fusion, UD_constraint_f, MIEstimator
from itertools import chain as ichain
import warnings
import random
import torch.distributions.normal as normal

warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_root", default=r'dataset/nus/', type=str)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--num_epochs", type=int, default=500)
config = parser.parse_args()
config.max_ACC = 0
Dataset = Dateset_mat(config.dataset_root, True)
dataset = Dataset.getdata()
label = np.array(dataset[dataset.__len__()-1]) - 1
label = np.squeeze(label)
cluster_num = max(label) + 1

img_1 = torch.tensor(dataset[0], dtype=torch.float32).to(device)
# img_2 = torch.tensor(dataset[1], dtype=torch.float32).to(device)
txt_1 = torch.tensor(dataset[2], dtype=torch.float32).to(device)
[a, b] = img_1.size()
print("clustering number: %d data number: %d data_feature: %d" % (cluster_num, a, b))
feature_dim = 130
criterion_cross = torch.nn.CrossEntropyLoss().to(device)

prior_loc = torch.zeros(a, feature_dim)
prior_scale = torch.ones(a, feature_dim)
prior = normal.Normal(prior_loc, prior_scale)


def run():
    max_ACC = 0
    model_1 = Model_1(cluster_num=cluster_num, data_dim=b, feature_dim=feature_dim).to(device)
    model_2 = Model_2(cluster_num=cluster_num, data_dim=b, feature_dim=feature_dim).to(device)
    model_f = Model_fusion(feature_dim=feature_dim, cluster_num=cluster_num).to(device)
    mi_estimator = MIEstimator(feature_dim, feature_dim).to(device)

    ge_chain_1 = ichain(model_1.parameters(), model_2.parameters())
    optimiser_1 = torch.optim.Adam(ge_chain_1, lr=config.lr)

    ge_chain_2 = ichain(model_f.parameters(), mi_estimator.parameters())
    optimiser_all = torch.optim.Adam(ge_chain_2, lr=config.lr)

    for epoch in trange(config.num_epochs):
        model_1.train()
        model_1.zero_grad()
        model_2.train()
        model_2.zero_grad()
        model_f.train()
        model_f.zero_grad()

        Flag = True
        beta = 1
        x_feature_1, _, P_1 = model_1(img_1)
        y_feature_1, _, P_2 = model_2(txt_1)
        fea, P_f = model_f(x_feature_1, y_feature_1)
        loss3_1 = mutual_information(fea, x_feature_1) \
                  + mutual_information(fea, y_feature_1)  # I(T1;Tf) + T(T2;Tf)
        lossKL, lossMI, lossM12KL = getMILoss(P_f, [P_1, P_2], mi_estimator)
        loss3_2 = lossKL + 10*lossMI + beta*lossM12KL
        loss3 = 0.01*loss3_2 - beta*loss3_1
        # loss3 = -0.1 * loss3_1

        print("loss3_1: %.3f lossKL %.3f lossM12KL %.3f" % (loss3_1, lossKL, lossM12KL))
        loss3.backward()
        optimiser_all.step()


        ############
        x_feature_1, _, _ = model_1(img_1)
        y_feature_1, _, _ = model_2(txt_1)
        fea, _ = model_f(x_feature_1, y_feature_1)

        x_feature_1, x_cluster_1, _ = model_1(img_1, Flag, fea)
        x_feature_2, x_cluster_2, _ = model_2(txt_1, Flag, fea)

        loss1_1 = mutual_information(x_cluster_1, x_cluster_2) + mutual_information(x_feature_1, x_feature_2)
        if epoch % 5 == 0:
            with torch.no_grad():
                UDC_img1 = UD_constraint_f(x_cluster_1).to(device)
                UDC_txt1 = UD_constraint_f(x_cluster_2).to(device)
        loss1_2 = criterion_cross(x_cluster_1, UDC_img1) + criterion_cross(x_cluster_2, UDC_txt1)
        loss1 = loss1_1 + loss1_2

        loss = loss1
        loss.backward()
        optimiser_1.step()

        if epoch % 5 == 0:
            acc1, nmi1 = getACC_NMI(model_1, img_1)
            acc2, nmi2 = getACC_NMI(model_2, txt_1)
            print("ACC1: %.4f NMI1: %.4f *** ACC2: %.4f NMI2: %.4f" % (acc1, nmi1, acc2, nmi2))
            print("loss1: %.3f loss3_1 %.3f loss3_2 %.3f" % (loss1, loss3_1, loss3_2))
            print("lossKL_1: %.3f lossMI %.3f lossM12KL %.3f" % (lossKL, lossMI, lossM12KL))

            if max(acc1, acc2) > max_ACC:
                max_ACC = acc1
    return max_ACC


def getACC_NMI(model, data1):
    model.eval()
    img_fea, x_out, _ = model(data1)

    pre_label = np.array(x_out.cpu().detach().numpy())
    pre_label = np.argmax(pre_label, axis=1)

    acc1 = utils1.metrics.acc(pre_label, label)
    nmi1 = utils1.metrics.nmi(pre_label, label)
    return acc1, nmi1


def getMILoss(P_F, P, mi_estimator):
    x_P_F = P_F.rsample()
    prior_sample = prior.sample().to(device)
    loss1_1 = 0
    loss2 = 0
    loss3 = 0
    for p in P:
        x_p = p.rsample()
        miG, _ = mi_estimator(x_P_F, x_p)
        loss2 += mutual_information(x_p, x_P_F)
        loss1_1 += -miG
        loss3 += torch.nn.functional.kl_div(x_p, prior_sample).to(device)
    return loss1_1, loss2, loss3


if __name__ == '__main__':
    run()


