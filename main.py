import argparse
import torch
import numpy as np
import utils1
from mutual_information import mutual_information
from dataset import Dateset_mat
from tqdm import trange
from model.model import Model_1, Model_2, Model_fusion, UD_constraint_f
from itertools import chain as ichain
import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_root", default=r'dataset\esp', type=str)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--num_epochs", type=int, default=500)
config = parser.parse_args()
config.max_ACC = 0
Dataset = Dateset_mat(config.dataset_root, True)
dataset = Dataset.getdata()
label = np.array(dataset[dataset.__len__()-1]) - 1
label = np.squeeze(label)
cluster_num = max(label) + 1

img_1 = torch.tensor(dataset[0], dtype=torch.float32).to(device)
img_2 = torch.tensor(dataset[1], dtype=torch.float32).to(device)
txt_1 = torch.tensor(dataset[2], dtype=torch.float32).to(device)
[a, b] = img_1.size()
print("clustering number: %d data number: %d data_feature: %d" % (cluster_num, a, b))
feature_dim = 130
criterion_cross = torch.nn.CrossEntropyLoss().to(device)


def run():
    rand = (torch.randn(a, b) * 0.02).to(device)
    txt_2 = rand + txt_1
    model_1 = Model_1(cluster_num=cluster_num, data_dim=b, feature_dim=feature_dim).to(device)
    model_2 = Model_2(cluster_num=cluster_num, data_dim=b, feature_dim=feature_dim).to(device)
    ge_chain = ichain(model_1.parameters(), model_2.parameters())
    optimiser_1 = torch.optim.Adam(ge_chain, lr=config.lr)
    model_f = Model_fusion(feature_dim=feature_dim, cluster_num=cluster_num).to(device)
    optimiser_all = torch.optim.Adam(model_f.parameters(), lr=config.lr)
    for epoch in trange(config.num_epochs):
        model_1.train()
        model_1.zero_grad()
        model_2.train()
        model_2.zero_grad()
        model_f.train()
        model_f.zero_grad()

        Flag = True
        x_feature_1 = model_1.net(img_1)
        y_feature_1 = model_2.net(txt_1)
        fea = model_f(x_feature_1, y_feature_1)
        loss3_1 = mutual_information(fea, x_feature_1) + mutual_information(fea, y_feature_1)

        loss3 = loss3_1
        loss3.backward(retain_graph=True)
        optimiser_all.step()

        ############
        x_feature_1 = model_1.net(img_1)
        y_feature_1 = model_2.net(txt_2)
        fea = model_f(x_feature_1, y_feature_1)

        x_feature_1, x_cluster_1 = model_1(img_1, Flag, fea)
        x_feature_2, x_cluster_2 = model_1(img_2, Flag, fea)

        loss1_1 = mutual_information(x_feature_1, x_feature_2) + mutual_information(x_cluster_1, x_cluster_2)
        if epoch % 5 == 0:
            with torch.no_grad():
                UDC_img1 = UD_constraint_f(x_cluster_1).to(device)
                UDC_txt1 = UD_constraint_f(x_cluster_2).to(device)
        loss1_2 = criterion_cross(x_cluster_1, UDC_img1) + criterion_cross(x_cluster_2, UDC_txt1)
        loss1 = loss1_1 + loss1_2

        ###########
        y_feature_1, y_cluster_1 = model_2(txt_1, Flag, fea)
        y_feature_2, y_cluster_2 = model_2(txt_2, Flag, fea)
        loss2_1 = mutual_information(y_feature_1, y_feature_2) + mutual_information(y_cluster_1, y_cluster_2)
        if epoch % 5 == 0:
            with torch.no_grad():
                UDC_txt1 = UD_constraint_f(y_cluster_1).to(device)
                UDC_txt2 = UD_constraint_f(y_cluster_2).to(device)
        loss2_2 = criterion_cross(y_cluster_1, UDC_txt1) + criterion_cross(y_cluster_2, UDC_txt2)
        loss2 = loss2_1 + loss2_2

        loss = loss1 + loss2
        loss.backward()
        optimiser_1.step()

        if epoch % 2 == 0:
            acc, nmi = getACC_NMI(model_2, txt_1)
            print("ACC: %.3f NMI: %.3f" % (acc, nmi))


def getACC_NMI(model, data1):
    model.eval()
    _, x_out = model(data1)

    pre_label = np.array(x_out.cpu().detach().numpy())
    pre_label = np.argmax(pre_label, axis=1)

    acc1 = utils1.metrics.acc(pre_label, label)
    nmi1 = utils1.metrics.nmi(pre_label, label)
    return acc1, nmi1


if __name__ == '__main__':
    run()

