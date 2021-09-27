import scipy.io
import torch
import numpy as np
import random


class Dateset_mat():
    def __init__(self, data_path, flag):
        self.flag = flag

        if self.flag:
            self.img_1 = scipy.io.loadmat(data_path + r"/img_1.mat")
            self.img_2 = scipy.io.loadmat(data_path + r"/img_2.mat")
            self.txt_1 = scipy.io.loadmat(data_path + r"/txt_1.mat")
        else:
            self.img = scipy.io.loadmat(data_path + r"/img.mat")
            self.txt = scipy.io.loadmat(data_path + r"/txt.mat")
        try:
            self.label = scipy.io.loadmat(data_path + r"/label.mat")
        except:
            self.label = scipy.io.loadmat(data_path + r"/L.mat")

    def getdata(self):
        self.data = []
        if self.flag:
            self.data.append(self.img_1["img"])
            self.data.append(self.img_2["img"])
            self.data.append(self.txt_1["txt"])
        else:
            self.data.append(self.img["img"])
            self.data.append(self.txt["txt"])
        self.data.append(self.label["L"])
        seed = 1230
        fix_seed(seed)
        return self.data


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

