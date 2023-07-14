
'''
utils.py

Some utility functions

'''

import matplotlib
import params

if params.device.type != 'cpu':
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.gridspec as gridspec
import numpy as np
from torch.utils import data
from torch.autograd import Variable
import torch
import os
import pickle
def SavePloat_Voxels(voxels, path, iteration):
    voxels = voxels[:2].__ge__(0.5)
    v=np.zeros([voxels.shape[0],voxels.shape[2],voxels.shape[3],voxels.shape[4]])
    for f in range(voxels.shape[0]):
        for m in range(voxels.shape[1]):
            for i in range(voxels.shape[2]):
                for j in range(voxels.shape[3]):
                    for k in range(voxels.shape[4]):
                        if voxels[f, m, i, j, k] == 1:
                            v[f, i, j, k] = 1
    fig = plt.figure(figsize=(32, 16))
    gs = gridspec.GridSpec(1, 2)
    gs.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(v):
        x, y, z = sample.nonzero()
        ax = plt.subplot(gs[i], projection='3d')
        ax.scatter(x, y, z, zdir='z', c='red')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # ax.set_aspect('equal')
    # print (path + '/{}.png'.format(str(iteration).zfill(3)))
    plt.savefig(path + '/{}.png'.format(str(iteration).zfill(3)), bbox_inches='tight')
    plt.close()


class ShapeNetDataset(data.Dataset):

    def __init__(self, root_models):
        self.root = root_models
        self.listdir = os.listdir(self.root)
        # print (self.listdir)
        # print (len(self.listdir)) # 10668

        data_size = len(self.listdir)
        #        self.listdir = self.listdir[0:int(data_size*0.7)]
        self.listdir = self.listdir[0:int(data_size)]

    def __getitem__(self, index):
        data = np.load(self.root + self.listdir[index], allow_pickle=True)
        return torch.FloatTensor(data)

    def __len__(self):
        return len(self.listdir)

def generateZ(batch):

    if params.z_dis == "norm":
        Z = torch.Tensor(batch, params.z_dim).normal_(0, 0.33).to(params.device)
    elif params.z_dis == "uni":
        Z = torch.randn(batch, params.z_dim).to(params.device).to(params.device)
    else:
        print("z_dist is not normal or uniform")

    return Z
