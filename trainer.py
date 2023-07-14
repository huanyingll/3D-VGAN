'''
trainer.py

Train 3dgan models
'''

import torch
from torch import optim
from torch import nn

from utils import *
import os

from model import net_G, net_D
import torch.nn.functional as F
# added
import datetime
import time
import matplotlib.pyplot as plt
import numpy as np
import params
from tqdm import tqdm

from utils import ShapeNetDataset

def loss_function(x_hat, x, mu, log_var):
    """
    Calculate the loss. Note that the loss includes two parts.
    :param x_hat:
    :param x:
    :param mu:
    :param log_var:
    :return: total loss, BCE and KLD of our model
    """
    # 1. the reconstruction loss.
    # We regard the MNIST as binary classification
    BCE = F.binary_cross_entropy(x_hat, x, reduction='sum')

    # 2. KL-divergence
    # D_KL(Q(z|X) || P(z)); calculate in closed form as both dist. are Gaussian
    # here we assume that \Sigma is a diagonal matrix, so as to simplify the computation
    KLD = 0.5 * torch.sum(torch.exp(log_var) + torch.pow(mu, 2) - 1. - log_var)

    # 3. total loss
    loss = BCE + KLD
    return loss
def trainer(args):
    now_epoch=0
    train_dsets = ShapeNetDataset('../../datasets/5_spines/L_npys/')
    # val_dsets = ShapeNetDataset(dsets_path, args, "val")

    train_dset_loaders = torch.utils.data.DataLoader(train_dsets, batch_size=params.batch_size, shuffle=True,
                                                     num_workers=1)
    # val_dset_loaders = torch.utils.data.DataLoader(val_dsets, batch_size=args.batch_size, shuffle=True, num_workers=1)

    # model define
    D = net_D()
    G = net_G()
    D_solver = optim.Adam(D.parameters(), lr=params.d_lr, betas=params.beta)
    G_solver = optim.Adam(G.parameters(), lr=params.g_lr, betas=params.beta)

    D.to(params.device)
    G.to(params.device)
    if now_epoch!=0:
        G.load_state_dict(torch.load(params.model_dir+'G_'+str(now_epoch)+'.pth'))
        D.load_state_dict(torch.load(params.model_dir+'D_'+str(now_epoch)+'.pth'))
    criterion_D = nn.MSELoss()
    criterion_G = nn.MSELoss()
    D.train()
    G.train()
    for epoch in range(params.epochs):
        epoch=epoch+now_epoch
        start = time.time()
        running_g_recon = 0.0
        running_d_loss = 0.0
        running_g_create = .0
        for i, X in tqdm(enumerate(train_dset_loaders),total=len(train_dset_loaders)):
            X = X.to(params.device)
            X =X.permute(0,4,1,2,3)  # 交换第二与第三维度
            Z = X
            # ============= Train the discriminator =============#
            d_real = D(X)

            fake, mu ,logvar = G(Z)
            d_fake = D(fake)
            real_labels = torch.ones_like(d_real).to(params.device)
            fake_labels = torch.zeros_like(d_fake).to(params.device)
            # print (d_fake.size(), fake_labels.size())

            # print (d_real.size(), real_labels.size())
            d_real_loss = criterion_D(d_real, real_labels)

            d_fake_loss = criterion_D(d_fake, fake_labels)

            d_loss = d_real_loss + d_fake_loss

            # no deleted
            d_real_acu = torch.ge(d_real.squeeze(), 0.5).float()
            d_fake_acu = torch.le(d_fake.squeeze(), 0.5).float()
            d_total_acu = torch.mean(torch.cat((d_real_acu, d_fake_acu), 0))
            if d_total_acu < params.d_thresh:
                D.zero_grad()
                d_loss.backward()
                D_solver.step()

            # =============== Train the generator ===============#
            # print (X)
            x_rec_logits,mu,logvar=G(X)
            fake, mu, logvar = G(Z)  # generated fake: 0-1, X: 0/1
            d_fake = D(fake)
            adv_g_loss = criterion_G(d_fake, real_labels)
            recon_g_loss = loss_function(x_rec_logits, X, mu, logvar)
            #recon_g_loss = criterion_G(fake, X)
            g_loss = adv_g_loss + recon_g_loss
            D.zero_grad()
            G.zero_grad()
            g_loss.backward()
            G_solver.step()

            # =============== logging each 10 iterations ===============#

            running_g_recon += recon_g_loss.item() * X.size(0)
            running_d_loss += d_loss.item() * X.size(0)
            running_g_create += recon_g_loss.item() * X.size(0)
            # =============== each epoch save model or save image ===============#
        epoch_g_recon = running_g_recon / len(train_dset_loaders)
        epoch_d_loss = running_d_loss / len(train_dset_loaders)
        epoch_g_create = running_g_create / len(train_dset_loaders)

        end = time.time()
        epoch_time = end - start
        print('Epochs-{} , D(x) : {:.4}, D(G(x)) : {:.4}, G(x)-recon : {:.4}'.format(epoch, epoch_d_loss, epoch_g_create,epoch_g_recon))
        print('Elapsed Time: {:.4} min'.format(epoch_time / 60.0))
        if (epoch+1) % params.model_save_step == 0:
            print('model_saved, images_saved...')
            torch.save(G.state_dict(),params.model_dir + 'G_'+str(epoch)+'.pth')
            torch.save(D.state_dict(),params.model_dir + 'D_'+str(epoch)+'.pth')
            samples = fake.cpu().data[:2].squeeze().numpy()
            # print (samples.shape)
            # image_saved_path = '../images'
            SavePloat_Voxels(samples, 'imgs', epoch)