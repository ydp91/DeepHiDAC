import os

import torch
from torch import nn
import numpy as np
from config import parse_args
from loader import get_data_loader, preprocessing
from loss import LossCompute
from model import NN_D, NN_C,NN_O,CVAE
from utils import cal_speed_num_error,saveModel,saveModel_VAE,writecsv
from evalue import getAllAvgScore,generateTrajbyFile,collision_count
from traj_vis import plot_trajectories



os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



def train(args):
    D = NN_D(args.dim, args.mlp_dim, args.depth, args.heads, args.dropout).to(
        args.device)
    C = NN_C(args.dim, args.dropout).to(args.device)
    O = NN_O(args.dim, args.dropout).to(args.device)
    print(D)
    print(C)
    print(O)
    optimizer_D = torch.optim.Adam(D.parameters(), lr=args.lr,eps=1e-3)
    optimizer_C = torch.optim.Adam(C.parameters(), lr=args.lr,eps=1e-3)
    optimizer_O = torch.optim.Adam(O.parameters(), lr=args.lr,eps=1e-3)
    scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=args.lr_step, gamma=args.lr_gamma)
    scheduler_C = torch.optim.lr_scheduler.StepLR(optimizer_C, step_size=args.lr_step, gamma=args.lr_gamma)
    scheduler_O = torch.optim.lr_scheduler.StepLR(optimizer_O, step_size=args.lr_step, gamma=args.lr_gamma)
    train_dl = get_data_loader(args, 'train')
    lossfn = LossCompute(D, C, O)
    iternum=0 #迭代次数
    for i in range(args.epoch):
        D.train()
        C.train()
        O.train()

        for j, batch in enumerate(train_dl):
            iternum += 1
            (traj_init,tran_sfm, seq_start_end,obstacle) = batch
            tran_sfm,seq_start_end=tran_sfm.cuda(),seq_start_end.cuda()
            #plot_trajectories(tran_sfm.cpu())
            traj, rel_traj,speed,obstacle, dest,target = preprocessing(tran_sfm,obstacle, seq_start_end,'train') #轨迹、相对轨迹、方向
            #训练一次D
            optimizer_D.zero_grad()
            optimizer_C.zero_grad()
            optimizer_O.zero_grad()
            loss,loss_mean=lossfn.compute_loss_NoR(traj, rel_traj,obstacle,speed,dest,target, seq_start_end)
            loss.backward()
            optimizer_D.step()
            optimizer_C.step()
            optimizer_O.step()
            # for p in NN_D.parameters():
            #     p.data.clamp_(-0.2, 0.2)
            # for p in NN_C.parameters():
            #     p.data.clamp_(-0.2, 0.2)
            # for p in NN_O.parameters():
            #     p.data.clamp_(-0.2, 0.2)

            print('Epoch:', i + 1, 'batch:', j)
            print("loss:", round(loss.item(), 3))
            print("loss_mean:", round(loss_mean.item(), 3))
        scheduler_D.step()
        scheduler_C.step()
        scheduler_O.step()
        if (i + 1) % 10 == 0:
            saveModel(D, C,O, args,str(i+1))

def train_cvae(args):
    D = NN_D(args.dim, args.mlp_dim, args.depth, args.heads, args.dropout).to(args.device)
    D.load_state_dict(torch.load(args.D_dict))
    C = NN_C(args.dim, args.dropout).to(args.device)
    C.load_state_dict(torch.load(args.C_dict))
    O = NN_O(args.dim, args.dropout).to(args.device)
    O.load_state_dict(torch.load(args.O_dict))
    D.eval()
    C.eval()
    O.eval()
    cvae=CVAE(args.dim,args.mlp_dim,args.heads,args.dropout,args.depth).to(args.device)
    #cvae.load_state_dict(torch.load(args.V_dict))
    print(cvae)
    optimizer = torch.optim.Adam(cvae.parameters(), lr=args.lr)
    train_dl = get_data_loader(args, 'train')
    lossfn = LossCompute(D, C, O,cvae,args)
    iternum=0 #迭代次数
    for i in range(args.epoch):
        for j, batch in enumerate(train_dl):
            iternum += 1
            (traj_init,tran_sfm, seq_start_end,obstacle) = batch
            tran_sfm,seq_start_end=tran_sfm.cuda(),seq_start_end.cuda()
            traj, rel_traj,speed,obstacle, dest,target = preprocessing(tran_sfm,obstacle, seq_start_end,'train') #轨迹、相对轨迹、方向
            #训练一次D
            optimizer.zero_grad()
            loss, l2, kl, loss_sfm, loss_cvae_recon,sfm_mean,cvae_mean=lossfn.compute_cvae_loss_NoR(traj, rel_traj, obstacle, speed,dest,target, seq_start_end)
            loss.backward()
            optimizer.step()
            print('Epoch:', i + 1, 'batch:', j)
            print("loss:", round(loss.item(), 3))
            print("cvae_l2:", round(l2.item(), 3))
            print("cvae_kl:", round(kl.item(), 3))
            #print("cvae_col:", round(col_loss.item(), 3))
            print("loss_sfm:", round(loss_sfm.item(), 3))
            print("loss_cvae_recon:", round(loss_cvae_recon.item(), 3))
            print("sfm_mean:", round(sfm_mean.item(), 3))
            print("cvae_mean:", round(cvae_mean.item(), 3))
        if (i + 1) % 10 == 0:
            saveModel_VAE(cvae, args,str(i+1))

if __name__ == '__main__':
    args = parse_args()
    train_cvae(args)



