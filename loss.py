import torch
import numpy as np
import random
from sfm import deepsfm_update,deepsfm_update_R


class LossCompute:
    def __init__(self, NN_D, NN_C,NN_O,CVAE=None,args=None):
        self.netD = NN_D
        self.netC = NN_C
        self.netO = NN_O
        self.cvae = CVAE
        self.args=args

    def compute_loss_NoR(self, traj, rel_traj,obstacle,speed,dest,target, seq_start_end):
        D_tau=self.netD(rel_traj,speed,target)
        C_index,C_lambda = self.netC(traj, speed,seq_start_end)
        O_index,O_lambda = self.netO(traj, speed,obstacle, seq_start_end)
        pred=deepsfm_update(traj,speed,dest,obstacle,seq_start_end,D_tau,C_index,C_lambda,O_index,O_lambda)
        ground=torch.concat([traj[:,1:,:],dest],dim=1)
        loss=self.l2_loss(ground,pred)
        return loss.sum(),loss.mean()

    def compute_loss(self, traj, rel_traj,obstacle,speed,dest,target, seq_start_end):
        D_tau=self.netD(rel_traj,speed,target)
        C_index,C_lambda = self.netC(traj, speed,seq_start_end)
        O_index,O_lambda = self.netO(traj, speed,obstacle, seq_start_end)
        pred,mask=deepsfm_update_R(traj,speed,dest,obstacle,seq_start_end,D_tau,C_index,C_lambda,O_index,O_lambda)
        ground=torch.concat([traj[:,1:,:],dest],dim=1)
        loss=self.l2_loss(ground[mask],pred[mask])
        return loss.sum(),loss.mean()

    def l2_loss(self, pred_traj, pred_traj_gt, mode=''):
            loss = (pred_traj_gt - pred_traj) ** 2
            if mode == 'sum':
                return torch.sum(loss)
            elif mode == 'mean':
                return loss.sum(dim=-1).mean(dim=1)
            elif mode == 'raw':
                return loss.sum(dim=-1).sum(dim=1)
            else:
                return loss.sum(dim=-1)

    def compute_cvae_loss_NoR(self, traj, rel_traj, obstacle, speed, dest, target, seq_start_end):
        D_tau = self.netD(rel_traj, speed, target)
        C_index, C_lambda = self.netC(traj, speed, seq_start_end)
        O_index, O_lambda = self.netO(traj, speed, obstacle, seq_start_end)
        pred = deepsfm_update(traj, speed, dest, obstacle, seq_start_end, D_tau, C_index, C_lambda, O_index, O_lambda)
        ground = torch.concat([traj[:, 1:, :], dest], dim=1)
        x = ground - pred
        x_hat, mu, logvar = self.cvae(x, traj, speed,pred)
        loss, l2, kl = self.loss_function(x_hat, x, mu, logvar)
        pred_ = self.cvae.decoder(torch.randn(x.size(0), x.size(1), self.args.dim).to(x), traj, speed,pred) + pred
        sfm = self.l2_loss(ground, pred)
        loss_sfm = sfm.sum()
        sfm_mean = sfm.mean()
        cvae_recon = self.l2_loss(ground, pred_)
        loss_cvae_recon = cvae_recon.sum()
        cvae_mean = cvae_recon.mean()
        return loss, l2, kl, loss_sfm, loss_cvae_recon, sfm_mean, cvae_mean
    def compute_cvae_loss(self, traj, rel_traj,obstacle,speed,dest,target, seq_start_end):
        D_tau = self.netD(rel_traj, speed, target)
        C_index, C_lambda = self.netC(traj, speed, seq_start_end)
        O_index, O_lambda = self.netO(traj, speed, obstacle, seq_start_end)
        pred,mask = deepsfm_update_R(traj, speed, dest, obstacle, seq_start_end, D_tau, C_index, C_lambda, O_index, O_lambda)
        ground = torch.concat([traj[:, 1:, :], dest], dim=1)
        x=ground-pred
        x_hat, mu, logvar=self.cvae(x, traj,speed,pred)
        loss,l2,kl=self.loss_function(x_hat,x,mu,logvar)
        pred_=self.cvae.decoder(torch.randn(x.size(0),x.size(1),self.args.dim).to(x),traj,speed,pred)+pred
        sfm=self.l2_loss(ground, pred)
        loss_sfm = sfm.sum()
        sfm_mean=sfm.mean()
        cvae_recon=self.l2_loss(ground[mask], pred_[mask])
        loss_cvae_recon = cvae_recon.sum()
        cvae_mean=cvae_recon.mean()
        return loss,l2,kl,loss_sfm,loss_cvae_recon,sfm_mean,cvae_mean

    def loss_function(self,x_hat, x, mu, logvar):
        reconstruction_loss = self.l2_loss(x_hat, x).sum()
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        #pred_=pred+x_hat
        # col_loss=0
        # for se in seq_start_end:
        #     col_loss+=self.colloss(pred_[se[0]:se[1]])
        return reconstruction_loss + kl_divergence,reconstruction_loss,kl_divergence#,col_loss

    def colloss(self,traj,collision_threshold=0.4):
        n, length, _ = traj.shape
        agents_current = traj.unsqueeze(0)
        agents_other = traj.unsqueeze(1)
        distance = torch.norm(agents_current - agents_other, dim=-1)
        mask = (distance < collision_threshold)& ~torch.eye(n, n, dtype=torch.bool).unsqueeze(2).to(self.args.device)
        distance=distance[mask]
        loss=torch.abs(distance-0.5).sum()
        return loss
