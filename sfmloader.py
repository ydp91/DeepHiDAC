import sfm_build_train

import torch
import numpy as np



def gen_sfm_traj(traj, col_info):
    # Initialize the SFM parameters
    col_info=col_info.max(dim=0)[0].squeeze()
    sfm_tarj=traj.clone()


    for i,col in enumerate(col_info):
        if col==1 and i>0:
            velocities=(traj[:, i, :]-traj[:,i-1,:])/0.4
            positions, velocities = sfm_build_train.social_force_update(traj[:,i-1,:], velocities, traj[:,i,:], time_step=0.4,desired_speed=torch.norm(velocities,dim=-1).unsqueeze(-1))
            sfm_tarj[:,i,:] = positions
    return sfm_tarj



