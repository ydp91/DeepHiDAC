import math
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from utils import collision_label,check_collisions_times
from config import parse_args
from sfmloader import gen_sfm_traj




def seq_collate(data):
    traj_list = [seq[0][0] for seq in data]
    _len = [len(seq) for seq in traj_list]  # 每组的人数
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]  # 开始位,结束位
    traj = torch.cat(traj_list, dim=0).permute(0, 2, 1)
    seq_start_end = torch.LongTensor(seq_start_end)
    #############
    obstacle_dir = './datasets/env/obstacle/meter/'
    obstacle_files = np.unique(np.array([seq[1] for seq in data]) )
    obstacle_info = {}
    for file in obstacle_files:
        obstacle = np.genfromtxt(obstacle_dir + file, delimiter=',', dtype=np.float32)
        obstacle_info[file] = torch.from_numpy(obstacle).cuda()
    obstacle=[obstacle_info[seq[1]] for seq in data]
    #############

    out = [
        traj, seq_start_end,obstacle
    ]  # input format:  batch,seq_len, input_size
    # 位置,每组开始索引及结束索引

    return tuple(out)

def train_seq_collate(data):

    traj_list = [seq[0] for seq in data] #
    _len = [len(seq) for seq in traj_list]  # 每组的人数
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]  # 开始位,结束位
    seq_start_end = torch.LongTensor(seq_start_end)
    traj_sfm_list = [seq[1] for seq in data]

    traj_list = torch.cat(traj_list, dim=0)
    traj_sfm_list= torch.cat(traj_sfm_list, dim=0)

    obstacle=[seq[2] for seq in data]
    out=[traj_list,traj_sfm_list,seq_start_end,obstacle]
    return tuple(out)


def min_distance(traj):
    '''
    Calculate the lable of the collision 1为无碰撞 0为有碰撞
    '''
    for j in range(len(traj)-1):
        distance = ((traj[j:j + 1] - traj[j + 1:]) ** 2).sum(axis=1) ** 0.5
        min = distance.min()
        if min<0.4:
            return False
    return True

# 读取文件为numpy
def read_file(_path, delim=','):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    elif delim == ',':
        delim = ','
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""

    def __init__(
            self, traj_len=8,dataset='', delim='\t', type='train'
    ):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()

        self.data_dir = './datasets/'+dataset+'/' + type + '/'
        self.traj_len = traj_len
        self.delim = delim
        #self.obstacle=
        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []
        env_name=[]

        # 提取全部运动序列分组（只考虑在当前连续的所有帧中全部出现的人,且最低人数大于要考虑人数的位置）
        for path in all_files:
            data = read_file(path, delim)  # 直接读出来的文本数据分割（帧ID，人ID,Pos_x,Pos_y）
            frames = np.unique(data[:, 0]).tolist()  # 按第一列去重（得到所有帧ID）
            frame_data = []  # 所有数据按第一列分组（按帧分组后的数据list ,每项都是n*（ 帧ID,人ID,Pos_x,Pos_y)）
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(len(frames) - self.traj_len + 1 )#序列数量

            for idx in range(0, num_sequences  + 1):
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.traj_len], axis=0)  # 将该组运动的全部序列拼接
                pre_frame_data=np.zeros((1,4))#上一帧运行信息
                if idx!=0:
                    pre_frame_data = frame_data[idx - 1]

                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])  # 该组运动涉及的全部人员id
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.traj_len+1))  # 该序列全部人绝对位置,2,全长（1+8）（第一个位置用于存放上次的位置）

                num_peds_considered = 0
                for _, ped_id in enumerate(peds_in_curr_seq):  # 遍历该组运动的所有人
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==
                                                 ped_id, :]  # 当前人的全部运动状态
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx  # 当前人该组运动开始帧编号
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1  # 当前人该组运动结束帧编号
                    if pad_end - pad_front != self.traj_len:  # 如果该人在所有帧都存在才继续执行
                        continue
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])  # 只留下位置信息并转置
                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front+1:pad_end+1] = curr_ped_seq  # 真实坐标，空出第一个位置（存放上一时刻位置）
                    curr_ped_pre=pre_frame_data[pre_frame_data[:,1]==ped_id,:] #当前人在上一帧的位置 （可能不存在）
                    if len(curr_ped_pre)==1:
                        curr_seq[_idx, :, 0]=curr_ped_pre[0,2:] # #当前人在上一帧的位置赋值
                    else:
                        curr_seq[_idx, :, 0]=curr_seq[_idx, :, 1]#如果上一帧最后位置不存在，将当前帧最初位置当上一帧最后位置

                    num_peds_considered += 1

                if num_peds_considered > 0:  # 如果该组存在全部帧的人数大于要考虑的最低人数(只要有人就算，单人也参与训练)
                    num_peds_in_seq.append(num_peds_considered)
                    seq_list.append(curr_seq[:num_peds_considered])
                    env_name.append(os.path.basename(path))



        self.num_seq = len(seq_list)  # 获取到的全部运动序列分组（只考虑在当前所有帧中全部出现的人,且最低人数大于要考虑人数的位置）
        seq_list = np.concatenate(seq_list, axis=0)  # 全部运动序列

        # Convert numpy -> Torch Tensor
        self.traj = torch.from_numpy(seq_list).type(torch.float)  # 全部序列
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()  # 每组运动人员开始的索引位置
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]  # 每组人的起始索引
        self.env_name=env_name

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        out = [
            self.traj[start:end, :]
        ]
        return out,self.env_name[index]

class TrainTrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""

    def __init__(
            self, traj_len=8,dataset=''
    ):
        """
        """
        super(TrainTrajectoryDataset, self).__init__()

        self.data_dir = './datasets/' + dataset + '/' + str(traj_len) + '/'
        self.traj_len = traj_len

        self.traj=torch.load(self.data_dir+ 'real.pt' )

        self.seq_start_end = torch.load(self.data_dir + 'seq_start_end.pt')
        self.sfm = torch.load(self.data_dir + 'sfm.pt')
        self.obstacle_env = np.load(self.data_dir + 'obstaclefileName.npy')
        obstacle_dir='./datasets/env/obstacle/meter/'
        obstacle=np.unique(self.obstacle_env)
        self.obstacle_info={}
        for file in obstacle:
            obstacle=np.genfromtxt(obstacle_dir+file,delimiter=',', dtype=np.float32)
            self.obstacle_info[file]=torch.from_numpy(obstacle).cuda()

    def __len__(self):
        return self.seq_start_end.size(0)

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        return self.traj[start:end,:,:],self.sfm[start:end,:,:],self.obstacle_info[self.obstacle_env[index]]

def get_data_loader(args, type):
    if not type=='train':
        dset = TrajectoryDataset(
            traj_len=args.traj_len,
            dataset=args.dataset,
            delim=args.delim,
            type=type)

        loader = DataLoader(
            dset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=seq_collate)
        return loader
    else:
        dset=TrainTrajectoryDataset(traj_len=args.traj_len,dataset=args.dataset)
        loader = DataLoader(
            dset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=train_seq_collate)
        return loader


def rotation_2d_torch(x, theta, origin=None):
    '''旋转'''
    if origin is None:
        origin = x.reshape(-1, 2).mean(dim=0)
    norm_x = x - origin
    norm_rot_x = torch.zeros_like(x)
    norm_rot_x[..., 0] = norm_x[..., 0] * torch.cos(theta) - norm_x[..., 1] * torch.sin(theta)
    norm_rot_x[..., 1] = norm_x[..., 0] * torch.sin(theta) + norm_x[..., 1] * torch.cos(theta)
    rot_x = norm_rot_x + origin
    return rot_x  # 旋转后位置


def preprocessing(traj,obstacle, seq_start_end, type='train'):
    traj_rot = torch.zeros_like(traj).to(traj)
    obstacle_rot=[]
    if type == 'train':  # 训练数据做随机旋转
        for i,se in enumerate(seq_start_end):
            theta = torch.rand(1).to(traj) * np.pi * 2
            origin=traj[se[0]:se[1]].reshape(-1, 2).mean(dim=0)
            traj_rot[se[0]:se[1]] = rotation_2d_torch(traj[se[0]:se[1]], theta,origin)
            obstacle_rot.append(rotation_2d_torch(obstacle[i].clone(),theta,origin))
    else:
        traj_rot = traj
        obstacle_rot=obstacle

    rel_traj = traj_rot[:, 1:, ] - traj_rot[:, :-1, ]  #相对运动
    speed=rel_traj/0.4 #速度  0-(t-1)时刻的速度
    rel_traj=traj_rot- traj_rot[:, 0:1, ] #相对第一个位置的轨迹 0-t时刻


    dest = traj_rot[:,-1:,:] #目标
    target=dest-traj_rot[:,1:-1,:]
    mean = torch.norm(target, dim=-1).unsqueeze(-1)+1e-5
    target = target / mean


    return traj_rot[:,1:-1,:], rel_traj[:,1:-1,:],speed[:,:-1,:],obstacle_rot, dest,target# 绝对位置(0位为补足，最后位无法验证速度),相对位置,速度,障碍,目标,每个位置与目标的方向





def init_train_data(ds_name):
    # 构造训练集文件夹
    args = parse_args()
    dset = TrajectoryDataset(
        traj_len=args.traj_len,
        dataset=ds_name,
        delim=args.delim,
        type='train')
    len = dset.__len__()
    list = []
    _len = []  # 每组的人数
    env_name_list=[]
    for i in range(len):
        item = dset.__getitem__(i)[0][0]
        env_name_list.append(dset.__getitem__(i)[1])#障碍物环境文件名称
        list.append(dset.__getitem__(i)[0][0])
        _len.append(item.size(0))

    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]  # 开始位,结束位
    traj = torch.cat(list, dim=0).permute(0, 2, 1)
    seq_start_end = torch.LongTensor(seq_start_end)

    data_dir = './datasets/' + ds_name + '/' + str(args.traj_len) + '/'
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    envs_name = np.array(env_name_list)
    np.save(data_dir+'obstaclefileName.npy',envs_name)
    torch.save(traj, data_dir + 'real.pt')
    torch.save(seq_start_end, data_dir + 'seq_start_end.pt')
    collision = collision_label(traj, seq_start_end)
    sfm = []

    for i, se in enumerate(seq_start_end):
        if collision[se[0]:se[1]].max() < 1:
            sfm.append(traj[se[0]:se[1]])
        else:
            sfm_traj=gen_sfm_traj(traj[se[0]:se[1]],collision[se[0]:se[1]])
            sfm.append(sfm_traj)
            print('sfm handle')
            print(check_collisions_times(traj[se[0]:se[1]]))
            print(check_collisions_times(sfm_traj))
        print(i, "/", len,"penson num:",se[1]-se[0])
    sfm = torch.concat(sfm, dim=0)
    torch.save(sfm, data_dir + 'sfm.pt')





if __name__ == '__main__':
    init_train_data('raw')


