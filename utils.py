import os
import csv
import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
import cv2
from skimage.metrics import structural_similarity

def load_agent_list_by_file(file):
    # 打开CSV文件
    with open('./resources/result/'+file, newline='') as csvfile:
        # 创建CSV读取器
        reader = csv.reader(csvfile, delimiter=',')
        # 跳过第一行
        next(reader)
        # 创建一个空列表来存储结果
        result = []
        # 遍历每一行并将其转换为Numpy数组
        for row in reader:
            # 跳过第一列
            data = np.array(row[1:], dtype=float)/100
            # 将数据添加到结果列表中
            data=np.reshape(data, (-1,2))
            result.append(data)
        # 将结果转换为Numpy数组
        result = np.array(result)

    # 打印结果
    return result


def saveMoveInfo(g_out, x_max, x_min, y_max, y_min, args, epoch, type):
    """
    :param g_out:[person_count,seq_len,2]
    :param args:
    :return:
    """
    x_max = x_max.cpu()
    x_min = x_min.cpu()
    y_max = y_max.cpu()
    y_min = y_min.cpu()
    timestamp = args.timestamp
    output_dir = "%s%s" % (args.out_dir, timestamp)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    g_out = g_out.view(-1, g_out.size(0), g_out.size(1), g_out.size(2))
    for i in range(g_out.size(0)):
        move = g_out[i].detach().clone().cpu()
        move[:, :, 0] = move[:, :, 0] * (x_max - x_min) + x_min
        move[:, :, 1] = move[:, :, 1] * (y_max - y_min) + y_min
        move = move.view(move.size(0), -1)
        csv = pd.DataFrame(move.numpy())
        csv.to_csv("%s/%s_%s_%s.csv" % (output_dir, type, epoch, i))


def getRealPos(base_pos, traj_rel):
    """
    :param traj_rel:[person_count,seq_len,2]
    :return:
    """
    displacement = torch.cumsum(traj_rel, dim=1)  # 预测轨迹累加
    return displacement + base_pos

def getRealPosBySpeed(base_pos, speed):
    """
    :param traj_rel:[person_count,seq_len,2]
    :return:
    """
    return speed + base_pos

# def collision_label(traj,seq_start_end):
#     '''
#     Calculate the lable of the collision 0为无碰撞 1为有碰撞
#     '''
#     col_lab=[]
#     for se in seq_start_end:
#         current_group=traj[se[0]:se[1]]
#         n_agents, n_positions, n_dims = current_group.size()
#         for i in range(n_agents):
#             distance= torch.sum( (current_group[i:i+1]-current_group)**2,dim=-1)**0.5
#             distance[i]=1 #与自身的距离设置为1，为了后续设置自身不碰撞
#             col_lab.append((torch.min(distance,dim=0)[0]<0.4).float())
#     col_lab=torch.stack(col_lab,dim=0).unsqueeze(dim=-1)
#     return col_lab



def collision_label(traj,seq_start_end):
    '''
    计算除最后一个位置（作为方向）是否产生了碰撞
    Calculate the lable of the collision 0为无碰撞 1为有碰撞
    '''

    col_lab=[]
    for se in seq_start_end:
        current_group=traj[se[0]:se[1]]
        n_agents = current_group.size(0)
        for i in range(n_agents):
            agent_i_pos=current_group[i:i+1,0:-1,:] #当前agent
            agent_other_pos=current_group[:,0:-1,:] #其他agent计算前时刻速度的下一个位置
            distance= torch.sum( (agent_i_pos-agent_other_pos)**2,dim=-1)**0.5
            distance[i]=1 #与自身的距离设置为1，(设置自身不碰撞)
            col_lab.append((torch.min(distance,dim=0)[0]<0.4).float())
    col_lab=torch.stack(col_lab,dim=0).unsqueeze(dim=-1)
    return col_lab


def show(move):
    move = move.permute(1, 0, 2)
    for i in range(move.size(0)):
        plt.scatter(move[i, :, 0], move[i, :, 1])
        plt.show()


def saveModel(G, D,C, args,epoch=''):
    timestamp = args.timestamp
    output_dir = "%s%s" % (args.model_dir, timestamp)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    torch.save(G.state_dict(), "%s/D_%s_%s.pth" % (output_dir,args.dataset,epoch))
    torch.save(D.state_dict(), "%s/C_%s_%s.pth" % (output_dir,args.dataset,epoch))
    torch.save(C.state_dict(), "%s/O_%s_%s.pth" % (output_dir, args.dataset, epoch))
def saveModel_VAE(VAE, args,epoch=''):
    timestamp = args.timestamp
    output_dir = "%s%s" % (args.model_dir, timestamp)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    torch.save(VAE.state_dict(), "%s/VAE_%s_%s.pth" % (output_dir,args.dataset,epoch))


def loadModel(G, D, args):
    G.load_state_dict(args.G_dict)
    D.load_state_dict(args.D_dict)


def cal_ade(real, fake, mode="mean"):
    loss = real - fake
    loss = loss ** 2
    loss = torch.sqrt(loss.sum(dim=2)).mean(dim=1)
    if mode == "mean":
        return torch.mean(loss)
    elif mode == "raw":
        return loss

def cal_speed_num_error(real, fake, mode="mean"):
    loss = real - fake
    loss = loss ** 2
    loss = torch.sqrt(loss.sum(dim=2)).mean(dim=1)
    if mode == "mean":
        return torch.mean(loss)
    elif mode == "raw":
        return loss

def cal_fde(real, fake, mode="mean"):
    loss = real[:, -1, :] - fake[:, -1, :]
    loss = loss ** 2
    loss = torch.sqrt(loss.sum(dim=1))
    if mode == "raw":
        return loss
    else:
        return torch.mean(loss)





def writecsv(path,info):
    with open(path, 'a+') as f:
        f.write(','+ str(info))



#计算Agent是否与其他Agent发生碰撞
def check_collisions(agent_positions, collision_threshold=0.4):
    '''
    碰撞检测
    :param agent_positions:
    :param collision_threshold:
    :return:
    '''
    n = agent_positions.size(0)
    collisions = [False] * n  # 初始化一个包含 n 个 False 的列表，表示每个代理初始时没有发生碰撞

    for i in range(n):
        for j in range(n):
            if i != j:
                # 计算代理 i 和代理 j 之间的距离
                distance = torch.norm(agent_positions[i] - agent_positions[j])

                # 如果距离小于碰撞阈值，认为发生碰撞
                if distance < collision_threshold:
                    collisions[i] = True
                    break  # 如果代理 i 已经碰撞，跳出内层循环

    return collisions

def check_collisions_times(agent_positions, collision_threshold=0.4):
    '''
    碰撞检测
    :param agent_positions:
    :param collision_threshold:
    :return:
    '''
    n = agent_positions.size(0)
    agents_current=agent_positions.unsqueeze(0)
    agents_other = agent_positions.unsqueeze(1)
    distance=torch.norm(agents_current-agents_other,dim=-1)
    count=(distance<collision_threshold).sum()
    count=(count-n*agent_positions.size(1))/2 #去掉与自身碰撞，再去掉互相碰撞的重复计算
    return count

def check_collisions_obstance_time(agent_positions,obstacle, collision_threshold=0.3):
    n, len, _ = agent_positions.shape
    _,num_obstacles, _ = obstacle.shape
    # 将人的位置和障碍物位置扩展为相同形状以进行广播
    people_positions_expanded = agent_positions.unsqueeze(2).expand(n, len, num_obstacles, 2)
    # 计算人和障碍物之间的距离
    vector = obstacle-people_positions_expanded
    distances = torch.norm(vector, dim=3)
    count = (distances < collision_threshold).sum()
    return count


#计算Agent是否与墙发生碰撞
def check_collisions_obstance(agent_positions,obstacle, collision_threshold=0.3):
    '''
    碰撞检测
    :param agent_positions:
    :param collision_threshold:
    :return:
    '''
    n = agent_positions.size(0)
    m=0
    if obstacle is not None:
        m = obstacle.size(0)
    collisions = [False] * n  # 初始化一个包含 n 个 False 的列表，表示每个代理初始时没有发生碰撞

    for i in range(n):
        for j in range(m):
            # 计算代理 i 和代理 j 之间的距离
            distance = torch.norm(agent_positions[i] - obstacle[j])

            # 如果距离小于碰撞阈值，认为发生碰撞
            if distance <= collision_threshold:
                collisions[i] = True
                break  # 如果代理 i 已经碰撞，跳出内层循环

    return collisions




def mse(imageA, imageB):
    # 计算两张图片的MSE相似度
    # 注意：两张图片必须具有相同的维度，因为是基于图像中的对应像素操作的
    # 对应像素相减并将结果累加起来
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    # 进行误差归一化
    err /= float(imageA.shape[0] * imageA.shape[1])

    # 返回结果，该值越小越好，越小说明两张图像越相似
    return err

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

def getDatasetPersonInfo(dataset, delim=','):
    '''
    获取数据集中轨迹情况
    :param dataset:
    :param delim:
    :return:
    '''
    all_files = ['./datasets/raw/val/' +dataset + '.txt']
    # all_files = os.listdir(data_dir)
    # all_files = [os.path.join(data_dir, _path) for _path in all_files]

    info=[] #结果数据 （开始帧集合，结束帧集合，初始位置集合，目标位置集合）
    for path in all_files:
        agent_data = []  # 所有数据按第一列分组（按帧分组后的数据list ,每项都是n*（ 帧ID,人ID,Pos_x,Pos_y)）
        data = read_file(path, delim)  # 直接读出来的文本数据分割（帧ID，人ID,Pos_x,Pos_y）
        agents = np.unique(data[:, 1]).tolist()  # 所有人的ID
        for agent in agents:
            agent_data.append(data[agent == data[:, 1], :])

        id,start,end,init,last,pos=[],[],[],[],[],[]#ID，开始帧，结束帧，初始位置，目标位置，轨迹序列
        for agent in agent_data:
            if len(agent)>2:
                if len(agent)==(agent[:, 0].max() / 10)-(agent[:,0].min()/10)+1:

                    id.append(agent[0][1])
                    start.append(agent[:,0].min()/10)
                    end.append(agent[:, 0].max() / 10)
                    init.append(agent[:2, 2:])
                    last.append(agent[-1, 2:])
                    pos.append(agent[:, 2:])

        info.append([id,start,end,init,last,pos])
    return info

def image_similarity(img1, img2):
    """
    :param img1: 图片1
    :param img2: 图片2
    :return: 返回图片1 2结构相似度
    """
    # load image
    image1 = cv2.imdecode(np.fromfile(img1, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    image2 = cv2.imdecode(np.fromfile(img2, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    if h1 != h2 or w1 != w2:
        image2 = cv2.resize(image2, (w1, h1))
    # convert the images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    # compute the Structural Similarity Index (SSIM) between the two
    # images, ensuring that the difference image is returned
    (score, diff) = structural_similarity(gray1, gray2, full=True)
    # diff = (diff * 255).astype("uint8")
    return score


if __name__ == '__main__':
    # real=cv2.imread('resources/eth/hot/real.png')
    # real=cv2.cvtColor(real, cv2.COLOR_BGR2GRAY)
    # fake = cv2.imread('resources/eth/hot/crowdgan-hot.png')
    # fake = cv2.cvtColor(fake, cv2.COLOR_BGR2GRAY)
    print(image_similarity('resources/eth/hot/real.png','resources/eth/hot/our-hot.png'))