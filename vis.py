import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image
import os

def create_animation_from_data_file(ds,ext='_R'):

    file_directory = 'datasets/env/resources/'

    file_name=ds+ext+'.txt'
    obstacle=np.genfromtxt('datasets/env/obstacle/meter/'+ds+'.txt', delimiter=',')

    # 从文件中加载您的数据
    data = np.genfromtxt(file_directory+file_name, delimiter=',')
    data[:,0]-=int(data[:, 0].min())
    # 获取数据的帧数和人数
    n_frames = int(data[:, 0].max())-int(data[:, 0].min())+1
    n = int(data[:, 1].max())
    # 设置场地的边界
    x_min, x_max = data[:, 2].min() - 1, data[:, 2].max() + 1  # 场地的 x 范围
    y_min, y_max = data[:, 3].min() - 1, data[:, 3].max() + 1  # 场地的 y 范围

    def interpolate_position(frame):
        prev_frame = frame//10*10
        next_frame = frame//10*10+10
        x,y,agentid=[],[],[]
        for agent_id in range(1, n + 1):
            prev_data = data[(data[:, 1] == agent_id) & (data[:, 0] == prev_frame)]
            next_data = data[(data[:, 1] == agent_id) & (data[:, 0] == next_frame)]
            if len(prev_data) == 0:
                continue
            if len(next_data) == 0 and (frame - prev_frame)==0:
                x.append(prev_data[0, 2])
                y.append(prev_data[0, 3])
                agentid.append(agent_id)
                continue
            if len(next_data) == 0:
                continue
            t = (frame - prev_frame) / (next_frame - prev_frame)
            x.append(prev_data[0, 2] + t * (next_data[0, 2]-prev_data[0, 2]))
            y.append(prev_data[0, 3] + t * (next_data[0, 3]-prev_data[0, 3]))
            agentid.append(agent_id)
        return x, y,agentid

    def update(frame):
        plt.clf()
        ax = plt.subplot()
        ax.grid(linestyle='dotted')
        ax.set_aspect(1.0, 'datalim')
        ax.set_axisbelow(True)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        x,y,id = interpolate_position(frame)
        title_text='Frame '+ str(frame)+' / '+str(frame/25.)+'s'
        plt.title(title_text)
        plt.scatter(x, y, c='b', marker='o',s=20)
        plt.scatter(obstacle[:,0], obstacle[:,1], c='r', marker='o',s=10)
        if frame%10==0:
            print('frame finish:',frame)
        return [plt]

    # 创建动画
    fig, ax = plt.subplots(figsize=(12, 12))
    ani = FuncAnimation(fig, update, frames=range(0, n_frames), blit=False)

    # 保存为 GIF 文件
    ani.save(file_directory+'/'+file_name+'.gif', writer='pillow', fps=25)  # 25 帧每秒

if __name__ == '__main__':
    create_animation_from_data_file('eth', '_SFM1')

