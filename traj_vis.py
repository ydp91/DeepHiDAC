import matplotlib.pyplot as plt
import torch

def plot_trajectories(trajectory_data):
    """
    绘制人群轨迹图

    参数:
    trajectory_data (torch.Tensor): 人群轨迹数据，形状为(agent_num, traj_len, 2)，其中agent_num表示代理数量，
                                     traj_len表示轨迹点的数量，2表示每个点有两个坐标（例如，x和y坐标）。
    """
    agent_num, traj_len, _ = trajectory_data.size()

    # 转换为NumPy数组以在matplotlib中使用
    trajectory_data = trajectory_data.numpy()

    # 创建一个绘图对象
    plt.figure(figsize=(8, 6))

    # 遍历每个代理的轨迹数据并绘制
    for agent_idx in range(agent_num):
        traj = trajectory_data[agent_idx]
        x = traj[:, 0]  # x坐标
        y = traj[:, 1]  # y坐标
        plt.plot(x, y, label=f'Agent {agent_idx + 1}')

    # 添加标题和标签
    plt.title('Agent Trajectories')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    # 添加图例
    plt.legend()

    # 显示图形
    plt.show()

