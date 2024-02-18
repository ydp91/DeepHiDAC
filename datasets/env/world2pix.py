import numpy as np
import cv2


def read_file(_path, delim=','):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    elif delim == ',':
        delim = ','
    #delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [i for i in line if not i == '']
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


if __name__ == '__main__':
    data = read_file('ethvis/eth_NoR.txt')#存的是世界坐标
    H = np.array([
        [2.8128700e-02, 2.0091900e-03, -4.6693600e+00],
        [8.0625700e-04, 2.5195500e-02, -5.0608800e+00],
        [3.4555400e-04, 9.2512200e-05, 4.6255300e-01]
    ]) #eth
    # H = np.array([
    #     [1.1048200e-02, 6.6958900e-04, -3.3295300e+00],
    #     [-1.5966000e-03, 1.1632400e-02, -5.3951400e+00],
    #     [1.1190700e-04, 1.3617400e-05, 5.4276600e-01]
    # ]) #hotel
    # H = np.array([
    #     [0.02104651, 0, 0],
    #     [0, -0.02386598, 13.74680446],
    #     [0, 0, 1]
    # ])

    H_inv = np.linalg.inv(H) #逆矩阵
    world_pos = np.vstack((data[:, 2], data[:, 3]))  # x,y世界坐标
    world_pos = np.vstack((world_pos, np.ones((world_pos.shape[1]))))  # x,y,1世界坐标
    pixel_pos = np.dot(H_inv, world_pos)  # 像素坐标
    pixel_pos_ = pixel_pos[:2, :] / pixel_pos[2:, :]  # 在除以由1得到的z
    positions = np.transpose(pixel_pos_)
    positions = np.int32(np.round(positions))
    data[:,2]=positions[:,1]
    data[:, 3] = positions[:, 0]
    np.savetxt('ethvis/eth_meter_NoR.txt', data, delimiter=',')

