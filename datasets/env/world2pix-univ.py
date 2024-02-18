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
    data = read_file('../univ/12.5hz/univ.txt')#存的是世界坐标
    H = np.array([
        [0.02104651 , 0, 0],
        [0, -0.02386598, 13.74680446],
        [0, 0, 1]
    ])
    H_inv = np.linalg.inv(H) #逆矩阵
    frames = np.unique(data[:, 0]).tolist()  # 总帧数
    frame_data = []
    pixel_poses = []
    for frame in frames:
        frame_data.append(data[frame == data[:, 0], :])  # 重组gt, frame_data列表的每一个元素代表当前帧下所有Person的出现情况
    for frame_data_ in frame_data:
        world_pos = np.vstack((frame_data_[:, 2], frame_data_[:, 3]))#x,y世界坐标
        world_pos = np.vstack((world_pos, np.ones((world_pos.shape[1]))))#x,y,1世界坐标
        pixel_pos = np.dot(H_inv, world_pos) #像素坐标
        pixel_pos_ = pixel_pos[:2, :] / pixel_pos[2:, :] #在除以由1得到的z
        pixel_poses.append(pixel_pos_)

    video = cv2.VideoCapture('students003.avi')
    k = 0
    index = 0
    while True:
        _, frame = video.read()
        if frame is None:
            break
        img = frame.copy()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if k == frames[index]:
            positions = pixel_poses[index]
            positions = np.transpose(positions)
            for p in positions:
                cx, cy = np.int32(np.round(p))
                cv2.rectangle(img, (cx - 10, cy - 20), (cx + 10, cy + 20), (255, 255, 255), thickness=2)
            index = index + 1
            #cv2.imwrite('{}.jpg'.format(k), img)
        cv2.imshow('video', img)
        cv2.waitKey(10)
        k = k + 1