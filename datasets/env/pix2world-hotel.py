import numpy as np
import pandas as pd

# 假设您有单应矩阵 H
H = np.array([
        [1.1048200e-02, 6.6958900e-04, -3.3295300e+00],
        [-1.5966000e-03, 1.1632400e-02, -5.3951400e+00],
        [1.1190700e-04, 1.3617400e-05, 5.4276600e-01]
    ])

def pixel_to_plane_coords(pixel_pos):
    # 将像素坐标扩展为齐次坐标
    pixel_pos_homogeneous = np.array([pixel_pos[:, 0], pixel_pos[:, 1], np.ones(pixel_pos.shape[0])])

    # 使用单应矩阵 H 将像素坐标映射到平面坐标
    plane_coords = np.dot(H, pixel_pos_homogeneous)
    plane_coords = plane_coords[:2] / plane_coords[2:]
    return plane_coords.T


def csv_to_numpy(csv_file):
    """
    从CSV文件中加载数据并返回一个NumPy数组。

    参数：
    csv_file (str) - CSV文件的路径

    返回：
    numpy_array (numpy.ndarray) - 包含CSV文件数据的NumPy数组
    """
    df = pd.read_csv(csv_file)
    numpy_array = df.values
    return numpy_array
if __name__ == '__main__':
    # 假设您有像素坐标
    pixel_pos = csv_to_numpy('obstacle/pixel/hotel.csv')[:, 1:]  # 示例像素坐标
    temp=pixel_pos[:,0].copy()
    pixel_pos[:,0]=pixel_pos[:,1]
    pixel_pos[:,1] = temp
    # 将像素坐标转换为平面坐标
    plane_coord = np.round(pixel_to_plane_coords(pixel_pos),3)


    print("平面坐标 (x, y):", plane_coord)
    np.savetxt('obstacle/meter/hotel_obstacle.csv', plane_coord, delimiter=',')
