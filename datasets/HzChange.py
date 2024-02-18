import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

# 读取数据
file_path = 'univ/2.5hz/univ.txt'  # 使用您上传的文件路径
data_df = pd.read_csv(file_path, header=None, names=['frameID', 'AgentID', 'X', 'Y'])

# 分组数据以便对每个AgentID进行插值
interpolated_data = []
for agent_id in data_df['AgentID'].unique():
    agent_data = data_df[data_df['AgentID'] == agent_id]

    # 创建插值函数
    interp_x = interp1d(agent_data['frameID'], agent_data['X'], kind='linear', fill_value="extrapolate")
    interp_y = interp1d(agent_data['frameID'], agent_data['Y'], kind='linear', fill_value="extrapolate")

    # 插值到新的frameID
    start_frame = agent_data['frameID'].min()
    end_frame = agent_data['frameID'].max()
    new_frameIDs = np.arange(start_frame, end_frame + 5, 5)
    new_xs = interp_x(new_frameIDs)
    new_ys = interp_y(new_frameIDs)

    # 构建新的DataFrame
    agent_interpolated = pd.DataFrame({
        'frameID': new_frameIDs,
        'AgentID': agent_id,
        'X': new_xs,
        'Y': new_ys
    })
    interpolated_data.append(agent_interpolated)

# 合并所有插值后的数据
interpolated_data_df = pd.concat(interpolated_data)

# 根据frameID和AgentID对数据进行排序
interpolated_data_sorted = interpolated_data_df.sort_values(by=['frameID', 'AgentID'])

# 保存排序后的插值数据到文件
output_file_path = 'univ/5hz/univ.txt'
interpolated_data_sorted.to_csv(output_file_path, index=False)
