import numpy as np
import torch
from utils import check_collisions,check_collisions_obstance,getDatasetPersonInfo

def social_force_update(locations, velocities, target_locations,obstacle=None,tau=0.5, time_step=0.4 ,
                        lambda_1=0.35,lambda_2=0.4, sensing=3, lambda_3=0.15,lambda_4=0.45,
                        desired_speed=1.3,angle_threshold=0.8333*3.1415926):
    """
    SFM
    Args:
    locations (torch.Tensor):形状为 [n, 2] 的张量，表示代理的当前位置（米）.
    velocities (torch.Tensor): 形状为 [n, 2] 的张量，表示代理的当前速度 (m/s).
    target_locations (torch.Tensor): 形状为 [n, 2] 的张量，表示代理的目标位置（米）.
    obstacle:形状为 [m, 2] 的张量，表示为障碍物的位置(规划为圆)
    time_step (float, optional): 更新位置和速度的时间步长（秒）。 默认值为 0.4.
    tau:𝜏目标吸引力系数
    lambda_1 (float, optional): 代理之间的排斥力强度（牛顿）。 默认值为 200.0.
    lambda_2 (float, optional): 感受到排斥力的调整参数。 默认值为 0.08.
    sensing (float, optional): 感受到排斥力的距离（米）。 默认值为 3.
    lambda_ (float, optional): 控制智能体回避力的调整参数。 默认值为 0.35.

    desired_speed (float, optional): 代理的所需速度（米/秒）。 默认值为 1.3.
    angle_threshold:视野角
    Returns:
    updated_locations (torch.Tensor): 形状为 [n, 2] 的张量，表示代理的更新位置（米）.
    updated_velocities (torch.Tensor): 形状为 [n, 2] 的张量，表示代理的更新速度 (m/s).
    """

    # Get the number of agents

    #一、目标吸引力计算#######
    #1.计算朝向
    desired_directions = (target_locations - locations) / (torch.norm(target_locations - locations, dim=-1, keepdim=True) + 1e-9) #目标方向
    #2. 计算期望速度
    desired_velocities = desired_speed * desired_directions #期望速度
    #3.计算速度差异
    dv = desired_velocities - velocities #期望速度与实际速度的差异
    #4.缩放
    dv=dv/tau
    #####################


    #二、计算Agent间力#################################
    social_force_agent = torch.zeros_like(locations)
    n = locations.size(0)
    # 计算碰撞情况
    for i in range(n):
        for j in range(n):
            if i != j:
                # 1.计算距离
                diff_loc = locations[i] - locations[j]
                distance = torch.norm(diff_loc)
                if distance <= sensing:
                    # 2.归一化(方向)
                    normalized_diff_loc = diff_loc / (distance + 1e-9) #朝向
                    dot_product = torch.dot(velocities[i], normalized_diff_loc)
                    cosine_similarity = dot_product / (torch.norm(velocities[i]) * torch.norm(normalized_diff_loc))
                    angle_between = torch.acos(torch.clamp(cosine_similarity, -1.0, 1.0))
                    # 检查夹角是否在限制范围内
                    if angle_between <= angle_threshold / 2:
                        # 3.计算社会力
                        social_force_ij = lambda_1 * torch.exp(-distance / lambda_2) * normalized_diff_loc
                        # 4.求和
                        social_force_agent[i] += social_force_ij
    ################################################

    # 三、计算障碍物力#################################
    social_force_obstance = torch.zeros_like(locations)
    n = locations.size(0)
    m = 0
    if obstacle is not None:
        m = obstacle.size(0)
    for i in range(n):
        for j in range(m):
            # 1.计算距离
            diff_loc = locations[i] - obstacle[j]
            distance = torch.norm(diff_loc)
            if distance <= sensing:
                # 2.归一化(方向)
                normalized_diff_loc = diff_loc / (distance + 1e-9)
                dot_product = torch.dot(velocities[i], normalized_diff_loc)
                cosine_similarity = dot_product / (torch.norm(velocities[i]) * torch.norm(normalized_diff_loc))
                angle_between = torch.acos(torch.clamp(cosine_similarity, -1.0, 1.0))
                # 检查夹角是否在限制范围内
                if angle_between <= angle_threshold / 2:
                    # 3.计算社会力
                    social_force_ij = lambda_3 * torch.exp(-distance / lambda_4) * normalized_diff_loc
                    # 4.求和
                    social_force_obstance[i] += social_force_ij
    ################################################
    force = (dv + social_force_agent+social_force_obstance)

    lengths = torch.sqrt(torch.sum(force ** 2, dim=1))
    mask = lengths > 1.5
    # 缩放这些向量
    force[mask] = force[mask] / lengths[mask].unsqueeze(1) * 1.5  # 防止Q弹，缩放向量
    # Update the velocities by adding the difference in velocities and social force, scaled by the time step
    updated_velocities = velocities + time_step * force

    # Update the locations by adding the updated velocities, scaled by the time step
    updated_locations = locations + time_step * updated_velocities

    return updated_locations, updated_velocities



def R_update(locations, velocities, target_locations,obstacle=None,tau=1, time_step=0.4 ,
             lambda_1=100.0,lambda_2=0.15, sensing=3, lambda_3=100.0,lambda_4=0.08, desired_speed=1.3,angle_threshold=0.8333*3.1415926):
    """
    SFM
    Args:
    locations (torch.Tensor):形状为 [n, 2] 的张量，表示代理的当前位置（米）.
    velocities (torch.Tensor): 形状为 [n, 2] 的张量，表示代理的当前速度 (m/s).
    target_locations (torch.Tensor): 形状为 [n, 2] 的张量，表示代理的目标位置（米）.
    obstacle:形状为 [m, 2] 的张量，表示为障碍物的位置(规划为圆)
    time_step (float, optional): 更新位置和速度的时间步长（秒）。 默认值为 0.4.
    tau:𝜏目标吸引力系数
    lambda_1 (float, optional): 代理之间的排斥力强度（牛顿）。 默认值为 200.0.
    lambda_2 (float, optional): 感受到排斥力的调整参数。 默认值为 0.08.
    sensing (float, optional): 感受到排斥力的距离（米）。 默认值为 3.
    lambda_ (float, optional): 控制智能体回避力的调整参数。 默认值为 0.35.

    desired_speed (float, optional): 代理的所需速度（米/秒）。 默认值为 1.3.
    angle_threshold:视野角
    Returns:
    updated_locations (torch.Tensor): 形状为 [n, 2] 的张量，表示代理的更新位置（米）.
    updated_velocities (torch.Tensor): 形状为 [n, 2] 的张量，表示代理的更新速度 (m/s).
    """

    # Get the number of agents

    #一、目标吸引力计算#######
    #1.计算朝向
    desired_directions = (target_locations - locations) / (torch.norm(target_locations - locations, dim=-1, keepdim=True) + 1e-9) #目标方向
    #2. 计算期望速度
    desired_velocities = desired_speed * desired_directions #期望速度
    #3.计算速度差异
    dv = desired_velocities - velocities #期望速度与实际速度的差异
    #4.缩放
    dv=dv/tau
    #####################

    col_agent=check_collisions(locations)
    check_collisions_obstance(locations,obstacle)

    # 三、计算障碍物力#################################
    social_force_obstance = torch.zeros_like(locations)
    n = locations.size(0)
    m = 0
    if obstacle is not None:
        m = obstacle.size(0)
    for i in range(n):
        for j in range(m):
            if i != j:
                # 1.计算距离
                diff_loc = locations[i] - obstacle[j]
                distance = torch.norm(diff_loc)
                if distance <= sensing:
                    # 2.归一化(方向)
                    normalized_diff_loc = diff_loc / (distance + 1e-9)
                    dot_product = torch.dot(velocities[i], normalized_diff_loc)
                    cosine_similarity = dot_product / (torch.norm(velocities[i]) * torch.norm(normalized_diff_loc))
                    angle_between = torch.acos(torch.clamp(cosine_similarity, -1.0, 1.0))
                    # 检查夹角是否在限制范围内
                    if angle_between <= angle_threshold / 2:
                        # 3.计算社会力或R_i
                        if col_agent[i]==True or check_collisions_obstance==True:
                            social_force_ij = diff_loc*(0.3+0.1-distance)/distance
                        else:
                            social_force_ij = lambda_3 * torch.exp(-distance / lambda_4) * normalized_diff_loc
                        # 4.求和
                        social_force_obstance[i] += social_force_ij
    ################################################

    #二、计算Agent间力#################################
    social_force_agent = torch.zeros_like(locations)
    n = locations.size(0)
    # 计算碰撞情况
    for i in range(n):
        for j in range(n):
            if i != j:
                # 1.计算距离
                diff_loc = locations[i] - locations[j]
                distance = torch.norm(diff_loc)
                if distance <= sensing:
                    # 2.归一化(方向)
                    normalized_diff_loc = diff_loc / (distance + 1e-9)
                    dot_product = torch.dot(velocities[i], normalized_diff_loc)
                    cosine_similarity = dot_product / (torch.norm(velocities[i]) * torch.norm(normalized_diff_loc))
                    angle_between = torch.acos(torch.clamp(cosine_similarity, -1.0, 1.0))
                    # 检查夹角是否在限制范围内
                    if angle_between <= angle_threshold / 2:
                        # 3.计算社会力或R_i
                        if col_agent[i] == True or check_collisions_obstance == True:
                            social_force_ij = diff_loc * (0.4 + 0.2 - distance) / distance
                        else:
                            social_force_ij = lambda_1 * torch.exp(-distance / lambda_2) * normalized_diff_loc
                        # 4.求和
                        social_force_agent[i] += social_force_ij
    ################################################

    # 如果有R_obstacle 则R_i设置缩放为0.3
    # 如果发生碰撞不用吸引力
    for i in range(locations.size(0)):
        if col_agent[i] == True or check_collisions_obstance == True:
            dv[i] = torch.tensor(0)
            if torch.norm(social_force_obstance[i])!=0:
                social_force_agent[i]*=0.3

    force=dv + social_force_agent+social_force_obstance

    lengths = torch.sqrt(torch.sum(force**2, dim=1))
    mask = lengths > 1.5
    # 缩放这些向量
    force[mask] = force[mask] / lengths[mask].unsqueeze(1) * 1.5#防止Q弹，缩放向量
    # Update the velocities by adding the difference in velocities and social force, scaled by the time step
    updated_velocities = velocities + time_step * force

    # Update the locations by adding the updated velocities, scaled by the time step
    updated_locations = locations + time_step * updated_velocities

    return updated_locations, updated_velocities

def find_agent_collision(people_positions, sensing=0.4):
    n, len, _ = people_positions.shape
    # 将人的位置进行广播
    position_data_expanded = people_positions.unsqueeze(1).expand(n, n, len, 2)
    vector = people_positions.unsqueeze(0) - position_data_expanded
    distances = torch.norm(vector, dim=3)
    # 创建一个布尔掩码，指示距离小于0.4米的情况
    mask = (distances < sensing)
    index = mask.nonzero(as_tuple=False)
    # 过滤掉第0位和第1位相同的索引
    index = index[index[:, 0] != index[:, 1]]
    return index,vector,distances  # 索引,全部差值,距离
def find_obstacles_collision(people_positions,  obstacle_positions,sensing=0.5):
    n, len, _ = people_positions.shape
    num_obstacles, _ = obstacle_positions.shape
    # 将人的位置和障碍物位置扩展为相同形状以进行广播
    people_positions_expanded = people_positions.unsqueeze(2).expand(n, len, num_obstacles, 2)
    # 计算人和障碍物之间的距离
    vector = obstacle_positions-people_positions_expanded
    distances = torch.norm(vector, dim=3)
    # 创建一个布尔掩码，指示距离小于2米的情况
    mask = (distances <= sensing)
    # 返回距离小于2米的障碍物的索引
    index = mask.nonzero(as_tuple=False)
    return index,vector,distances #索引,全部差值,距离
def R_force(locations,obstacle,seq_start_end,lambda_a=0.3):
    r_force_c=torch.zeros_like(locations)
    for se in seq_start_end:
        indexs,vector,distances=find_agent_collision(locations[se[0]:se[1]])
        if indexs.size(0)>0:
            force_c= -vector*(0.4 + 0.2 - distances.unsqueeze(-1)) / distances.unsqueeze(-1)
            nan_mask = torch.isnan(force_c)
            force_c[nan_mask] = 0
            agent_len_index=indexs[:,[0,2]] #Agent在哪个位置发生碰撞的索引
            agent_len_index=torch.unique(agent_len_index, dim=0)
            for index in agent_len_index:
                mask=indexs[(indexs[:,0]==index[0])&(indexs[:,2]==index[1])]
                agent_force_c=force_c[mask[:, 0], mask[:, 1], mask[:, 2]]
                r_force_c[index[0]][index[1]]=agent_force_c.mean(dim=0)


    r_force_o = torch.zeros_like(locations)
    for i,se in enumerate(seq_start_end):
        indexs, vector, distances = find_obstacles_collision(locations[se[0]:se[1]],obstacle[i])
        if indexs.size(0) > 0:
            force_o = -vector * (0.3 + 0.3 - distances.unsqueeze(-1)) / distances.unsqueeze(-1)
            nan_mask = torch.isnan(force_o)
            force_o[nan_mask] = 0
            agent_len_index=indexs[:,[0,1]] #Agent在哪个位置发生碰撞的索引
            agent_len_index=torch.unique(agent_len_index, dim=0)
            for index in agent_len_index:
                mask = indexs[(indexs[:, 0] == index[0]) & (indexs[:, 1] == index[1])]
                agent_force_o = force_o[mask[:, 0], mask[:, 1], mask[:, 2]]
                r_force_o[index[0]][index[1]] = agent_force_o.mean(dim=0)
    o_mask=torch.norm(r_force_o,dim=-1)>0
    r_force_c[o_mask]*=lambda_a
    r=(r_force_c+r_force_o)
    return r






# 示例代码保持不变

def get_collision_times(locations, collision_threshold=0.4):
    n = locations.size(0)
    collision_times = 0
    for i in range(n):
        for j in range(i + 1, n):
            if (torch.norm(locations[i] - locations[j]) < collision_threshold):
                collision_times += 1
    return collision_times


def social_force_collision(dsname,dspeed=2.2):
    # 数据集时间步长为0.4
    time_step = 0.4
    dataset_collision_times = []
    dataset_sfm_trajs = []
    datainfo = getDatasetPersonInfo(dsname)
    # print(len(datainfo))
    for info in datainfo:
        # region DATA VISULIZATION
        # Create the figure
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        # plt.ion()
        # endregion

        # id，开始帧，结束帧，开始位置，目标位
        id, start, end, init, dest,pos = info
        # print(init)

        # id和轨迹的键值对
        traj_dict = {}
        # id和速度的键值对
        velocities_dict = {}
        for i in range(len(id)):
            traj_dict[id[i]] = init[i]
            velocities_dict[id[i]] = (init[i][1] - init[i][0]) / time_step
        # print(velocities_dict)

        # 初始化碰撞次数为0
        collision_times = 0
        # 以info中最小的开始帧初始化当前帧
        first_frame = min(start)
        cur_frame = first_frame
        # info中的最大帧
        last_frame = max(end)
        # 当前帧小于最大帧时，进行计算
        while cur_frame < last_frame - 1:
            cur_frame += 1
            curt_ids = []  # 当前帧存在个体的的ID
            cur_trajs = []  # 当前帧存在的个体的轨迹
            cur_velocities = []  # 当前帧存在个体的速度
            cur_targets = []  # 当前帧存在个体的目标

            # 遍历每个人，将当前帧需要更新人的索引、轨迹、目标存储到数组中
            for i in range(len(id)):
                if start[i] < cur_frame and end[i] > cur_frame:  # 判断agent在当前帧是否存在
                    curt_ids.append(id[i])  # id
                    cur_trajs.append(traj_dict[id[i]])  # 轨迹
                    cur_velocities.append(velocities_dict[id[i]])  # 速度
                    cur_targets.append(dest[i])  # 目标
            if len(curt_ids) == 0:
                continue

            # 获取当前帧所有个体的位置
            locations = [cur_trajs[i][-1] for i in range(len(cur_trajs))]
            locations = np.stack(locations, axis=0)
            locations = torch.from_numpy(locations).to(dtype=torch.float32)
            # print("location: " + str(locations.shape))

            # region DATA VISULIZATION
            # Draw the updated frame
            # ax1.cla()
            # ax1.scatter(locations[:,0], locations[:,1], c='r', marker='o', label='SFM')
            # ax1.set_title('SFM'+str(locations.shape[0]))
            # ax1.set_xlabel('X-axis')
            # ax1.set_ylabel('Y-axis')
            # ax1.legend()

            # ax2.cla()
            # ax2.scatter(locations[:,0], locations[:,1], c='b', marker='o', label='Real Data')
            # ax2.set_title('Sample Data'+str(locations.shape[0]))
            # ax2.set_xlabel('X-axis')
            # ax2.set_ylabel('Y-axis')
            # ax2.legend()
            # endregion

            # 获取当前帧所有个体的速度
            # velocities = [(cur_trajs[i][-1]-cur_trajs[i][-2])/time_step for i in range(len(cur_trajs))]
            velocities = np.stack(cur_velocities, axis=0)
            velocities = torch.from_numpy(velocities).to(dtype=torch.float32)
            # print("Velocity: " + str(velocities.shape))

            # 获取当前帧所有个体的目标位置
            target_locations = np.stack(cur_targets, axis=0)
            target_locations = torch.from_numpy(target_locations).to(dtype=torch.float32)
            # print("target_locations: " + str(target_locations.shape))

            # 根据当前帧位置计算碰撞次数
            collision_times += get_collision_times(locations, collision_threshold=0.4)

            # 根据sfm计算下一帧位置，为了sfm的效果，通过sfm_multi_times控制每timestep之间sfm的迭代次数
            sfm_multi_times = 5
            for _ in range(sfm_multi_times):
                locations, velocities = social_force_update(locations, velocities, target_locations,
                                                            time_step=time_step / sfm_multi_times,desired_speed=dspeed)
                # region DATA VISULIZATION
                # ax1.cla()
                # ax1.scatter(locations[:,0], locations[:,1], c='r', marker='o', label='SFM')
                # ax1.set_title('SFM'+str(locations.shape[0]))
                # ax1.set_xlabel('X-axis')
                # ax1.set_ylabel('Y-axis')
                # ax1.legend()
                # plt.pause(time_step/sfm_multi_times)
                # endregion

            # 如果是最后一帧，再单独计算一次碰撞次数
            if (cur_frame == last_frame):
                collision_times += get_collision_times(locations, collision_threshold=0.4)

            # 将sfm计算得出的位置存入轨迹dict中
            for i, cur_id in enumerate(curt_ids):
                traj_dict[cur_id] = np.concatenate([traj_dict[cur_id], locations[i].unsqueeze(0).numpy()], axis=0)
                velocities_dict[cur_id] = velocities[i].numpy()
                # print("dict shape: " + str(dict[cur_id].shape))
                # print("loc shape: " + str(locations[i].unsqueeze(0).numpy().shape))

        # region DATA VISULIZATION
        # plt.ioff()
        # plt.show()
        # endregion

        print(collision_times)
        dataset_collision_times.append(collision_times)
        dataset_sfm_trajs.append(list(traj_dict.values()))

    return dataset_collision_times, dataset_sfm_trajs



def deepsfm_update(locations, velocities, target_locations,obstacle,seq_start_end,tau,index_c, lambda_c,index_o, lambda_o,time_step=0.4,desired_speed=None):

    # Get the number of agents

    #一、目标吸引力计算#######
    #1.计算朝向
    vector=target_locations- locations #与最后一个位置的方向差值
    desired_directions = vector / (torch.norm(vector, dim=-1, keepdim=True) + 1e-5) #目标方向


    # 2. 计算期望速度
    if desired_speed is not None:
        desired_velocities=desired_speed * desired_directions#期望速度
    else:
        n, length, _ = vector.shape
        scaling_factors = torch.arange(length, 0, -1).cuda() * 0.4
        scaling_factors = scaling_factors.view(1, length, 1)  # Reshape for broadcasting
        # Perform the operation
        desired_velocities = vector / scaling_factors


    #3.计算速度差异
    dv = desired_velocities - velocities#期望速度与实际速度的差异
    #4.缩放
    dv=dv/(tau+0.4) #0.4为时间常数
    #####################


    #二、计算Agent间力#################################
    social_force_agent = torch.zeros_like(locations)
    # 计算碰撞情况
    num=0
    for i,se in enumerate(seq_start_end):
        n=se[1]-se[0]
        len=locations.size(1)
        traj_group=locations[se[0]:se[1]]
        position_data_expanded = traj_group.unsqueeze(1).expand(n, n, len, 2)
        vector = traj_group.unsqueeze(0) - position_data_expanded #i到j (j-i)
        distances = torch.norm(vector,dim=-1)
        directions = vector / (distances.unsqueeze(3) + 1e-5)
        social_force_ij = torch.clamp(lambda_c[i][:,:,:,0:1],min=0.1) * torch.exp(-distances.unsqueeze(-1) / (lambda_c[i][:,:,:,1:])) * (-directions)
        for index in index_c[i]:
            num_i,num_j,num_len=index[0],index[1],index[2]
            social_force_agent[num+num_i][num_len] += social_force_ij[num_i][num_j][num_len]
        num+=n
    ################################################

    # 三、计算障碍物力#################################
    social_force_obstance = torch.zeros_like(locations)
    num=0
    for i,se in enumerate(seq_start_end):
        n = se[1] - se[0]
        len = locations.size(1)
        traj_group = locations[se[0]:se[1]]
        obstacle_group=obstacle[i]
        num_obstacles, _ = obstacle_group.shape
        # 将人的位置和障碍物位置扩展为相同形状以进行广播
        people_positions_expanded = traj_group.unsqueeze(2).expand(n, len, num_obstacles, 2)

        # 计算人和障碍物之间的距离

        vector = obstacle_group - people_positions_expanded
        distances = torch.norm(vector, dim=3)
        directions = vector / (distances.unsqueeze(3) + 1e-5)
        social_force_io = torch.clamp(lambda_o[i][:, :, :, 0:1],min=0.1) * torch.exp(-distances.unsqueeze(-1) / (lambda_o[i][:, :, :, 1:])) * (-directions)
        for index in index_o[i]:
            num_i, num_len, num_ob = index[0], index[1], index[2]
            social_force_obstance[num + num_i][num_len] += social_force_io[num_i][num_len][num_ob]
        num += n
    ################################################

    # Update the velocities by adding the difference in velocities and social force, scaled by the time step
    updated_velocities = velocities + time_step * (dv + social_force_agent+social_force_obstance)

    # Update the locations by adding the updated velocities, scaled by the time step
    updated_locations = locations + time_step * updated_velocities

    return updated_locations

def deepsfm_update_R(locations, velocities, target_locations,obstacle,seq_start_end,tau,index_c, lambda_c,index_o, lambda_o,time_step=0.4,desired_speed=None,col_label=None):

    # Get the number of agents

    # 一、目标吸引力计算#######
    # 1.计算朝向
    vector = target_locations - locations  # 与最后一个位置的方向差值
    desired_directions = vector / (torch.norm(vector, dim=-1, keepdim=True) + 1e-5)  # 目标方向

    # 2. 计算期望速度
    if desired_speed is not None:
        desired_velocities = desired_speed * desired_directions  # 期望速度
    else:
        n, length, _ = vector.shape
        scaling_factors = torch.arange(length, 0, -1).cuda() * 0.4
        scaling_factors = scaling_factors.view(1, length, 1)  # Reshape for broadcasting
        # Perform the operation
        desired_velocities = vector / scaling_factors



    #3.计算速度差异
    dv = desired_velocities - velocities#期望速度与实际速度的差异
    #4.缩放
    dv=dv/(tau+0.4) #0.4为时间常数
    #####################


    #二、计算Agent间力#################################
    social_force_agent = torch.zeros_like(locations)
    # 计算碰撞情况
    num=0
    for i,se in enumerate(seq_start_end):
        n=se[1]-se[0]
        len=locations.size(1)
        traj_group=locations[se[0]:se[1]]
        position_data_expanded = traj_group.unsqueeze(1).expand(n, n, len, 2)
        vector = traj_group.unsqueeze(0) - position_data_expanded #i到j (j-i)
        distances = torch.norm(vector,dim=-1)
        directions = vector / (distances.unsqueeze(3) + 1e-5)
        social_force_ij = torch.clamp(lambda_c[i][:,:,:,0:1],min=0.1) * torch.exp(-distances.unsqueeze(-1) / (lambda_c[i][:,:,:,1:])) * (-directions)
        for index in index_c[i]:
            num_i,num_j,num_len=index[0],index[1],index[2]
            social_force_agent[num+num_i][num_len] += social_force_ij[num_i][num_j][num_len]
        num+=n
    ################################################

    # 三、计算障碍物力#################################
    social_force_obstance = torch.zeros_like(locations)
    num=0
    for i,se in enumerate(seq_start_end):
        n = se[1] - se[0]
        len = locations.size(1)
        traj_group = locations[se[0]:se[1]]
        obstacle_group=obstacle[i]
        num_obstacles, _ = obstacle_group.shape
        # 将人的位置和障碍物位置扩展为相同形状以进行广播
        people_positions_expanded = traj_group.unsqueeze(2).expand(n, len, num_obstacles, 2)

        # 计算人和障碍物之间的距离

        vector = obstacle_group - people_positions_expanded
        distances = torch.norm(vector, dim=3)
        directions = vector / (distances.unsqueeze(3) + 1e-5)
        social_force_io = torch.clamp(lambda_o[i][:, :, :, 0:1],min=0.1) * torch.exp(-distances.unsqueeze(-1) / (lambda_o[i][:, :, :, 1:])) * (-directions)
        for index in index_o[i]:
            num_i, num_len, num_ob = index[0], index[1], index[2]
            social_force_obstance[num + num_i][num_len] += social_force_io[num_i][num_len][num_ob]
        num += n
    ################################################
    R_f=R_force(locations,obstacle,seq_start_end)
    # Update the velocities by adding the difference in velocities and social force, scaled by the time step
    social_force=dv + social_force_agent+social_force_obstance
    mask=torch.norm(R_f,dim=-1)>0

    social_force[mask]=R_f[mask]
    updated_velocities = velocities + time_step * social_force
    #updated_velocities[mask]=0
    # Update the locations by adding the updated velocities, scaled by the time step
    updated_locations = locations + time_step * updated_velocities
    if col_label!=None:#判断是否连续碰撞两次
        mask_col = torch.norm(R_f, dim=-1) > 0 & col_label == 1
        updated_locations[mask_col]=locations[mask_col]+R_f[mask_col]


    return updated_locations,torch.norm(R_f,dim=-1)==0

def deepsfm_update_R_with_stoporwaitRule(locations, velocities, target_locations,obstacle,seq_start_end,tau,index_c, lambda_c,index_o, lambda_o,time_step=0.4,col_label=None,desired_speed=None):
    'col_label:Agent是否连续碰撞障碍物两次'
    pass

def waiting_condition(agent, locations, obstacles, other_agents):
    # Define a threshold distance for waiting
    wait_distance = 2.0  # example threshold

    # Check distance to a specific point or condition
    # Example: waiting if within a certain area
    wait_area_center = torch.tensor([5, 5])  # example area center
    if torch.norm(locations[agent] - wait_area_center) < wait_distance:
        return True  # Wait if within the wait area

    pass

    return False

if __name__ == "__main__":
    # Example usage
    n = 100
    locations = torch.rand(n, 2)*10
    velocities = torch.rand(n, 2)
    target_locations = torch.rand(n, 2)*100
    print("Target Locations:\n", target_locations)
    list=[]
    for i in range(100):
        locations, velocities = R_update(locations, velocities, target_locations)
        print("Updated Locations:\n", locations)
        #print("Updated Velocities:\n", velocities)
        list.append(locations)
    frames=torch.stack(list)

    collision_times, trajs = social_force_collision("eth", 2.2)
    #collision_times, trajs = social_force_collision("hotel", 2.2)
    #collision_times, trajs = social_force_collision("univ",1.3)
    #collision_times, trajs = social_force_collision("zara1", 1.3)
    #collision_times, trajs = social_force_collision("zara2", 1.3)

    #print(collision_times)

    # traj_all=[]
    # for traj in trajs:
    #     for t in traj:
    #         traj_all.append(t)
    #
    # #print( getAllAvgScoreByTraj('eth',traj_all))
    # OutCsv(traj_all,'zara2_sfm')
