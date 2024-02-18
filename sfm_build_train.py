import torch
import numpy as np
from evalue import getDatasetPersonInfo,getAllAvgScoreByTraj
from evaluefunction import OutCsv
def social_force_update(locations, velocities, target_locations, time_step=0.4, alpha=200.0, beta=0.08, lambda_=0.35, desired_speed=2.2):
    """
    Update function for the Social Force Model using PyTorch.

    Args:
    locations (torch.Tensor): A tensor of shape [n, 2] representing the current locations (meters) of the agents.
    velocities (torch.Tensor): A tensor of shape [n, 2] representing the current velocities (m/s) of the agents.
    target_locations (torch.Tensor): A tensor of shape [n, 2] representing the target locations (meters) of the agents.
    time_step (float, optional): The time step for updating the locations and velocities (seconds). Default is 0.1.
    alpha (float, optional): The repulsive force strength (Newtons) between agents. Default is 200.0.
    beta (float, optional): The distance (meters) at which the repulsive force is felt. Default is 0.08.
    lambda_ (float, optional): A tuning parameter to control the agent's avoidance force. Default is 0.35.
    desired_speed (float, optional): The desired speed (m/s) of the agents. Default is 1.3.

    Returns:
    updated_locations (torch.Tensor): A tensor of shape [n, 2] representing the updated locations (meters) of the agents.
    updated_velocities (torch.Tensor): A tensor of shape [n, 2] representing the updated velocities (m/s) of the agents.
    """

    # Get the number of agents
    n = locations.size(0)

    # Calculate the desired directions
    desired_directions = (target_locations - locations) / (torch.norm(target_locations - locations, dim=-1, keepdim=True) + 1e-9)

    # Calculate the desired velocities
    desired_velocities = desired_speed * desired_directions

    # Calculate the difference in desired velocities and current velocities
    dv = desired_velocities - velocities

    # Calculate the social force
    social_force = torch.zeros_like(locations)
    for i in range(n):
        for j in range(n):
            if i != j:
                # Calculate the difference in locations between agent i and agent j
                diff_loc = locations[i] - locations[j]
                # Calculate the distance between agent i and agent j
                distance = torch.norm(diff_loc)
                # Normalize the difference in locations
                normalized_diff_loc = diff_loc / (distance + 1e-9)
                # Calculate the social force between agent i and agent j
                social_force_ij = alpha * torch.exp(-distance / beta) * (normalized_diff_loc - lambda_ * velocities[i])
                # Add the social force to the total social force acting on agent i
                social_force[i] += social_force_ij

    # Update the velocities by adding the difference in velocities and social force, scaled by the time step
    updated_velocities = velocities + time_step * (dv + social_force)

    # Update the locations by adding the updated velocities, scaled by the time step
    updated_locations = locations + time_step * updated_velocities

    return updated_locations, updated_velocities



def get_collision_times(locations, collision_threshold=0.4):
    n = locations.size(0)
    collision_times = 0
    for i in range(n):
        for j in range(i + 1, n):
            if (torch.norm(locations[i] - locations[j]) < collision_threshold):
                collision_times += 1
    return collision_times



# if __name__ == "__main__":
#     # Example usage
#     # n = 5
#     # locations = torch.rand(n, 2)
#     # velocities = torch.rand(n, 2)
#     # target_locations = torch.rand(n, 2)
#
#     # updated_locations, updated_velocities = social_force_update(locations, velocities, target_locations)
#     # print("Updated Locations:\n", updated_locations)
#     # print("Updated Velocities:\n", updated_velocities)
#
#     #collision_times, trajs = social_force_collision("eth", 2.2)
#     #collision_times, trajs = social_force_collision("hotel", 2.2)
#     #collision_times, trajs = social_force_collision("univ",1.3)
#     #collision_times, trajs = social_force_collision("zara1", 1.3)
#     #collision_times, trajs = social_force_collision("zara2", 1.3)
#
#     #print(collision_times)
#
#     traj_all=[]
#     for traj in trajs:
#         for t in traj:
#             traj_all.append(t)
#
#     #print( getAllAvgScoreByTraj('eth',traj_all))
#     OutCsv(traj_all,'zara2_sfm')
