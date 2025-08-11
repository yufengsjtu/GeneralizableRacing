# *******************************************************************************
# *                                                                             *
# *  Private and Confidential                                                   *
# *                                                                             *
# *  Unauthorized copying of this file, via any medium is strictly prohibited.  *
# *  Proprietary and confidential.                                              *
# *                                                                             *
# *  © 2024 DiffLab. All rights reserved.                                       *
# *                                                                             *
# *  Author: Su Yang                                                            *
# *  Data: 2024/11/14                                           	            *
# *  Contact: None                                                              *
# *  Description: None                                                          *
# *******************************************************************************
from __future__ import annotations
import torch
from typing import TYPE_CHECKING
from omni.isaac.lab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

def linear_velocity_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
)-> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    lin_vel = torch.sum(torch.square(asset.data.root_lin_vel_b), dim=1)
    return lin_vel


def angular_velocity_reward(
    env: ManagerBasedRLEnv,  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
)-> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    ang_vel = torch.sum(torch.square(asset.data.root_ang_vel_b), dim=1)
    return ang_vel

def target_reward(
        env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
        ) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    _desired_pos_w = env.command_manager.get_command(command_name)[:, :3]
    distance_to_goal = torch.linalg.norm(_desired_pos_w - asset.data.root_pos_w, dim=1)
    distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)
    return distance_to_goal_mapped