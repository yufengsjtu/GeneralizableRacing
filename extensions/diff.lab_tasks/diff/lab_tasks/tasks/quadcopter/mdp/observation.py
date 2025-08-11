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
from omni.isaac.lab.utils.math import subtract_frame_transforms
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import SceneEntityCfg
if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

def desired_position_b(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), command_name: str = "desired_pos_w") -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    desired_position_w = env.command_manager.get_command(command_name)[:, :3]
    desired_position_b, _ = subtract_frame_transforms(
            asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], desired_position_w
        )
    return desired_position_b