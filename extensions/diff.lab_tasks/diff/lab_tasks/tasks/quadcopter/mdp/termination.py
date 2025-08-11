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

def out_of_bound(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    died = torch.logical_or(asset.data.root_pos_w[:, 2] < 0.1, asset.data.root_pos_w[:, 2] > 2.0)
    return died