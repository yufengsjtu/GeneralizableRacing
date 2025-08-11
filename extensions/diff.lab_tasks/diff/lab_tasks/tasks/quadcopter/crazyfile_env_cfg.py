# *******************************************************************************
# *                                                                             *
# *  Private and Confidential                                                   *
# *                                                                             *
# *  Unauthorized copying of this file, via any medium is strictly prohibited.  *
# *  Proprietary and confidential.                                              *
# *                                                                             *
# *  © 2024 DiffLab. All rights reserved.                                       *
# *                                                                             *
# *  Author: Yu Feng                                                            *
# *  Data: 2024/12/21     	                                                    *
# *  Contact: yu-feng@sjtu.edu.cn                                               *
# *  Description: None                                                          *
# *******************************************************************************
from omni.isaac.lab.utils import configclass
from omni.isaac.lab_assets import CRAZYFLIE_CFG  # isort: skip
from .quadcopter_env_cfg import QuadcopterEnvCfg

@configclass
class CrazyfileEnvCfg(QuadcopterEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # adjust episode length seconds
        self.episode_length_s = 5.0
        # switch robot to crazyflie
        self.scene.robot = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")


