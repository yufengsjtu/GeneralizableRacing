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
from dataclasses import MISSING
from typing import TYPE_CHECKING
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import ActionTerm, ActionTermCfg
from omni.isaac.lab.utils import configclass
if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

class PropellerThrustActions(ActionTerm):
    '''
    This is a simple quadcopter thrust action term used to apply thrust to the propellers of a quadcopter. \n

    Convention:
        1. Coordination system follows FLU frame. 
        2. Propeller are numbered in clockwise order: 
            1: front right (e.g m1_prop in crazyflie) 
            2: back right (e.g m2_prop in crazyflie) 
            3: back left (e.g m3_prop in crazyflie) 
            4: front left (e.g m4_prop in crazyflie) 
        3. The thrust is applied to the propeller along the body z-axis. 
            
             4            1
            ***          ***
             *            *
               *        *
                 *    *
                   **
                 *    *
               *        *
             *            *
            ***           ***
             3             2
    Mathamatical model:
        Input: four propeller throttle t_i values in range [0.0, 1.0]
        F_i = thrust_to_weight * m_robot * gravity * t_i
        F_total = F1 + F2 + F3 + F4 (computed by isaac-sim)
        tau_x = coeff_T * (F1 + F2 - F3 - F4) (computed by isaac-sim)
        tau_y = coeff_T * (F2 + F3 - F1 - F4) (computed by isaac-sim)
        tau_z = coeff_M * (F1 - F2 + F3 - F4) (computed by this class)
        coeff_T = d * cos(45) (d is the distance from the center of the quadcopter to the propeller)
        coeff_M can be obtained by experiments.

    '''
    cfg: PropellerThrustActionsCfg
    _thrusts: torch.Tensor
    _torques: torch.Tensor

    def __init__(self, cfg:PropellerThrustActionsCfg, env:ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._propeller_ids, self._propeller_names = self.robot.find_bodies(self.cfg.body_names, preserve_order = True)
        self._base_ids, self._base_names = self.robot.find_bodies("body", preserve_order = True)
        self._processed_actions = torch.zeros_like(self.raw_actions)
        # 4 propellers thrust, each with 3 components
        self._thrusts = torch.zeros(self.num_envs, 4, 3, device=self.device)
        # torques on base (the first 2 are zero, which are computed by isaac-sim using the \
        # thrusts above, the last one stands for the invertd-torque computed by this class)
        self._torques = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._gravity_magnitude = torch.tensor(self.cfg.gravity, device=self.device).norm()
        self._robot_mass = self.robot.root_physx_view.get_masses()[0].sum()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 4

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions
    
    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        # actions: four propeller throttle values in range [0.0, 1.0]
        self._raw_actions[:] = actions.clone().sigmoid_()   
        self._thrusts[..., 2] = self.cfg.thrust_to_weight * self._robot_weight * self._raw_actions
        # 0.0131 = C_M / C_T = (1.574 * 10^-7) / (1.201 * 10^-5) (T-motor)
        torques_ = 0.0131 * (- self._thrusts[:,0,:] + self._thrusts[:,1,:] - self._thrusts[:,2,:] + self._thrusts[:,3,:])
        self._torques[:,0,:] = torques_
        
    def apply_actions(self):
        # set the external force to the propellers
        self.robot.set_external_force_and_torque(self._thrusts, torch.zeros_like(self._thrusts), body_ids=self._propeller_ids)
        # set the external torque to the base
        self.robot.set_external_force_and_torque(torch.zeros_like(self._torques), self._torques, body_ids=self._base_ids)
    

@configclass
class PropellerThrustActionsCfg(ActionTermCfg):
    """Configuration for the joint position action term.

    See :class:`JointPositionAction` for more details.
    """

    class_type: type[ActionTerm] = PropellerThrustActions
    asset_name: str = MISSING
    body_names: str = "m.*_prop"
    thrust_to_weight: float = 0.375
    gravity: float = 9.81