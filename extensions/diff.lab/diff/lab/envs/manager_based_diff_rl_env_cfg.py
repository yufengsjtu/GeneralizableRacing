from omni.isaac.lab.envs.manager_based_rl_env_cfg import ManagerBasedRLEnvCfg
from omni.isaac.lab.utils import configclass
from dataclasses import MISSING

@configclass
class ManagerBasedDiffRLEnvCfg(ManagerBasedRLEnvCfg):
    losses: object = MISSING
    is_differentiable_physics: bool = True