# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Quacopter environment.
"""

import gymnasium as gym

from . import agents

from .reach_target_ctbr_env import QuadcopterReachTargetCTBREnvCfg
from .racing_ctbr_env import QuadcopterRacingCTBREnvCfg
from .reach_target_lv_env import QuadcopterReachTargetLVEnvCfg
##
# Register Gym environments.
##

gym.register(
    id="DiffLab-Quadcopter-LV-ReachTarget-v0",
    entry_point="diff.lab.envs:ManagerBasedDiffRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": QuadcopterReachTargetLVEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadcopterPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "diff_rl_cfg_entry_point": f"{agents.__name__}.diff_rl_naive_cfg:QuadcopterDIffRLRunnerCfg"
    },
)

gym.register(
    id="DiffLab-Quadcopter-CTBR-ReachTarget-v0",
    entry_point="diff.lab.envs:ManagerBasedDiffRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": QuadcopterReachTargetCTBREnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadcopterPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "diff_rl_cfg_entry_point": f"{agents.__name__}.diff_rl_naive_cfg:QuadcopterDIffRLRunnerCfg"
    },

)

gym.register(
    id="DiffLab-Quadcopter-CTBR-Racing-v0",
    entry_point="diff.lab.envs:ManagerBasedDiffRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": QuadcopterRacingCTBREnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadcopterVisionPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "diff_rl_cfg_entry_point": f"{agents.__name__}.diff_rl_naive_cfg:QuadcopterDIffRLVisionRunnerCfg"
    },

)

gym.register(
    id="DiffLab-Quadcopter-CTBR-Racing-BC-v0",
    entry_point="diff.lab.envs:ManagerBasedDiffRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": QuadcopterRacingCTBREnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadcopterVisionBCRunnerCfg",
    },
)
