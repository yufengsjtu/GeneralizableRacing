# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass
from typing import Literal
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class QuadcopterPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 500
    experiment_name = "racing_ppo"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[128, 128],
        critic_hidden_dims=[128, 128],
        activation="lrelu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
@configclass
class RslRlPpoVisionActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name: str = "VisionActorCritic"
    img_res: tuple[int, int] = (72, 96)
    dim_hidden_input: int = 192
    init_noise_std=1.0
    actor_hidden_dims=[128, 128]
    critic_hidden_dims=[128, 128]
    activation="lrelu"
    noise_std_type: Literal["scalar", "log"] = "scalar"

@configclass
class RslRlBCVisionActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name: str = "VisionStudentTeacher"
    img_res: tuple[int, int] = (72, 96)
    dim_hidden_input: int = 192
    init_noise_std=0.1
    student_hidden_dims=[128, 128]
    teacher_hidden_dims=[128, 128]
    activation="lrelu"
    noise_std_type: Literal["scalar", "log"] = "scalar"

@configclass
class RslRlPpoVisionActorCriticRecurrentCfg(RslRlPpoActorCriticCfg):
    class_name: str = "VisionActorCriticRecurrent"
    img_res: tuple[int, int] = (72, 96)
    dim_hidden_input: int = 192
    rnn_type: Literal["lstm", "gru"] = "gru"
    rnn_hidden_size: int = 192
    rnn_num_layers: int = 1
    init_noise_std=1.0
    actor_hidden_dims=[128, 128]
    critic_hidden_dims=[128, 128]
    activation="lrelu"
    noise_std_type: Literal["scalar", "log"] = "scalar"

@configclass
class QuadcopterVisionPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 4000
    save_interval = 500
    experiment_name = "racing_ppo_l2c2_vision"
    empirical_normalization = False
    policy = RslRlPpoVisionActorCriticCfg()

    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPOL2C2",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    # algorithm.__setattr__("grad_penalty_coef_schedule",[0.1, 0.1, 700, 1000])
    # algorithm.__setattr__("use_auxiliary_loss", True)
    policy.__setattr__("use_auxiliary_loss", True)

@configclass
class QuadcopterVisionBCRunnerCfg(RslRlOnPolicyRunnerCfg):
    max_iterations = 1000
    save_interval = 200
    experiment_name = "racing_ppo_vision"
    empirical_normalization = False
    policy = RslRlBCVisionActorCriticCfg()
    resume = True
    num_steps_per_env = 24

    algorithm = RslRlPpoAlgorithmCfg(
        class_name="Distillation",
        num_learning_epochs=5,
        learning_rate=5e-4,
    )

