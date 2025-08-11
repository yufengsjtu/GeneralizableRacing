from omni.isaac.lab.utils import configclass
from standalone.diff_rl.algorithms.config import (
    DiffRLPolicyRunnerCfg,
    DiffRLModelCfg,
    DiffRLAlgorithmCfg
)
from typing import Literal
@configclass
class QuadcopterDIffRLRunnerCfg(DiffRLPolicyRunnerCfg):
    num_steps_per_env = 48
    max_iterations = 2000
    save_interval = 200
    experiment_name = "test_hover"
    # experiment_name = "test_racing"
    empirical_normalization = False
    init_at_random_ep_len = True

    algorithm = DiffRLAlgorithmCfg(
        class_name="BPTT",
        schedule="CosineAnnealingLR",
        optimizer="AdamW",
        learning_rate=5e-4

    )

    policy = DiffRLModelCfg(
        class_name="BaseModel",
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[256, 128],
        activation="lrelu",
        init_noise_std=1.0,
    )

@configclass
class DiffRLVisionModelCfg(DiffRLModelCfg):
    class_name: str = "VisionActorCritic"
    img_res: tuple[int, int] = (72, 96)
    dim_hidden_input: int = 192
    init_noise_std=1.0
    actor_hidden_dims=[128, 128]
    critic_hidden_dims=[128, 128]
    activation="lrelu"
    noise_std_type: Literal["scalar", "log"] = "scalar"


@configclass
class DiffRLVisionRecurrentModelCfg(DiffRLModelCfg):
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
class QuadcopterDIffRLVisionRunnerCfg(DiffRLPolicyRunnerCfg):
    num_steps_per_env = 48
    max_iterations = 2000
    save_interval = 200
    experiment_name = "vision_racing"
    empirical_normalization = False
    init_at_random_ep_len = True

    algorithm = DiffRLAlgorithmCfg(
        class_name="BPTT",
        schedule="CosineAnnealingLR",
        optimizer="AdamW",
        learning_rate=5e-4

    )

    policy = DiffRLVisionModelCfg()