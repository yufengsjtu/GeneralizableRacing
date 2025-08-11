"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# local imports
from standalone.diff_rl import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Test environment.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from torch.autograd import Variable
from rsl_rl.runners import OnPolicyRunner
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
import time
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        return self.fc2(self.leaky_relu(self.fc1(x)))

from omni.isaac.lab.envs import DirectMARLEnv, multi_agent_to_single_agent
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)

# Import extensions to set up environment tasks
import diff.lab # noqa: F401
import diff.lab_tasks  # noqa: F401


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
   
    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)
    # intialize model
    model = MLP(env.num_obs, 256, env.num_actions).to(args_cli.device)
    # reset environment
    obs, _ = env.get_observations()
    # torch.autograd.set_detect_anomaly(True)
    steps = 1
    # simulate environment
    while simulation_app.is_running():
        loss_history = []
        obs, _ = env.reset()
        for i in range(steps):
            actions = model(obs)
            obs, rews, dones, extras = env.step(actions)
            loss_history.append(extras["losses"])
        loss = torch.stack(loss_history).mean()
        import torchviz
        pdf = torchviz.make_dot(loss)
        pdf.render(filename=f"{steps}", view=False, format='pdf')

        loss.backward()

        break

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
        
    