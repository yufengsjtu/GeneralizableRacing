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

# keep seed of the environment fixed
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env_cfg.seed = 0
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
    cnt = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        # with torch.inference_mode():
        obs, _ = env.reset()
        aligned_traj = []
        nominal_traj = []
        gt_traj = []
        nominal_acc_traj = []
        gt_acc_traj = []
        #
        steps = 200
        # agent stepping
        t0 = time.time()
        for i in range(steps):
            # env stepping
            actions = model(obs)
            obs, rews, dones, extras = env.step(actions)
            aligned_traj.append(extras["aligned_states"])
            nominal_traj.append(extras["nominal_states"])
            nominal_acc_traj.append(extras["acc"])
            gt_acc = env.unwrapped.scene["robot"].data.body_lin_acc_w[:, 0, :].clone()
            gt_acc_traj.append(gt_acc)
            gt_state = env.unwrapped.scene["robot"].data.root_state_w.clone()
            gt_state[:, :3] -= env.unwrapped.scene.env_origins
            gt_traj.append(gt_state)
        print(f"Data collection for {steps} steps costs {time.time() - t0} seconds.")
        loss = torch.stack(aligned_traj).mean()
        # import torchviz
        # pdf = torchviz.make_dot(loss)
        # pdf.render(filename=f"{cnt}", view=False, format='pdf')
        
        # loss.backward()
        # print(f"Running epoch-[{cnt}]")
        # cnt += 1
        
        # visualize the states
        nominal_traj = torch.stack(nominal_traj).detach().cpu().numpy()
        gt_traj = torch.stack(gt_traj).detach().cpu().numpy()
        aligned_traj = torch.stack(aligned_traj).detach().cpu().numpy()
        nominal_acc_traj = torch.stack(nominal_acc_traj).detach().cpu().numpy()
        gt_acc_traj = torch.stack(gt_acc_traj).detach().cpu().numpy()
        import random
        ind = random.randint(0, env.num_envs - 1)
        print(f"Visualizing trajectory for env-{ind}")
        fig, ax = plt.subplots(3, 1)
        ax[0].plot(range(steps), aligned_traj[:, ind, 0], label="Aligned_x")
        ax[0].plot(range(steps), nominal_traj[:, ind, 0], label="Nominal_x")
        ax[0].plot(range(steps), gt_traj[:, ind, 0], label="GT_x")
        # ax[0].set_ylim(-1, 1)
        plt.legend()
        ax[1].plot(range(steps), aligned_traj[:, ind, 1], label="Aligned_y")
        ax[1].plot(range(steps), nominal_traj[:, ind, 1], label="Nominal_y")
        ax[1].plot(range(steps), gt_traj[:, ind, 1], label="GT_y")
        # ax[1].set_ylim(-1, 1)
        plt.legend()
        ax[2].plot(range(steps), aligned_traj[:, ind, 2], label="Aligned_z")
        ax[2].plot(range(steps), nominal_traj[:, ind, 2], label="Nominal_z")
        ax[2].plot(range(steps), gt_traj[:, ind, 2], label="GT_z")
        # ax[2].set_ylim(0, 1)
        plt.legend()
        plt.savefig("test_p.png")

        fig, ax = plt.subplots(3, 1)
        ax[0].plot(range(steps), aligned_traj[:, ind, 7], label="Aligned_vx")
        ax[0].plot(range(steps), nominal_traj[:, ind, 7], label="Nominal_vx")
        ax[0].plot(range(steps), gt_traj[:, ind, 7], label="GT_vx")
        # ax[0].set_ylim(-1, 1)
        plt.legend()
        ax[1].plot(range(steps), aligned_traj[:, ind, 8], label="Aligned_vy")
        ax[1].plot(range(steps), nominal_traj[:, ind, 8], label="Nominal_vy")
        ax[1].plot(range(steps), gt_traj[:, ind, 8], label="GT_vy")
        # ax[1].set_ylim(-1, 1)
        plt.legend()
        ax[2].plot(range(steps), aligned_traj[:, ind, 9], label="Aligned_vz")
        ax[2].plot(range(steps), nominal_traj[:, ind, 9], label="Nominal_vz")
        ax[2].plot(range(steps), gt_traj[:, ind, 9], label="GT_vz")
        # ax[2].set_ylim(0, 1)
        plt.legend()
        plt.savefig("test_v.png")

        fig, ax = plt.subplots(4, 1)
        ax[0].plot(range(steps), aligned_traj[:, ind, 3], label="Aligned_qw")
        ax[0].plot(range(steps), nominal_traj[:, ind, 3], label="Nominal_qw")
        ax[0].plot(range(steps), gt_traj[:, ind, 3], label="GT_qw")
        # ax[0].set_ylim(-1, 1)
        plt.legend()
        ax[1].plot(range(steps), aligned_traj[:, ind, 4], label="Aligned_qx")
        ax[1].plot(range(steps), nominal_traj[:, ind, 4], label="Nominal_qx")
        ax[1].plot(range(steps), gt_traj[:, ind, 4], label="GT_qx")
        # ax[1].set_ylim(-1, 1)
        plt.legend()
        ax[2].plot(range(steps), aligned_traj[:, ind, 5], label="Aligned_qy")
        ax[2].plot(range(steps), nominal_traj[:, ind, 5], label="Nominal_qy")
        ax[2].plot(range(steps), gt_traj[:, ind, 5], label="GT_qy")
        # ax[2].set_ylim(-1, 1)
        plt.legend()
        ax[3].plot(range(steps), aligned_traj[:, ind, 6], label="Aligned_qz")
        ax[3].plot(range(steps), nominal_traj[:, ind, 6], label="Nominal_qz")
        ax[3].plot(range(steps), gt_traj[:, ind, 6], label="GT_qz")
        # ax[3].set_ylim(-1, 1)
        plt.legend()
        plt.savefig("test_q.png")

        fig, ax = plt.subplots(3, 1)
        ax[0].plot(range(steps), aligned_traj[:, ind, 10], label="Aligned_wx")
        ax[0].plot(range(steps), nominal_traj[:, ind, 10], label="Nominal_wx")
        ax[0].plot(range(steps), gt_traj[:, ind, 10], label="GT_wx")
        # ax[0].set_ylim(-1, 1)
        plt.legend()
        ax[1].plot(range(steps), aligned_traj[:, ind, 11], label="Aligned_wy")
        ax[1].plot(range(steps), nominal_traj[:, ind, 11], label="Nominal_wy")
        ax[1].plot(range(steps), gt_traj[:, ind, 11], label="GT_wy")
        # ax[1].set_ylim(-1, 1)
        plt.legend()
        ax[2].plot(range(steps), aligned_traj[:, ind, 12], label="Aligned_wz")
        ax[2].plot(range(steps), nominal_traj[:, ind, 12], label="Nominal_wz")
        ax[2].plot(range(steps), gt_traj[:, ind, 12], label="GT_wz")
        # ax[2].set_ylim(-1, 1)
        plt.legend()
        plt.savefig("test_w.png")

        fig, ax = plt.subplots(3, 1)
        ax[0].plot(range(10), aligned_traj[60:70, ind, 10], label="Aligned_wx")
        ax[0].plot(range(10), nominal_traj[60:70, ind, 10], label="Nominal_wx")
        ax[0].plot(range(10), gt_traj[60:70, ind, 10], label="GT_wx")
        # ax[0].set_ylim(-1, 1)
        plt.legend()
        ax[1].plot(range(10), aligned_traj[60:70, ind, 11], label="Aligned_wy")
        ax[1].plot(range(10), nominal_traj[60:70, ind, 11], label="Nominal_wy")
        ax[1].plot(range(10), gt_traj[60:70, ind, 11], label="GT_wy")
        # ax[1].set_ylim(-1, 1)
        plt.legend()
        ax[2].plot(range(10), aligned_traj[60:70, ind, 12], label="Aligned_wz")
        ax[2].plot(range(10), nominal_traj[60:70, ind, 12], label="Nominal_wz")
        ax[2].plot(range(10), gt_traj[60:70, ind, 12], label="GT_wz")
        # ax[2].set_ylim(-1, 1)
        plt.legend()
        plt.savefig("test_w_local.png")
    
        fig, ax = plt.subplots(3, 1)
        ax[0].plot(range(steps), nominal_acc_traj[:, ind, 0], label="Nominal_ax")
        ax[0].plot(range(steps), gt_acc_traj[:, ind, 0], label="GT_ax")
        # ax[0].set_ylim(-1, 1)
        plt.legend()
        ax[1].plot(range(steps), nominal_acc_traj[:, ind, 1], label="Nominal_ay")
        ax[1].plot(range(steps), gt_acc_traj[:, ind, 1], label="GT_ay")
        # ax[1].set_ylim(-1, 1)
        plt.legend()
        ax[2].plot(range(steps), nominal_acc_traj[:, ind, 2], label="Nominal_az")
        ax[2].plot(range(steps), gt_acc_traj[:, ind, 2], label="GT_az")
        # ax[2].set_ylim(-1, 1)
        plt.legend()
        plt.savefig("test_acc.png")

        break
        

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
