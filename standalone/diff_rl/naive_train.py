"""Script to train RL agent with DIFF-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from omni.isaac.lab.app import AppLauncher

# local imports
import cli_args  # isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with DIFF-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
# append DIFF-RL cli arguments
cli_args.add_diff_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from datetime import datetime
from standalone.diff_rl.naive_model import *
from tqdm import tqdm
from tensorboardX import SummaryWriter
from collections import defaultdict

from omni.isaac.lab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml
from omni.isaac.lab_tasks.utils.hydra import hydra_task_config
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

# Import extensions to set up environment tasks
import diff.lab
import diff.lab_tasks

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

# helper function
scaler_q = defaultdict(list)
def smooth_dict(ori_dict):
    for k, v in ori_dict.items():
        scaler_q[k].append(float(v))

@hydra_task_config(args_cli.task, "diff_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with DIFF-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "diff_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # summry writer
    writer = SummaryWriter(log_dir, flush_secs=1)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)
    # initialize model
    model = MLP(env.num_obs, args_cli.dim_hidden, env.num_actions).to(agent_cfg.device)
    model.train()
    if agent_cfg.resume:
        state_dict = torch.load(os.path.join(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint), map_location=agent_cfg.device)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, False)
        if missing_keys:
            print("missing_keys:", missing_keys)
        if unexpected_keys:
            print("unexpected_keys:", unexpected_keys)
    # initialize optimizer
    optim = AdamW(model.parameters(), agent_cfg.algorithm.learning_rate)
    sched = CosineAnnealingLR(optim, agent_cfg.max_iterations, agent_cfg.algorithm.learning_rate * 0.01)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)
    
    # [NOTE] used for debugging gradients flow
    # [NOTE] commentting this line can accelerate training speed
    # torch.autograd.set_detect_anomaly(True)
    
    # randomize initial episode lengths (for exploration)
    if agent_cfg.init_at_random_ep_len:
        env.episode_length_buf = torch.randint_like(
            env.episode_length_buf, high=int(env.max_episode_length)
        )
    obs, _ = env.get_observations()
    
    # run training
    pbar = tqdm(range(agent_cfg.max_iterations), ncols=80)
    for i in pbar:
        dones_history = []
        loss_history = []
        log_loss_history = {}
        # detach env
        env.unwrapped.detach()
        # start rollout
        for t in range(agent_cfg.num_steps_per_env):
            actions = model(obs)
            obs, rews, dones, extras = env.step(actions)
            dones_history.append(dones)
            loss_history.append(extras["losses"])
            for item in extras["log_losses"]:
                if item[0] not in log_loss_history:
                    log_loss_history[item[0]] = []
                log_loss_history[item[0]].append(item[1])

        ####################
        # loss
        ####################
        loss_history = torch.stack(loss_history)
        for k, v in log_loss_history.items():
            log_loss_history[k] = sum(v) / len(v)
        loss = loss_history.mean()
        pbar.set_description_str(f'loss: {loss.cpu().item():.4f}')
        # import torchviz
        # pdf = torchviz.make_dot(loss)
        # pdf.render(filename=f"{i}", directory=os.path.join(log_dir, "debug"), view=False, format='pdf')

        optim.zero_grad()
        loss.backward()
        optim.step()
        sched.step()
        
        torch.cuda.empty_cache()

        #####################
        # log
        #####################
        with torch.no_grad():
            smooth_dict(
                {
                    "loss": loss.cpu().item(),
                    **log_loss_history,
                }
            )
            log_loss_history.clear()

            if i % 25 == 0:
                for k, v in scaler_q.items():
                    writer.add_scalar(k, sum(v) / len(v), i+1)
                scaler_q.clear()

            if (i + 1) % agent_cfg.save_interval == 0:
                torch.save(model.state_dict(), log_dir + f'/model_{i}.pt')

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
