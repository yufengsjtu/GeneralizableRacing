"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# local imports
from standalone.rsl_rl import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--onnx_path", type=str, default=None, help="Path to the onnx model.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch

from standalone.rsl_rl.ext.runners import OnPolicyRunner

from omni.isaac.lab.envs import DirectMARLEnv, multi_agent_to_single_agent
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
)
from standalone.rsl_rl.ext.utils import export_vision_policy_as_onnx

# Import extensions to set up environment tasks
import diff.lab # noqa: F401
import diff.lab_tasks  # noqa: F401

import onnxruntime as ort
import numpy as np
import time
class OnnxPolicy:
    def __init__(self, model_path):
        self.model = ort.InferenceSession(model_path)
        self.input_info = [(i.name, i.shape) for i in self.model.get_inputs()]
        self.output_info = [(i.name, i.shape) for i in self.model.get_outputs()]
        print("Input information: ", self.input_info)
        print("Input information: ", self.output_info)

    def preheat(self, ):
        self.__parser_info__()
        t0 = time.time()
        for _ in range(6):
            self.model.run(
                None,
                self.input_dict
            )
        t1 = time.time()
        print("Preheat time: ", t1 - t0)

    def __parser_info__(self, ):
        self.input_dict = {}
        for name, shape in self.input_info:
            self.input_dict[name] = np.zeros(shape, dtype=np.float32)
        self.output_dict = {}
        for name, shape in self.output_info:
            self.output_dict[name] = np.zeros(shape, dtype=np.float32)

    def act(self, obs_dict):
        assert isinstance(obs_dict, dict), "obs_dict must be a dict"
        for name, shape in self.input_info:
            if name in obs_dict:
                if obs_dict[name] is None:
                    print("[Warning] obs_dict[{}] is None".format(name))
                    self.input_dict[name] = np.zeros(shape, dtype=np.float32)
                    continue
                if tuple(obs_dict[name].shape) != tuple(shape):
                    raise ValueError(f"obs_dict[{name}] shape {obs_dict[name].shape} must be {shape}")
                self.input_dict[name] = obs_dict[name]
            else:
                raise ValueError(f"obs_dict must contain {name}")
        # run the model
        output = self.model.run(
            None,
            self.input_dict
        )
        return output
import cv2 as cv

def visualize_depth_image(depth_map):
    depth_normalized = depth_map.copy()
    depth_normalized.clip(min=0, max=1.0, out=depth_normalized)
    
    depth_normalized = 255 * (depth_map)
    depth_uint8 = np.uint8(np.clip(depth_normalized, 0, 255))

    return depth_uint8

def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
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

    policy = OnnxPolicy(args_cli.onnx_path)
    policy.preheat()

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            # parser observation
            obs_dict = {}
            obs_dict["state"] = obs[..., :15].cpu().numpy()
            obs_dict["img"] = obs[..., 15:].reshape(-1, 1, 72, 96).cpu().numpy()
            # depth_vis = visualize_depth_image(obs_dict["img"][0][0])
            # cv.imwrite("depth_vis.png", depth_vis)
            # exit()
            actions = policy.act(obs_dict)
            # env stepping
            obs, _, _, _ = env.step(torch.from_numpy(actions[0]).to(env.unwrapped.device))
        
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

# test command:
# python standalone/rsl_rl/ext/utils/onnx_validation.py --onnx_path ./logs/rsl_rl/racing_ppo_vision/2025-04-24_19-06-38/exported/vision_policy.onnx --task DiffLab-Quadcopter-CTBR-Racing-v0 --num_envs 1