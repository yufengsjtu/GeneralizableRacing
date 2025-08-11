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
# *  Data: 2025/02/27     	             *
# *  Contact: yu-feng@sjtu.edu.cn                                               *
# *  Description: None                                                          *
# *******************************************************************************
import argparse

from omni.isaac.lab.app import AppLauncher

# local imports
from standalone.diff_rl import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Test bare scene.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
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
import torch

from omni.isaac.lab.envs import DirectMARLEnv, multi_agent_to_single_agent
from omni.isaac.lab_tasks.utils import parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper

# Import extensions to set up environment tasks
import diff.lab # noqa: F401
import diff.lab_tasks  # noqa: F401
import cv2 as cv
from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

FRAME_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "frame": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
            scale=(0.1, 0.1, 0.1),
        )
    }
)

frame_marker = FRAME_MARKER_CFG.replace(prim_path="/Visuals/RayCaster")

visualizer = VisualizationMarkers(cfg = frame_marker)

def visualize_depth_image(image):
    depth_normalized = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    # depth_colored = cv.applyColorMap(depth_normalized, cv.COLORMAP_JET)
    cv.imshow("Depth Image", depth_normalized)
    cv.waitKey(1)
    return

def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)
    cnt = 1
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # env stepping
            obs, _, _, _, = env.step(torch.zeros(args_cli.num_envs, env.num_actions, device=env.device))
            image = env.unwrapped.scene.sensors["front_camera"].data.output["distance_to_image_plane"]
            visualize_depth_image(image[1, :, :, 0].cpu().numpy())
            print("Step: ", cnt)
            cnt += 1
            visualizer.visualize(translations=env.unwrapped.scene.sensors["front_camera"].data.pos_w, orientations=env.unwrapped.scene.sensors["front_camera"].data.quat_w_world)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    cv.destroyAllWindows()
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()