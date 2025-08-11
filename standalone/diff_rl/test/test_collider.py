# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to add and simulate on-board sensors for a robot.

We add the following sensors on the quadruped robot, ANYmal-C (ANYbotics):

* USD-Camera: This is a camera sensor that is attached to the robot's base.
* Height Scanner: This is a height scanner sensor that is attached to the robot's base.
* Contact Sensor: This is a contact sensor that is attached to the robot's feet.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/04_sensors/add_sensors_on_robot.py --enable_cameras

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on adding sensors on a robot.")
parser.add_argument("--num_envs", type=int, default=256, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns
from omni.isaac.lab.utils import configclass
import diff.lab
import diff.lab_tasks
from diff.lab_assets import DRONE_CFG, DRONE_NO_COLLIDER_CFG
from diff.lab.utils import get_uav_collision_num_ray, LATTICE_TENSOR
from diff.lab.terrains import TerrainImporterCfg as diff_TerrainImporterCfg
from diff.lab_tasks.tasks.quadcopter_diff.terrains import *
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR
import time

@configclass
class SensorsSceneCfg(InteractiveSceneCfg):
    """Design the scene with sensors on the robot."""

    # ground plane
    terrain = diff_TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator = RacingTerrainWOObsPPOCfg.replace(num_cols=10, num_rows=10),
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
        )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # robot
    # robot: ArticulationCfg = DRONE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot: ArticulationCfg = DRONE_NO_COLLIDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    # contact_forces = ContactSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/body", update_period=0.0, history_length=6, debug_vis=True
    # )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Run the simulator."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    collision = []

    # Simulate physics
    while simulation_app.is_running():
        # Reset
        if count % 400 == 0:
            root_state = scene["robot"].data.default_root_state.clone()
            # root_state[:, :3] += scene.env_origins
            root_state[:, :3] = scene.terrain.extras["gate_pose"][scene.terrain.terrain_types, scene.terrain.terrain_levels, 0, :3] + scene.env_origins
            scene["robot"].write_root_pose_to_sim(root_state[:, :7])
            scene["robot"].write_root_velocity_to_sim(root_state[:, 7:])
            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting robot state...")
        # apply random actions
        # ids, _ = scene["robot"].find_bodies("body", preserve_order = True)
        # scene["robot"].set_external_force_and_torque(torch.randn(2, 1, 3, device=scene.device), torch.randn(2, 1, 3, device=scene.device), body_ids=ids)
        # -- write data to sim
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        scene.update(sim_dt)

        # print("-------------------------------")
        # print(scene["contact_forces"])
        # print("Received max contact force of: ", torch.max(scene["contact_forces"].data.net_forces_w).item())
        # contact_force = (scene["contact_forces"].data.net_forces_w[0, 0].norm(2, -1)).item()
        # collision.append(contact_force > 1.0)
        t0 = time.time()
        num_collision = get_uav_collision_num_ray(
            scene.terrain.warp_meshes["terrain"],
            scene["robot"].data.root_pos_w,
            scene["robot"].data.root_quat_w,
            0.09,
            0.05,
            1e3,
            LATTICE_TENSOR.to(scene.device),
        )
        print("Collision detection time: ", time.time() - t0)
        print("num_collision",num_collision.shape)
        collision.append(num_collision)

        if count > 1200:
            break
    return collision

def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])
    # design scene
    scene_cfg = SensorsSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    return run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    collsion = main()
    collsion = torch.stack(collsion).cpu().numpy()
    print("plot collsion")
    import matplotlib.pyplot as plt
    import random
    plt.subplot(2, 1, 1)
    plt.plot(collsion[:, random.randint(0, args_cli.num_envs - 1)], label="env_0")
    plt.legend()
    plt.title("Collision detection")
    plt.subplot(2, 1, 2)
    plt.plot(collsion[:, random.randint(0, args_cli.num_envs - 1)], label="env_1")
    plt.legend()
    plt.title("Collision detection")
    
    plt.show()
    # close sim app
    simulation_app.close()
    