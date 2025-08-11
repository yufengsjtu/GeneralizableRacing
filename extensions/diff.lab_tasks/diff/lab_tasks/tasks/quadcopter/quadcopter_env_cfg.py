# *******************************************************************************
# *                                                                             *
# *  Private and Confidential                                                   *
# *                                                                             *
# *  Unauthorized copying of this file, via any medium is strictly prohibited.  *
# *  Proprietary and confidential.                                              *
# *                                                                             *
# *  © 2024 DiffLab. All rights reserved.                                       *
# *                                                                             *
# *  Author: Su Yang                                                            *
# *  Data: 2024/11/14     	                                                    *
# *  Contact: None                                                              *
# *  Description: None                                                          *
# *******************************************************************************

from __future__ import annotations
from dataclasses import MISSING
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from diff.lab.terrains import TerrainImporterCfg as diff_TerrainImporterCfg
from omni.isaac.lab.utils import configclass
import diff.lab_tasks.tasks.quadcopter.mdp as mdp
from diff.lab.terrains.config import RACINGTERRAIN_CFG
from diff.lab_assets.quadcopter import DRONE_CFG
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR
@configclass
class SceneCfg(InteractiveSceneCfg):
    # terrain = TerrainImporterCfg(
    #         prim_path="/World/ground",
    #         terrain_type="plane",
    #         collision_group=-1,
    #         physics_material=sim_utils.RigidBodyMaterialCfg(
    #             friction_combine_mode="multiply",
    #             restitution_combine_mode="multiply",
    #             static_friction=1.0,
    #             dynamic_friction=1.0,
    #             restitution=0.0,
    #         ),
    #         debug_vis=False,
    #     )
    
    terrain = diff_TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator = RACINGTERRAIN_CFG,
        max_init_terrain_level=None,
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
    
    robot: ArticulationCfg = DRONE_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )
    distant_light = AssetBaseCfg(
        prim_path="/World/DistantLight",
        spawn=sim_utils.DistantLightCfg(color=(0.9, 0.9, 0.9), intensity=2500.0),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.738, 0.477, 0.477, 0.0)),
    )

@configclass
class CommandsCfg: #TODO considering the root pos
    """Command specifications for the environment."""
    desired_position_w = mdp.WorldPoseCommandCfg(
        asset_name="robot",
        body_name="body", 
        resampling_time_range=(10.0, 10.0),
        debug_vis=True,
        ranges = mdp.WorldPoseCommandCfg.Ranges(
            pos_x=(-2.0,2.0),pos_y=(-2.0,2.0),pos_z=(0.5,1.0),roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        )
    )

@configclass
class ActionsCfg:
    """Action specifications for the environment."""
    force_torque = mdp.PropellerThrustActionsCfg(asset_name="robot")


@configclass
class ObservationsCfg:
    """Observation specifications for the environment."""
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        desired_pos_b = ObsTerm(
            func = mdp.desired_position_b,
            params = {
                "command_name":"desired_position_w"
            })

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    
    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    """Event specifications for the environment."""
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1),"z":(0.7,1.6), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )

@configclass
class RewardsCfg:
    """Reward specifications for the environment."""
    linear_velocity_reward = RewTerm(
        func = mdp.linear_velocity_reward,
        weight= -0.03,
    )
    
    angular_velocity_reward = RewTerm(
        func = mdp.angular_velocity_reward,
        weight= -0.08,
    )

    target_reward = RewTerm(
        func = mdp.target_reward,
        weight = 30.0,
        params={
            "command_name": "desired_position_w"
        }
    )

    body_linear_accel_reward = RewTerm(
        func = mdp.body_lin_acc_l2,
        weight= -0.0005,
    )

    die_reward = RewTerm(
        func= mdp.is_terminated,
        weight= -1000
    )

@configclass
class TerminationsCfg:
    """Termination specifications for the environment."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    out_of_bound = DoneTerm(func=mdp.out_of_bound)

@configclass
class CurriculumCfg:
    """Curriculum specifications for the environment."""
    pass

@configclass
class QuadcopterEnvCfg(ManagerBasedRLEnvCfg):
    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:

        """Post initialization."""

        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        # self.sim.disable_contact_processing=True
        self.sim.physics_material = self.scene.terrain.physics_material

