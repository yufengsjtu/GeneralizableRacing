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
# *  Data: 2024/12/23     	                                                    *
# *  Contact: None                                                              *
# *  Description: None                                                          *
# *******************************************************************************

from __future__ import annotations
from dataclasses import MISSING
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from diff.lab.managers import LossTermCfg as LossTerm
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from diff.lab.controllers.controller_diff_cfg import CTBRControllerCfg
from diff.lab.envs.manager_based_diff_rl_env_cfg import ManagerBasedDiffRLEnvCfg
from omni.isaac.lab.sensors import ContactSensorCfg
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from diff.lab_assets.quadcopter import DRONE_CFG

import diff.lab_tasks.tasks.quadcopter_diff.mdp as mdp
SIM2REAL_TEST = True
@configclass
class SceneCfg(InteractiveSceneCfg):
    if not SIM2REAL_TEST:
        terrain = TerrainImporterCfg(
                prim_path="/World/ground",
                terrain_type="plane",
                collision_group=-1,
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
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
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/body", history_length=3, track_air_time=True, debug_vis=True)

@configclass
class CommandsCfg: #TODO considering the root pos
    desired_pos_b = mdp.WorldPoseCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        debug_vis=True,
        ranges = mdp.WorldPoseCommandCfg.Ranges(
            pos_x=(-2.0,2.0),pos_y=(-2.0,2.0),pos_z=(0.5,2.5),roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        )
    )

@configclass
class ActionsCfg:
    """Action specifications for the environment."""
    force_torque = mdp.DiffActionCfg(asset_name="robot", 
                                    command_type="CTBRController", 
                                    controller_cfg=CTBRControllerCfg(
                                        rate_gain_p=[35, 35, 35],
                                        rate_gain_i=[0.0, 0.0, 0.0],
                                        rate_gain_d=[0.0005, 0.0005, 0.0003],
                                        body_rate_bound=[-6, 6],
                                        thrust_ctrl_delay=0.03,
                                        torque_ctrl_delay=(0.03, 0.03, 0.03),
                                    ), 
                                    random_drag=False,
                                    action_lag=1,
                                    sim2real_test=SIM2REAL_TEST)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        last_action = ObsTerm(func=mdp.modified_last_action, params={"action_name": "force_torque"})
        base_orientation = ObsTerm(func=mdp.base_orientation_q)
        # base_pos = ObsTerm(func= mdp.root_pos_w)
        desired_pos_b = ObsTerm(
            func = mdp.desired_position_b,
            params = {
                "command_name":"desired_pos_b"
            })

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    
    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1),"z":(1.0, 2.0), "yaw": (-3.14, 3.14)},
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
    move_towards = RewTerm(
        func=mdp.target_reward,
        weight=1.0,
        params={"command_name": "desired_pos_b"}
    )

    orientation_reward = RewTerm(
        func=mdp.orientation_reward,
        weight=0.5,
    )

    move_in_dir = RewTerm(
        func=mdp.move_in_dir,
        weight=1.0,
        params={"threshold": 0.4}
    )

    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.001,
    )

    # time_penalty = RewTerm(
    #     func=mdp.time_penalty,
    #     weight=-0.1
    # )

    reach_target = RewTerm(
        func=mdp.reach_target,
        weight=10.0,
        params={"threshold": 0.1}
    )

    smooth_ang_vel = RewTerm(
        func=mdp.ang_vel_reward,
        weight=-0.001
    )

    smooth_lin_acc = RewTerm(
        func=mdp.body_lin_acc_l2,
        weight=-0.001
    )

    smooth_ang_acc = RewTerm(
        func=mdp.body_ang_acc_l2,
        weight=-0.0001
    )

    early_termination = RewTerm(
        func=mdp.is_terminated,
        weight=-200
    )

    hover_state = RewTerm(
        func=mdp.hover_state,
        weight= 1.0,
        params={"threshold":0.2, "ratio": 0.2}
    )

@configclass
class LossesCfg:
    move_towards_goal = LossTerm(
        func=mdp.target_diff,
        weight=1.0,
        params={"command_name": "desired_pos_b"},
        use_diff_states=True,
    )

    orientation_tracking = LossTerm(
        func=mdp.orientation_diff,
        weight=1.0,
        use_diff_states=True
    )

    move_in_dir = LossTerm(
        func=mdp.move_in_dir_diff,
        weight=1.0,
        use_diff_states=True,
        params={"command_name": "desired_pos_b", "threshold": 0.1}
    )

    smooth_vel = LossTerm(
        func=mdp.smooth_vel_diff,
        weight=0.1,
        use_diff_states=True,
        params={"ratio": 0.5}
    )

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # outofbound = DoneTerm(func=mdp.out_of_bound)
    base_contact = DoneTerm(
         func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="body"), "threshold": 1.0},
    )

@configclass
class CurriculumCfg:
    pass

@configclass
class QuadcopterReachTargetCTBREnvCfg(ManagerBasedDiffRLEnvCfg):
    scene: SceneCfg = SceneCfg(num_envs=2048, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    losses: LossesCfg = LossesCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:

        """Post initialization."""
        self.is_differentiable_physics = True
        
        # general settings
        self.decimation = 6
        self.episode_length_s = 60.0 if SIM2REAL_TEST else 6.0
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        # self.sim.disable_contact_processing=True
        if hasattr(self.scene, "terrain"):
            self.sim.physics_material = self.scene.terrain.physics_material

