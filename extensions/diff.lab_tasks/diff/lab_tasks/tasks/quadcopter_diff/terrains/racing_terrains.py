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
# *  Data: 2025/03/01     	             *
# *  Contact: yu-feng@sjtu.edu.cn                                               *
# *  Description: None                                                          *
# *******************************************************************************
from diff.lab.terrains.trimesh import CircleRacingTrackTerrainCfg, SquareRacingTrackTerrainCfg, FigureEightTrackTerrainCfg, ZigzagRacingTerrainCfg, EllipseRacingTerrainCfg
from omni.isaac.lab.terrains import TerrainGeneratorCfg

RacingTerrainWOObsPPOCfg = TerrainGeneratorCfg(
    seed=42,
    size=(14.0, 14.0),
    num_rows=10,
    num_cols=10,
    border_width=20.0,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=True,
    sub_terrains={
        "circular": CircleRacingTrackTerrainCfg(
            proportion=1.0,
            radius=4.0,
            num_gate=6,
            gate_size=[0.8, 1.2],
            gate_thickness=[0.08, 0.12],
            pos_noise_scale=[0.5, 1.0],
            rot_noise_scale=[0.0, 30.0],
            only_yaw=True,
            num_wall_seg=[1, 5],
            wall_size=[0.4, 1.0],
            wall_thickness=[0.04, 0.08],
            num_orbit_seg=[1, 5],
            add_border=False,
            add_obs=False
        )   # type: ignore
    },
)

RacingTerrainWObsPPOCfg = TerrainGeneratorCfg(
    seed=42,
    size=(14.0, 14.0),
    num_rows=10,
    num_cols=10,
    border_width=20.0,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=True,
    sub_terrains={
        "circular": CircleRacingTrackTerrainCfg(
            proportion=1.0,
            radius=5.0,
            num_gate=6,
            gate_size=[0.8, 1.2],
            gate_thickness=[0.08, 0.12],
            pos_noise_scale=[0.5, 1.0],
            rot_noise_scale=[0.0, 30.0],
            only_yaw=True,
            num_wall_seg=[2, 4],
            wall_size=[0.4, 1.0],
            wall_thickness=[0.04, 0.08],
            num_orbit_seg=[2, 4],
            add_border=False,
            add_obs=True,
            add_ground_obs=True,
            num_ground_obs=[1, 4]
        )   # type: ignore
    },
)

RacingSquareTerrainCfg = TerrainGeneratorCfg(
    seed=42,
    size=(14.0, 14.0),
    num_rows=10,
    num_cols=20,
    border_width=20.0,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=True,
    sub_terrains={
        "circular": SquareRacingTrackTerrainCfg(
            proportion=1.0,
            radius=[2.0, 5.0],
            num_gate=4,
            gate_size=[0.8, 1.2],
            gate_thickness=[0.03, 0.06],
            pos_noise_scale=[0.2, 1.0],
            rot_noise_scale=[0.0, 30.0],
            only_yaw=True,
            num_wall_seg=[1, 4],
            wall_size=[0.4, 1.0],
            wall_thickness=[0.04, 0.08],
            num_orbit_seg=[1, 4],
            add_border=False,
            add_obs=True,
            add_ground_obs=True,
            num_ground_obs=[1, 4],
            adj_dir_shift_prop=[0.2, 0.6],
            radius_dir_shift_prop=[0.5, 1.0],
        )   # type: ignore
    },
)

RacingTestTerrainCfg = TerrainGeneratorCfg(
    seed=42,
    size=(18.0, 18.0),
    num_rows=1,
    num_cols=1,
    border_width=20.0,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "circular": FigureEightTrackTerrainCfg(
            proportion=1.0,
            gate_size=[0.8, 1.2],
            gate_thickness=[0.08, 0.12],
            pos_noise_scale=[0.0, 0.0],
            rot_noise_scale=[0.0, 0.0],
            only_yaw=True,
        )   # type: ignore
    },
)

# NOTE num_gate should be same
RacingComplexTerrainCfg = TerrainGeneratorCfg(
    seed=42,
    size=(40.0, 40.0),
    num_rows=10,
    num_cols=20,
    border_width=20.0,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "zigzag": ZigzagRacingTerrainCfg(
            proportion=0.3,
            track_length=35.0,
            num_gate=8,
            gate_size=[0.8, 1.2],
            gate_thickness=[0.03, 0.06],
            pos_noise_scale=[1.0, 4.0],
            pos_z_noise_scale=[0.1, 1.0],
            rot_noise_scale=[0.0, 30.0],
            only_yaw=True,
            num_wall_seg=[2, 6],
            wall_size=[0.4, 1.0],
            wall_thickness=[0.04, 0.08],
            num_orbit_seg=[2, 6],
            add_border=False,
            add_obs=True,
            add_ground_obs=True,
            num_ground_obs=[1, 4],
            adj_dir_shift_prop=[0.6, 0.6],
            radius_dir_shift_prop=[6, 6],
            no_obs_range = 1.5
        ),   # type: ignore
        "circular": SquareRacingTrackTerrainCfg(
            proportion=0.3,
            radius=[5.0, 8.0],
            num_gate=8,
            gate_size=[0.8, 1.2],
            gate_thickness=[0.03, 0.06],
            pos_noise_scale=[0.2, 1.0],
            rot_noise_scale=[0.0, 30.0],
            only_yaw=True,
            num_wall_seg=[1, 4],
            wall_size=[0.4, 1.0],
            wall_thickness=[0.04, 0.08],
            num_orbit_seg=[1, 4],
            add_border=False,
            add_obs=True,
            add_ground_obs=True,
            num_ground_obs=[1, 4],
            adj_dir_shift_prop=[0.6, 0.6],
            radius_dir_shift_prop=[0.5, 0.5],
        ),   # type: ignore
        "ellipse": EllipseRacingTerrainCfg(
            proportion=0.4,
            gate_distance=5.0,
            num_gate=8,
            gate_size=[0.8, 1.2],
            gate_thickness=[0.03, 0.06],
            pos_noise_scale=[0.2, 1.0],
            rot_noise_scale=[0.0, 30.0],
            only_yaw=True,
            num_wall_seg=[1, 4],
            wall_size=[0.4, 1.0],
            wall_thickness=[0.04, 0.08],
            num_orbit_seg=[1, 4],
            add_border=False,
            add_obs=True,
            add_ground_obs=True,
            num_ground_obs=[1, 2],
            adj_dir_shift_prop=[0.6, 0.6],
            radius_dir_shift_prop=[0.5, 0.5],
        )   # type: ignore
    },
)