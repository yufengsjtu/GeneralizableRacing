# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

from diff.lab.terrains.trimesh import CircleRacingTrackTerrainCfg
from omni.isaac.lab.terrains import TerrainGeneratorCfg

RACINGTERRAIN_CFG = TerrainGeneratorCfg(
    size=(14.0, 14.0),
    num_rows=5,
    num_cols=5,
    border_width=20.0,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum= True,
    sub_terrains={
        "circular": CircleRacingTrackTerrainCfg(
            proportion=1.0,
            radius=4.0,
            num_gate=6,
            gate_size=[0.4, 0.5],
            gate_thickness=[0.04, 0.08],
            pos_noise_scale=[0.1, 1.0],
            rot_noise_scale=[0.0, 30.0],
            only_yaw=True,
            num_wall_seg=[1, 5],
            wall_size=[0.4, 1.0],
            wall_thickness=[0.04, 0.08],
            num_orbit_seg=[1, 5],
            add_border=False,
            add_obs=True
        )   # type: ignore
    },
)
"""Rough terrains configuration."""
