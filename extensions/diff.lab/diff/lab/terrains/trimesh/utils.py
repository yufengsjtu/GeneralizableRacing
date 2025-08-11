# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import scipy.spatial.transform as tf
import trimesh
import random
def make_gate(outer_extents, inner_extents, position=[0, 0, 0], rotation_angles=[0, 0, 0])->trimesh.Trimesh:
    """
    create a gate trimesh, supporting to adjust the size, the position and rotation of the gate
    Args:
    - outer_extents: size of outer [width, height, depth]
    - inner_extents: size of inner [width, height, depth]
    - position: set position [x, y, z], default to [0, 0, 0]
    - rotation_angles: rotation angle [roll, pitch, yaw] (in degree), default to [0, 0, 0]

    Returns:
    -  a gate trimesh object
    """
    outer_box = trimesh.creation.box(extents=outer_extents)
    inner_box = trimesh.creation.box(extents=inner_extents)
    # apply difference
    gate = outer_box.difference(inner_box)
    # rotation matrix
    rotation_matrix = trimesh.transformations.euler_matrix(np.radians(rotation_angles[0]),
                                                           np.radians(rotation_angles[1]),
                                                           np.radians(rotation_angles[2]), 
                                                           'rxyz')
    gate.apply_transform(rotation_matrix)
    gate.apply_translation(position)
    return gate

def make_wall(size, position, euler):
    """
    create a wall trimesh, supporting to adjust the size, the position and rotation of the wall
    Args:
    - size: size of wall [width, height, depth]
    - position: set position [x, y, z]
    - euler: rotation angle [roll, pitch, yaw] (in degree)

    Returns:
    -  a wall trimesh object
    """
    wall = trimesh.creation.box(extents=size)
    # rotation matrix
    rotation_matrix = trimesh.transformations.euler_matrix(np.radians(euler[0]),
                                                           np.radians(euler[1]),
                                                           np.radians(euler[2]), 
                                                           'rxyz')
    wall.apply_transform(rotation_matrix)
    wall.apply_translation(position)
    return wall

def make_orbit(position, euler):
    prob = random.random()
    if prob < 0.2:     # box
        size = np.random.uniform(0.1, 0.5, 3)
        orbit = trimesh.creation.box(extents=size)
    elif prob >= 0.2 and prob < 0.4:        # cylinder
        radius = np.random.uniform(0.1, 0.3)
        height = np.random.uniform(0.2, 0.6)
        orbit = trimesh.creation.cylinder(radius=radius, height=height)
    elif prob >= 0.4 and prob < 0.6:        # sphere
        radius = np.random.uniform(0.1, 0.3)
        orbit = trimesh.creation.icosphere(radius=radius)
    elif prob >= 0.8 and prob < 0.8:       # cone 
        radius = np.random.uniform(0.1, 0.3)
        height = np.random.uniform(0.2, 0.6)
        orbit = trimesh.creation.cone(radius=radius, height=height)
    else:       # capsule
        radius = np.random.uniform(0.1, 0.3)
        height = np.random.uniform(0.2, 0.6)
        orbit = trimesh.creation.capsule(radius=radius, height=height)
    # rotation matrix
    rotation_matrix = trimesh.transformations.euler_matrix(np.radians(euler[0]),
                                                           np.radians(euler[1]),
                                                           np.radians(euler[2]), 
                                                           'rxyz')
    orbit.apply_transform(rotation_matrix)
    orbit.apply_translation(position)
    return orbit

def make_ground_high_obs(position, euler):
    height = 1.0 + random.uniform(0.0, 2.0)
    position[2] = height / 2
    prob = random.random()
    if prob < 0.5:
        # box
        size_xy = np.random.uniform(0.05, 1.0, 2)
        ground_obs = trimesh.creation.box(extents=[size_xy[0], size_xy[1], height])
    else:
        # cylinder
        radius = np.random.uniform(0.025, 0.5)
        ground_obs = trimesh.creation.cylinder(radius=radius, height=height)
    # rotation matrix
    # rotation_matrix = trimesh.transformations.euler_matrix(0,
    #                                                        np.radians(euler[2]),
    #                                                        0, 
    #                                                        'rxyz')
    # ground_obs.apply_transform(rotation_matrix)
    ground_obs.apply_translation(position)
    return ground_obs

def make_ground_little_obj(position, euler):
    prob = random.random()
    if prob < 0.33:
        # box
        size = np.random.uniform(0.1, 1.5, 3)
        position[2] = size[2] / 2 + random.uniform(-0.2, 0.5)
        ground_little_obj = trimesh.creation.box(extents=size)
    elif prob >= 0.33 and prob < 0.66:
        # cylinder
        radius = random.uniform(0.025, 0.5)
        height = random.uniform(0.1, 1.0)
        position[2] = height / 2 + random.uniform(-0.2, 0.5)
        ground_little_obj = trimesh.creation.cylinder(radius=radius, height=height) 
    else:
        # sphere
        radius = random.uniform(0.05, 0.5)
        position[2] = random.uniform(-radius, radius) + random.uniform(-0.2, 0.5)
        ground_little_obj = trimesh.creation.icosphere(radius=radius)
    # rotation matrix
    # rotation_matrix = trimesh.transformations.euler_matrix(0,
    #                                                        np.radians(euler[2]),
    #                                                        0, 
    #                                                        'rxyz')
    # ground_little_obj.apply_transform(rotation_matrix)
    ground_little_obj.apply_translation(position)
    return ground_little_obj

# test
# gate = create_gate(outer_extents=[2, 2, 0.1], inner_extents=[1.8, 1.8, 0.1], 
#                    position=[1, 2, 0], rotation_angles=[0, 0, 10])
# scene = trimesh.Scene(gate)
# scene.show()