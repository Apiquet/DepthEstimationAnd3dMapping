#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Function to evaluation segmentation module
"""

import cv2
import math
import imageio
from matplotlib import pyplot as plt
from PIL import Image
import tensorflow as tf
from tqdm import tqdm
import os
from matplotlib import cm
import random
import numpy as np

from . import depth_manager


IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
DEPTH_WIDTH = 256
DEPTH_HEIGHT = 256

x_depth_rescale_factor = DEPTH_WIDTH / IMAGE_WIDTH
y_depth_rescale_factor = DEPTH_HEIGHT / IMAGE_HEIGHT

H_FOV_DEGREES = 60
H_FOV_RAD = math.radians(H_FOV_DEGREES)
# v_fov is wrong but cannot find the real value on camera's documentation
V_FOV_RAD = math.radians(IMAGE_HEIGHT/IMAGE_WIDTH*H_FOV_DEGREES)

X_FOCAL = IMAGE_WIDTH / (2*math.tan(H_FOV_RAD/2))
Y_FOCAL = IMAGE_HEIGHT / (2*math.tan(V_FOV_RAD/2))

X_CENTER_COORDINATE = (0.5*IMAGE_WIDTH)
Y_CENTER_COORDINATE = (0.5*IMAGE_HEIGHT)


def init_camera_params(image_width, image_height, depth_width, depth_height,
                       h_fov_degrees, v_fov_degrees=None):
    """
    Function to initialize camera specs

    Args:
        - (int) image width
        - (int) image height
        - (int) depth width
        - (int) depth height
        - (float) horizontal fov in degrees
        - (float) vertical fov in degrees
    """
    IMAGE_WIDTH = image_width
    IMAGE_HEIGHT = image_height
    DEPTH_WIDTH = depth_width
    DEPTH_HEIGHT = depth_height

    x_depth_rescale_factor = DEPTH_WIDTH / IMAGE_WIDTH
    y_depth_rescale_factor = DEPTH_HEIGHT / IMAGE_HEIGHT

    H_FOV_DEGREES = h_fov_degrees
    H_FOV_RAD = math.radians(H_FOV_DEGREES)
    if v_fov_degrees:
        V_FOV_RAD = math.radians(v_fov_degrees)
    else:
        # if v_fov not known
        V_FOV_RAD = math.radians(IMAGE_HEIGHT/IMAGE_WIDTH*H_FOV_DEGREES)

    X_FOCAL = IMAGE_WIDTH / (2*math.tan(H_FOV_RAD/2))
    Y_FOCAL = IMAGE_HEIGHT / (2*math.tan(V_FOV_RAD/2))

    X_CENTER_COORDINATE = (0.5*IMAGE_WIDTH)
    Y_CENTER_COORDINATE = (0.5*IMAGE_HEIGHT)


def get_cmap(values, cmap_name='rainbow'):
    """
    Function to get cmap according to values

    Args:
        - (str) tflite_model_url: URL of the model to download
        - (str) path_to_save: path to save the model
        - (bool) verbose: print model input and output details
    Return:
        - (tf.lite.Interpreter) tflite interpreter
    """
    cmap = cm.get_cmap(cmap_name, 12)
    depth_values_normalized = values/max(values)
    return cmap(depth_values_normalized)


def get_rotation_matrix(orientation):
    """
    Get the rotation matrix for a rotation around the x axis of n radians

    Args:
        - (float) orientation in radian
    Return:
        - (np.array) rotation matrix for a rotation around the x axis
    """
    rotation_matrix = np.array(
        [[1, 0, 0],
         [0, math.cos(orientation), -math.sin(orientation)],
         [0, math.sin(orientation), math.cos(orientation)]])
    return rotation_matrix


def get_3d_points_from_depthmap(points_in_ned, depth_values,
                                depth_map, x_orientation_degrees,
                                pourcentage_to_keep=1):
    """
    Project depth values into 3D point according to the robot orientation
    Uses global variable x_orientation

    Args:
        - (np.array) points_in_ned array to add new 3D points
        - (list) depth_values list to add the depth value of each point
        - (cv2 image) depth_map format (width, height, 1)
        - (float) x orientation of the robot in degrees
        - (int) pourcentage_to_keep: pourcentage of depth points to project
    Return:
        - (np.array) rotation matrix for a rotation around the x axis
    """
    for x in range(IMAGE_WIDTH):
        for y in range(IMAGE_HEIGHT):

            # keep 0.1% of the points
            if random.randint(0, 999) >= pourcentage_to_keep:
                continue

            # get depth value
            x_depth_pos = int(x*x_depth_rescale_factor)
            y_depth_pos = int(y*y_depth_rescale_factor)
            depth_value = depth_map[x_depth_pos, y_depth_pos, 0]

            # get 3d vector
            x_point = depth_value * (x - X_CENTER_COORDINATE) / X_FOCAL
            y_point = depth_value * (y - Y_CENTER_COORDINATE) / Y_FOCAL
            point_3d_before_rotation = np.array([x_point, y_point,
                                                 depth_value])

            # projection in function of the orientation
            point_3d_after_rotation = np.matmul(
                get_rotation_matrix(math.radians(x_orientation_degrees)),
                point_3d_before_rotation)
            points_in_ned = np.append(points_in_ned, point_3d_after_rotation)
            depth_values.append(depth_value)
    return points_in_ned, depth_values


def get_3d_pos_from_x_orientation(x_orientation):
    """
    Get a 3d position x, y, z for a specific x orientation in degrees

    Args:
        - (float) orientation around x axis
    Return:
        - (float) x position [0; 1]
        - (float) y position [0; 1]
        - (float) z position [0; 1]
    """
    x_orientation_rad = math.radians(x_orientation)
    x_pos = 0
    y_pos = -math.sin(x_orientation_rad)
    z_pos = math.cos(x_orientation_rad)
    return x_pos, y_pos, z_pos


def get_closest_corner(orientation, corners_distance):
    """
    Function to call to know which corner to rescale for a given orientation
    And the value to use

    Args:
        - (float) current orientation for a depth map to rescale
        - (dict of tuples 2 values) format orientation: (top left, top right)
    Return:
        - (bool) isLeft true if left, false if right
        - (float) value for the rescale
    """
    isLeft = True
    value = 0

    closest_orientation = min(corners_distance.keys(),
                              key=lambda x:abs(x-orientation))

    if closest_orientation < orientation:
        isLeft = False
    
    return isLeft, corners_distance[closest_orientation][isLeft]


def plot_env(fig, x_orientation, points_in_ned, depth_values, rgb_img,
             interpreter, project_depth, orientations_done, orientations_todo,
             depth_map, overlaps_img_depth, corners_distance,
             min_projection_value=1., max_projection_value=2.,
             pourcentage_to_project=1, offset_ok=5.,
             min_dist=None, max_dist=None):
    """
    Project depth values into 3D point according to the robot orientation
    Uses global variable x_orientation

    Args:
        - (matplotlib.figure) fig to plot environment
        - (float) x_orientation of the robot
        - (np.array) points_in_ned to display in the 3D env
        - (list) depth_values to calculate cmap and boundaries
        - (cv2 image) image in rgb format
        - (tf.lite.Interpreter) tflite interpreter
        - (boolean) project_depth enables depth calculation and projection
        - (list) orientations_done list of orientations already projected
        - (list) orientations_todo list of orientations to project
        - (cv2 image) depth_map format (width, height, 1)
        - (dict of orientation: PIL image) overlap between img and depth_map
        - (dict of tuples 2 values) format orientation: (top left, top right)
        - (float) min_projection_value min depth value
        - (float) max_projection_value max depth value
        - (int) pourcentage_to_keep: pourcentage of depth points to project
        - (float) offset to accept the current orientation of the robot to do
            and angle in orientations_todo
        - (float) min_dist to rescale the generated depth map
        - (float) max_dist to rescale the generated depth map
        If min/max_dist: depth=depth/depth.max*(max_dist-min_dist)+min_dist
    """
    plt.gcf().clear()

    ax = fig.add_subplot(111, projection='3d')

    if project_depth:
        for i, orientation in enumerate(orientations_todo):
            if orientation - offset_ok <= x_orientation\
                    <= orientation + offset_ok:
                # get 3d points in real referential
                depth_map = depth_manager.run_tflite_interpreter(rgb_img,
                                                                 interpreter)
                overlaps_img_depth[orientation] = \
                    depth_manager.overlap_img_with_segmap(rgb_img, depth_map)

                if min_dist is not None and max_dist is not None:
                    depth_max = depth_map.max()
                    total_range = max_dist - min_dist
                    depth_map = depth_map / depth_max * total_range + min_dist
                    corners_distance[orientation] =\
                        [depth_map[0, 0], depth_map[0, -1]]
                else:
                    isLeft, corner_value =\
                        get_closest_corner(orientation, corners_distance)
                    depth_map = depth_map / depth_map[0, -int(not isLeft)]\
                        * total_range + min_dist
                    corners_distance[orientation] =\
                        [depth_map[0, 0], depth_map[0, -1]]

                points_in_ned, depth_values = \
                    get_3d_points_from_depthmap(points_in_ned, depth_values,
                                                depth_map, x_orientation,
                                                pourcentage_to_project)
                orientations_done.append(orientation)
                del orientations_todo[i]
                break

    if len(points_in_ned) > 0:
        # get colormap
        min_projection_value = min(depth_values)
        max_projection_value = max(depth_values)
        depth_values_normalized = depth_values/max_projection_value
        colormap = get_cmap(depth_values_normalized)

        # plot 3D projected points in  simulation referential (-z, y, x)
        points_in_ned = points_in_ned.reshape([-1, 3])
        ax.scatter(-points_in_ned[:, 2], points_in_ned[:, 1],
                   points_in_ned[:, 0], c=colormap, s=5)

    # plot origin as blue sphere
    ax.scatter(0, 0, s=100, c='b')

    # plot x, y, z referential with arrows
    ax.quiver(0, 0, 0, 1, 0, 0,
              length=min_projection_value/2, normalize=True, color='c')
    ax.text(min_projection_value/2, 0, 0, "x", color='c', size=15)
    ax.quiver(0, 0, 0, 0, 1, 0,
              length=min_projection_value/2, normalize=True, color='m')
    ax.text(0, min_projection_value/2, 0, "y", color='m', size=15)
    ax.quiver(0, 0, 0, 0, 0, 1,
              length=min_projection_value/2, normalize=True, color='b')
    ax.text(0, 0, min_projection_value/2, "z", color='b', size=15)

    # orientations to do
    for orientation in orientations_todo:
        x_pos, y_pos, z_pos = get_3d_pos_from_x_orientation(orientation)
        ax.quiver(0, 0, 0, -z_pos, y_pos, x_pos,
                  length=min_projection_value, normalize=True, color='r')
        ax.text(-z_pos*min_projection_value, y_pos*min_projection_value,
                x_pos*min_projection_value, str(orientation)+'°', color='r',
                size=15)

    # orientations done
    for orientation in orientations_done:
        x_pos, y_pos, z_pos = get_3d_pos_from_x_orientation(orientation)
        ax.quiver(0, 0, 0, -z_pos, y_pos, x_pos,
                  length=min_projection_value, normalize=True, color='g')
        ax.text(-z_pos*min_projection_value, y_pos*min_projection_value,
                x_pos*min_projection_value, str(orientation)+'°', color='g',
                size=15)

    # get robot orientation in real referential
    x_pos, y_pos, z_pos = get_3d_pos_from_x_orientation(x_orientation)

    ax.view_init(elev=30, azim=10)
    ax.set_xlim(-max_projection_value*0.7, max_projection_value*0.7)
    ax.set_ylim(-max_projection_value*0.7, max_projection_value*0.7)
    ax.set_zlim(-max_projection_value*0.7, max_projection_value*0.7)

    # plot arrow for robot orientation in simulation referential (-z, y, x)
    ax.quiver(0, 0, 0, -z_pos, y_pos, x_pos,
              length=min_projection_value*0.7, normalize=True, color='black')
    ax.text(-z_pos*min_projection_value*0.7, y_pos*min_projection_value*0.7,
            x_pos*min_projection_value*0.7, 'robot', color='black',
            size=15)

    plt.show()
    plt.pause(0.1)

    return depth_map, points_in_ned, depth_values
