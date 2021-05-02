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
import numpy as np
import random

from . import depth_manager


IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

H_FOV_DEGREES = 60
H_FOV_RAD = math.radians(H_FOV_DEGREES)
# v_fov is wrong but cannot find the real value on camera's documentation
V_FOV_RAD = math.radians(IMAGE_HEIGHT/IMAGE_WIDTH*H_FOV_DEGREES)

X_FOCAL = IMAGE_WIDTH / (2*math.tan(H_FOV_RAD/2))
Y_FOCAL = IMAGE_HEIGHT / (2*math.tan(V_FOV_RAD/2))

X_CENTER_COORDINATE = (0.5*IMAGE_WIDTH)
Y_CENTER_COORDINATE = (0.5*IMAGE_HEIGHT)


def init_camera_params(image_width, image_height, h_fov_degrees,
                       v_fov_degrees=None):
    """
    Function to initialize camera specs

    Args:
        - (int) image width
        - (int) image height
        - (float) horizontal fov in degrees
        - (float) vertical fov in degrees
    """
    IMAGE_WIDTH = image_width
    IMAGE_HEIGHT = image_height

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
                                per_mil_to_keep=1):
    """
    Project depth values into 3D point according to the robot orientation
    Uses global variable x_orientation

    Args:
        - (np.array) points_in_ned array to add new 3D points
        - (list) depth_values list to add the depth value of each point
        - (cv2 image) depth_map format (width, height, 1)
        - (float) x orientation of the robot in degrees
        - (int) per_mil_to_keep: per-mil of depth points to project
    Return:
        - (np.array) rotation matrix for a rotation around the x axis
    """
    depth_width, depth_height, _ = depth_map.shape

    x_depth_rescale_factor = depth_width / IMAGE_WIDTH
    y_depth_rescale_factor = depth_height / IMAGE_HEIGHT

    for x in range(IMAGE_WIDTH):
        for y in range(IMAGE_HEIGHT):

            # keep n per-mil points
            if random.randint(0, 999) >= per_mil_to_keep:
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
                              key=lambda x: abs(x-orientation))

    if closest_orientation < orientation:
        isLeft = False

    return isLeft, corners_distance[closest_orientation][isLeft]


def plot_arrow_text(ax, x0_1, x0_2, x0_3, x1_1, x1_2, x1_3, dist, txt, c, s,
                    txt_offset=[0, 0, 0]):
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
    ax.quiver(x0_1, x0_2, x0_3, x1_1, x1_2, x1_3,
              length=dist, normalize=True, color=c)
    ax.text(x1_1*dist+txt_offset[0], x1_2*dist+txt_offset[1],
            x1_3*dist+txt_offset[2], txt, color=c, size=s)


def plot_referential(ax, x_orientation, orientations_todo, orientations_done,
                     min_projection_value, max_projection_value):
    """
    Function to plot:
        - x, y, z arrows for the referential
        - the robot arrow
        - the arrows for todo orientations (red) and orientations done (green)

    Args:
        - (matplotlib ax3D) ax to plot arrows
        - (float) orientation of the robot in 3D
        - (list) orientations_done list of orientations already projected
        - (list) orientations_todo list of orientations to project
        - (float) min_projection_value min depth value
        - (float) max_projection_value max depth value
    Return:
        - (matplotlib ax3D) ax with the four arrows
    """
    # plot origin as blue sphere
    ax.scatter(0, 0, s=100, c='b')

    # plot x, y, z referential with arrows
    plot_arrow_text(ax, 0, 0, 0, 1, 0, 0, max_projection_value/4, 'x', 'c', 15)
    plot_arrow_text(ax, 0, 0, 0, 0, 1, 0, max_projection_value/4, 'y', 'm', 15)
    plot_arrow_text(ax, 0, 0, 0, 0, 0, 1, max_projection_value/4, 'z', 'b', 15)

    # orientations to do
    for orientation in orientations_todo:
        x_pos, y_pos, z_pos = get_3d_pos_from_x_orientation(orientation)
        x_origin_offset = -max_projection_value/2
        plot_arrow_text(ax, 0, 0, x_origin_offset, -z_pos, y_pos, x_pos,
                        max_projection_value/3, str(orientation)+'°', 'r', 15,
                        [0, 0, x_origin_offset])

    # orientations done
    for orientation in orientations_done:
        x_pos, y_pos, z_pos = get_3d_pos_from_x_orientation(orientation)
        x_origin_offset = -max_projection_value/2
        plot_arrow_text(ax, 0, 0, x_origin_offset, -z_pos, y_pos, x_pos,
                        max_projection_value/3, str(orientation)+'°', 'g', 15,
                        [0, 0, x_origin_offset])

    # get robot orientation in real referential
    x_pos, y_pos, z_pos = get_3d_pos_from_x_orientation(x_orientation)

    # plot arrow for robot orientation in simulation referential (-z, y, x)
    plot_arrow_text(ax, 0, 0, 0, -z_pos, y_pos, x_pos,
                    max_projection_value/3, 'robot', 'black', 15)

    ax.set_xlim(-max_projection_value*0.7, max_projection_value*0.7)
    ax.set_ylim(-max_projection_value*0.7, max_projection_value*0.7)
    ax.set_zlim(-max_projection_value*0.7, max_projection_value*0.7)


def plot_3d_points(ax, points_in_ned, depth_values,
                   max_projection_value):
    """
    Function to plot 3d points in environment with a colormap from get_cmap()

    Args:
        - (np.array) points_in_ned to display in the 3D env
        - (list) depth_values to calculate cmap and boundaries
        - (float) max_projection_value max depth values
    Return:
        - (bool) isLeft true if left, false if right
        - (float) value for the rescale
    """
    # get colormap
    depth_values_normalized = depth_values/max_projection_value
    colormap = get_cmap(depth_values_normalized)

    # plot 3D projected points in simulation referential (-z, y, x)
    points_in_ned = points_in_ned.reshape([-1, 3])
    ax.scatter(-points_in_ned[:, 2], points_in_ned[:, 1],
               points_in_ned[:, 0], c=colormap, s=5)


def plot_env(fig, x_orientation, points_in_ned, depth_values, rgb_img,
             interpreter, project_depth, orientations_done, orientations_todo,
             depth_map, overlaps_img_depth, corners_distance,
             min_projection_value=1., max_projection_value=2.,
             per_mil_to_keep=1, offset_ok=2.5,
             percentage_margin_on_depth=0):
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
        - (int) per_mil_to_keep: per-mil of depth points to project
        - (float) offset to accept the current orientation of the robot to do
            and angle in orientations_todo
        - (float) margin percentage to remove from the depth [0: 100]
    """
    plt.gcf().clear()

    ax = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)

    if len(points_in_ned) > 0:
        min_projection_value = min(depth_values)
        max_projection_value = max(depth_values)
        plot_3d_points(ax, points_in_ned, depth_values,
                       max_projection_value)

    plot_referential(ax, x_orientation, orientations_todo, orientations_done,
                     min_projection_value, max_projection_value)

    if project_depth:
        for i, orientation in enumerate(orientations_todo):
            if orientation - offset_ok <= x_orientation\
                    <= orientation + offset_ok:
                # get 3d points in real referential
                depth_map = depth_manager.run_tflite_interpreter(rgb_img,
                                                                 interpreter)
                overlap = depth_manager.overlap_img_with_segmap(rgb_img,
                                                                depth_map)
                overlaps_img_depth[orientation] = overlap

                ax2.imshow(overlap)
                plt.show()
                plt.pause(0.2)

                depth_map = depth_manager.crop_depth_map(
                    depth_map, percentage_margin_on_depth)

                min_dist = float(input("Min distance: "))
                max_dist = float(input("Max distance: "))

                depth_map = depth_manager.rescale_depth_map(
                    depth_map, min_dist, max_dist)

                if len(orientations_done) == 0:
                    corners_distance[orientation] =\
                        [depth_map[0, 0, 0], depth_map[0, -1, 0]]
                else:
                    isLeft, corner_value =\
                        get_closest_corner(orientation, corners_distance)
                    depth_map = depth_map / depth_map[0, -int(not isLeft)]\
                        * corner_value
                    corners_distance[orientation] =\
                        [depth_map[0, 0], depth_map[0, -1]]

                points_in_ned, depth_values = \
                    get_3d_points_from_depthmap(points_in_ned, depth_values,
                                                depth_map, x_orientation,
                                                per_mil_to_keep)
                orientations_done.append(orientation)
                del orientations_todo[i]
                break

    ax2.imshow(rgb_img)

    plt.show()
    plt.pause(0.2)

    return depth_map, points_in_ned, depth_values
