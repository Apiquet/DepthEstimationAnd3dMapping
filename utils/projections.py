#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Functions to manage the 3d simulation
"""

import math
import os
import random

from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np

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
    global IMAGE_WIDTH
    global IMAGE_HEIGHT
    global H_FOV_DEGREES
    global H_FOV_RAD
    global V_FOV_RAD
    global X_FOCAL
    global Y_FOCAL
    global X_CENTER_COORDINATE
    global Y_CENTER_COORDINATE

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


def get_3d_points_from_depthmap(points_in_3d, depth_values,
                                depth_map, x_orientation_degrees,
                                per_mil_to_keep=1):
    """
    Project depth values into 3D point according to the robot orientation
    Uses global variable x_orientation

    Args:
        - (np.array) points_in_3d array to add new 3D points
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
            points_in_3d = np.append(points_in_3d, point_3d_after_rotation)
            depth_values.append(depth_value)
    return points_in_3d, depth_values


def get_3d_pos_from_x_orientation(x_orientation, norm=1):
    """
    Get a 3d position x, y, z for a specific x orientation in degrees

    Args:
        - (float) orientation around x axis
        - (float) norm to rescale output vector
    Return:
        - (float) x position [0; 1] * norm
        - (float) y position [0; 1] * norm
        - (float) z position [0; 1] * norm
    """
    x_orientation_rad = math.radians(x_orientation)
    x_pos = 0
    y_pos = -math.sin(x_orientation_rad)
    z_pos = math.cos(x_orientation_rad)
    return x_pos*norm, y_pos*norm, z_pos*norm


def get_closest_corner(orientation, corners_distance):
    """
    Function to call to know which corner to rescale for a given orientation
    And the value to use

    Args:
        - (float) current orientation for a depth map to rescale
        - (dict of tuples 2 values) format orientation: (top left, top right)
    Return:
        - (bool) is_left true if left, false if right
        - (float) value for the rescale
    """
    is_left = True

    closest_orientation = min(corners_distance.keys(),
                              key=lambda x: abs(x-orientation))

    if closest_orientation < orientation:
        is_left = False

    return is_left, corners_distance[closest_orientation][is_left]


def plot_arrow_text(ax, x0_1, x0_2, x0_3, x1_1, x1_2, x1_3, dist, txt, c, s,
                    txt_offset=None):
    """
    Function to plot an arrow and a text at the end of it

    Args:
        - (matplotlib axis) ax to plot arrows
        - (float) x0_1, x0_2, x0_3: arrow origin
        - (float) x1_1, x1_2, x1_3: direction coordinate
        - (float) dist: norm of the arrow
        - (str) txt: text to print on arrow end
        - (str) c: color
        - (int) s: size
        - (list) txt_offset: offset on text position, format [x, y, z]
    """
    if txt_offset is None:
        txt_offset=[0, 0, 0]

    ax.quiver(x0_1, x0_2, x0_3, x1_1, x1_2, x1_3,
              length=dist, normalize=True, color=c)
    ax.text(x1_1*dist+txt_offset[0], x1_2*dist+txt_offset[1],
            x1_3*dist+txt_offset[2], txt, color=c, size=s)


def plot_referential(ax, max_projection_value, x_orientation=None):
    """
    Function to plot:
        - x, y, z arrows for the referential
        - the robot arrow
        - the arrows for todo orientations (red) and orientations done (green)

    Args:
        - (matplotlib axis) ax to plot arrows
        - (float) max_projection_value max depth value
        - (float) orientation of the robot (not required)
    """
    # plot origin as blue sphere
    ax.scatter(0, 0, s=100, c='b')

    # plot x, y, z referential with arrows
    plot_arrow_text(ax, 0, 0, 0, 1, 0, 0, max_projection_value/4, 'x', 'c', 15)
    plot_arrow_text(ax, 0, 0, 0, 0, 1, 0, max_projection_value/4, 'y', 'm', 15)
    plot_arrow_text(ax, 0, 0, 0, 0, 0, 1, max_projection_value/4, 'z', 'b', 15)

    if x_orientation:
        # get robot orientation in real referential
        x_pos, y_pos, z_pos = get_3d_pos_from_x_orientation(x_orientation)

        # plot arrow for robot orientation in simulation referential (-z, y, x)
        plot_arrow_text(ax, 0, 0, 0, -z_pos, y_pos, x_pos,
                        max_projection_value/3, 'robot', 'black', 15)

    ax.set_xlim(-max_projection_value*0.7, max_projection_value*0.7)
    ax.set_ylim(-max_projection_value*0.7, max_projection_value*0.7)
    ax.set_zlim(-max_projection_value*0.7, max_projection_value*0.7)


def plot_2d_top_view_referential(ax, x_orientation, orientations_todo,
                                 orientations_done):
    """
    Function to plot:
        - the robot arrow in black
        - the arrows for todo orientations (red) and orientations done (green)

    Args:
        - (matplotlib ax3D) ax to plot arrows
        - (float) orientation of the robot in 3D
        - (list) orientations_done list of orientations already projected
        - (list) orientations_todo list of orientations to project
    """
    # plot origin as blue sphere
    ax.scatter(0, 0, s=100, c='b')

    # orientations to do
    for orientation in orientations_todo:
        _, y_pos, z_pos = get_3d_pos_from_x_orientation(orientation)
        # simulation referential (-z, y, x)
        ax.arrow(0, 0, -z_pos, y_pos, head_width=0.05, head_length=0.1, color='r')
        ax.text(-z_pos, y_pos, str(orientation)+'°', color='r', size=15)

    # orientations to do
    for orientation in orientations_done:
        x_pos, y_pos, z_pos = get_3d_pos_from_x_orientation(orientation)
        # simulation referential (-z, y, x)
        ax.arrow(0, 0, -z_pos, y_pos, head_width=0.05, head_length=0.1, color='g')
        ax.text(-z_pos, y_pos, str(orientation)+'°', color='g', size=15)

    # get robot orientation in real referential
    x_pos, y_pos, z_pos = get_3d_pos_from_x_orientation(x_orientation, norm=1.5)

    # plot arrow for robot orientation in simulation referential (-z, y, x)
    ax.arrow(0, 0, -z_pos, y_pos, head_width=0.05, head_length=0.2, color='black')
    ax.text(-z_pos, y_pos, 'robot', color='black', size=15)

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)


def plot_3d_points(ax, points_in_3d, depth_values,
                   max_projection_value):
    """
    Function to plot 3d points in environment with a colormap from get_cmap()

    Args:
        - (matplotlib ax3D) ax to plot arrows
        - (np.array) points_in_3d to display in the 3D env
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
    points_in_3d = points_in_3d.reshape([-1, 3])
    ax.scatter(-points_in_3d[:, 2], points_in_3d[:, 1],
               points_in_3d[:, 0], c=colormap, s=5)


def save_3d_scene(path, points_in_3d, depth_values, images=None,
                  depth_maps=None, overlaps=None):
    """
    Function save numpy arrays of the 3D scene

    Args:
        - (str) path to save 3D scene
        - (np.array) points_in_3d to display in the 3D env
        - (list) depth_values to calculate cmap and boundaries
        - (dict of orientation: numpy.array) images used for projections
        - (dict of orientation: numpy.array) depth_maps used for projections
        - (dict of orientation: numpy.array) overlap between img and depth_map
    """
    os.makedirs(path, exist_ok=True)
    np.savetxt(path + 'points_in_3d.txt', points_in_3d, fmt='%f')
    np.savetxt(path + 'depth_values.txt', depth_values, fmt='%f')

    if images is not None:
        np.save(path + 'images.npy', images)

    if depth_maps is not None:
        np.save(path + 'depth_maps.npy', depth_maps)

    if overlaps is not None:
        np.save(path + 'overlaps.npy', overlaps)


def load_3d_scene(path):
    """
    Function to load saved 3D scene with save_3d_scene() function

    Args:
        - (str) path of the saved 3D scene
    Return:
        - (np.array) points_in_3d to display in the 3D env
        - (np.array) depth_values to calculate cmap and boundaries
        The next returned values are None if path/object does not exist
        - (dict of orientation: numpy.array) images used for projections
        - (dict of orientation: numpy.array) depth_maps used for projections
        - (dict of orientation: numpy.array) overlap between img and depth_map
    """
    points_in_3d = np.loadtxt(path + 'points_in_3d.txt', dtype=float)
    depth_values = np.loadtxt(path + 'depth_values.txt', dtype=float)

    images, depth_maps, overlaps = None, None, None

    if os.path.exists(path + 'images.npy'):
        images = np.load(path + 'images.npy', allow_pickle='TRUE').item()

    if os.path.exists(path + 'depth_maps.npy'):
        depth_maps = np.load(path + 'depth_maps.npy',
                             allow_pickle='TRUE').item()

    if os.path.exists(path + 'overlaps.npy'):
        overlaps = np.load(path + 'overlaps.npy', allow_pickle='TRUE').item()

    return points_in_3d, depth_values, images, depth_maps, overlaps


def plot_3d_scene(fig, points_in_3d, depth_values):
    """
    Function plot a 3D scene from points and depth values

    Args:
        - (matplotlib figure) fig
        - (np.array) points_in_3d to display in the 3D env
        - (list) depth_values to calculate cmap and boundaries
    """
    ax = fig.add_subplot(111, projection='3d')

    # get colormap
    max_projection_value = max(depth_values)
    depth_values_normalized = depth_values/max_projection_value
    colormap = get_cmap(depth_values_normalized)

    # plot referential x, y, z
    plot_referential(ax, max_projection_value)

    # plot 3D projected points in simulation referential (-z, y, x)
    points_in_3d = points_in_3d.reshape([-1, 3])
    ax.scatter(-points_in_3d[:, 2], points_in_3d[:, 1],
               points_in_3d[:, 0], c=colormap, s=5)

    plt.show()
    plt.pause(0.5)


def plot_env(fig, x_orientation, points_in_3d, depth_values, rgb_img,
             interpreter, orientations_done, orientations_todo,
             depth_map, overlaps_img_depth, images, depth_maps,
             corners_distance, max_projection_value=2., per_mil_to_keep=1,
             offset_ok=2.5, project_depth=True, percentage_margin_on_depth=0):
    """
    Project depth values into 3D point according to the robot orientation
    Uses global variable x_orientation

    Args:
        - (matplotlib.figure) fig to plot environment
        - (float) x_orientation of the robot
        - (np.array) points_in_3d to display in the 3D env
        - (list) depth_values to calculate cmap and boundaries
        - (cv2 image) image in rgb format
        - (tf.lite.Interpreter) tflite interpreter
        - (list) orientations_done list of orientations already projected
        - (list) orientations_todo list of orientations to project
        - (cv2 image) depth_map format (width, height, 1)
        - (dict of orientation: numpy.array) images used for projections
        - (dict of orientation: numpy.array) depth_maps used for projections
        - (dict of orientation: numpy.array) overlap between img and depth_map
        - (dict of tuples 2 values) format orientation: (top left, top right)
        - (float) max_projection_value max depth value
        - (int) per_mil_to_keep: per-mil of depth points to project
        - (float) offset to accept the current orientation of the robot to do
            and angle in orientations_todo
        - (boolean) project_depth enables depth calculation and projection
        - (float) margin percentage to remove from the depth [0: 100]
    """
    plt.gcf().clear()

    ax = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(224)

    if len(points_in_3d) > 0:
        max_projection_value = max(depth_values)
        plot_3d_points(ax, points_in_3d, depth_values,
                       max_projection_value)

    plot_referential(ax, max_projection_value, x_orientation=x_orientation)

    plot_2d_top_view_referential(ax3, x_orientation,
                                 orientations_todo, orientations_done)

    if project_depth:
        for i, orientation in enumerate(orientations_todo):
            if orientation - offset_ok <= x_orientation\
                    <= orientation + offset_ok:
                # get 3d points in real referential
                depth_map = depth_manager.run_tflite_interpreter(rgb_img,
                                                                 interpreter)
                overlap = depth_manager.overlap_img_with_segmap(rgb_img,
                                                                depth_map)
                images[orientation] = np.asarray(rgb_img, dtype="int32")
                depth_maps[orientation] = depth_map
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
                    is_left, corner_value =\
                        get_closest_corner(orientation, corners_distance)
                    depth_map = depth_map / depth_map[0, -int(not is_left)]\
                        * corner_value
                    corners_distance[orientation] =\
                        [depth_map[0, 0], depth_map[0, -1]]

                points_in_3d, depth_values = \
                    get_3d_points_from_depthmap(points_in_3d, depth_values,
                                                depth_map, x_orientation,
                                                per_mil_to_keep)
                orientations_done.append(orientation)
                del orientations_todo[i]
                break

    ax2.imshow(rgb_img)

    plt.show()
    plt.pause(0.2)

    return depth_map, points_in_3d, depth_values,\
        images, depth_maps, overlaps_img_depth
