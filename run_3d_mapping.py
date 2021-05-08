#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create a 3D mapping of a room from monocular camera and IMU

Get the orientation of the camera from an Arduino that output's gyro data
Infer a Depth Estimation model on the camera at specific orientations
Project the points of the scene according to the depth and the orientation

Prerequisite:
    - Connect an USB camera
    - Connect an Arduino with the code from get_IMU_output_from_Arduino
"""

import argparse
import os
from threading import Thread
import re
import time

import cv2
import matplotlib
matplotlib.interactive(True)

import numpy as np
import serial

from utils import depth_manager
from utils import projections


# initialize x orientation
X_ORIENTATION = 0

# 119 got from Arduino with IMU.gyroscopeSampleRate();
GYROSCOPE_SAMPLE_RATE = 119


def parse_serial(serial_msg):
    """
    Function to parse serial data to extract float values
    format 'x:19.34 y:23.01 z:-33.83' to x, y, z float values

    Args:
        - (str) string with format 'x:19.34 y:23.01 z:-33.83'
    Return:
        - (list) x, y, z float values
    """
    xyz_list = re.findall('[-+]?[0-9]*\.?[0-9]*', serial_msg)
    return [float(i) for i in filter(lambda item: item, xyz_list)]


def update_orientation(ser):
    """
    Function to integration the x data from the Gyroscope
    and update the global variable x_orientation with the new value

    Args:
        - (serial.Serial) serial to get the gyroscope data
    """
    global X_ORIENTATION

    while True:
        serial_msg_bytes = ser.readline()
        serial_msg = serial_msg_bytes.decode()
        dx, dy, dz = parse_serial(serial_msg)

        # The gyroscope values are in degrees-per-second
        # divide each value by the number of samples per second
        dx_normalized = dx / GYROSCOPE_SAMPLE_RATE

        # remove noise
        if abs(dx_normalized) > 0.004:
            # update orientation
            X_ORIENTATION = X_ORIENTATION - dx_normalized*1.25
            X_ORIENTATION = X_ORIENTATION%360


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--permil_to_project",
        required=False,
        default=1,
        type=int,
        help="per-mil of depth points to project"
    )
    parser.add_argument(
        "-d",
        "--degrees_interval",
        required=False,
        default=30,
        type=int,
        help="Do projection every N degrees: for N=60 projection will appear\
            at orientations [0, 60, 120, 180, 240, 300]. Default is 30"
    )
    parser.add_argument(
        "-m",
        "--percentage_margin",
        required=False,
        default=2,
        type=int,
        help="Use N% margin on depth map to avoid potential outliers"
    )

    args = parser.parse_args()

    # connect to the Serial
    serial_connection = serial.Serial('COM3', 9600)
    time.sleep(2)

    # run the thread to update the x orientation in real time
    Thread(target=update_orientation, args=(serial_connection,)).start()

    interpreter = depth_manager.get_tflite_interpreter(
        "https://tfhub.dev/intel/lite-model/midas/v2_1_small/1/lite/1?lite-format=tflite",
        os.path.dirname(os.path.realpath(__file__)) + "/model/midas_v2_1_small.tflite")

    vid = cv2.VideoCapture(0)
    # get first image for depth calculation
    ret, frame = vid.read()
    rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    fig_simulation = matplotlib.pyplot.figure()

    points_in_3d = np.array([])
    depth_values = []

    # Orientations todo: orientations between 0:360 with degrees_interval steps
    orientations_done = []
    orientations_todo = [orientation for orientation
                         in range(0, 359, args.degrees_interval)]

    depth_map = depth_manager.run_tflite_interpreter(rgb_img, interpreter)

    # overlaps with 0.6 alpha between image and depth
    overlaps_img_depth = {}

    # distance of the corners for each depthmap for rescaling purpose
    # it avoid having a discontinuous 3D scene
    corners_distance = {}

    try:
        while True:
            ret, frame = vid.read()
            rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if ret:
                depth_map, points_in_3d, depth_values = \
                    projections.plot_env(
                        fig_simulation, X_ORIENTATION, points_in_3d, depth_values,
                        rgb_img, interpreter, orientations_done,
                        orientations_todo, depth_map, overlaps_img_depth,
                        corners_distance, per_mil_to_keep=args.permil_to_project,
                        percentage_margin_on_depth=args.percentage_margin)

            # stop if all todo orientations were done
            if not orientations_todo:
                break
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
