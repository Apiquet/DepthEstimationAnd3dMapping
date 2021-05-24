# Depth estimation and 3D mapping

## Description

The project is described in detail [here](https://apiquet.com/2021/04/09/depth-estimation-and-3d-mapping/)

Depth estimation using a Deep Learning algorithm to map 3D points in the robot referential. An Arduino Nano is also used to get the orientation of the robot thanks to an IMU to project the 3D points to a 3D simulation view of the real world.

It can then be embedded in a robot to map its environment while rotating around a single axis.

## Hardware needed

This code was tested on a Jetson Nano connected to:

- an USD camera (Logitech C270),
- an Arduino Nano.

A piece was also printed with a 3D printer to put the electronic boards and camera together.

## Installation

- Build the code "get_IMU_output_from_Arduino" and flash it to the Arduino Nano
- Clone the repository to the Jetson Nano
- (optional) Print solidworks_parts/toprint_main_part.STL and toprint_rotation_piece.STL

## Usage

### Create a 3D view of a scene

``` shell
python3 run_3d_mapping -o path/to/save/3dscene/
```

### See a saved 3D scene

- Clone the repository on a computer
- run:

``` shell
python3 run_3d_mapping -s -o path/to/saved/3dscene/
```

An example is already saved in the repository, to see it:

``` shell
python3 run_3d_mapping -s -o repo/3d_scene/
```

The scene should look like:

![Image](images/projection_example.gif)

The 3D scene was done for the following room:

![Image](images/projection_example_explained.gif)
