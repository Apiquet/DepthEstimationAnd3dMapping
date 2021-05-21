# DepthEstimationAnd3DMapping

This project uses a Deep Learning algorithm for monocular depth estimation and an IMU to know the orientation of the camera.

It can be embedded in a robot to map its environment in a 3D simulation view.

The project is described [here](https://apiquet.com/2021/04/09/depth-estimation-and-3d-mapping/).

The code is embedded on a Jetson Nano (run_3d_mapping.py) connected to an Arduino Nano which sends the gyroscope data by serial connection (get_IMU_output_from_Arduino/).
