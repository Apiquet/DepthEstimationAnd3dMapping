# Depth estimation model

## Some research

The models were found on the kitti dataset benchmark: http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction

### Metrics used 

* SILog error : "defined to measure the relationship between points in the scene irrespective of the absolute global scale" (from https://arxiv.org/pdf/2009.09934.pdf)

<img src="https://render.githubusercontent.com/render/math?math=SILog=\frac{1}{T}\sum_{i}^{}d_{i}^{2}-\frac{1}{T^{2}}(\sum_{i}^{}d_{i})^{2}">

* sqErrorRel
* absErrorRel
* iRMSE

### Some papers

* ViP-DeepLab: Learning Visual Perception with Depth-aware Video Panoptic
Segmentation

* Patchwork: A Patch-wise Attention Network for
Efficient Object Detection and Segmentation in Video Streams
