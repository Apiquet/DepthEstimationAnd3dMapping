# Depth estimation model

## Some research

The models were found on the KITTI dataset benchmark: http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction

### Metrics used in KITTI dataset benchmark

* SILog error : "defined to measure the relationship between points in the scene irrespective of the absolute global scale" (from https://arxiv.org/pdf/2009.09934.pdf)

<img src="https://render.githubusercontent.com/render/math?math=SILog=\frac{1}{T}\sum_{i}^{}d_{i}^{2}-\frac{1}{T^{2}}(\sum_{i}^{}d_{i})^{2}\text{,%20with%20}d_{i}=log(y_{i})-log(y_{i}^{*})">

* sqErrorRel : 

<img src="https://render.githubusercontent.com/render/math?math=SRE=\sqrt{\frac{1}{T}\sum_{i}^{}}\frac{\left|\left|y_{i}-y_{i}^{*}\right|\right|^{2}}{y_{i}^{*}}">


* iRMSE :

<img src="https://render.githubusercontent.com/render/math?math=iRMSE=\sqrt{\frac{1}{\left|N\right|}\sum_{i\in%20N}^{}\left|\frac{1}{d_{i}}-\frac{1}{d_{i}^{*}}\right|}">

### Some papers

* [ViP-DeepLab](https://arxiv.org/pdf/2012.05258.pdf): Learning Visual Perception with Depth-aware Video Panoptic
Segmentation; 9 Dec 2020

  * The paper approaches the problem by jointly performing monocular depth estimation and video panoptic segmentation.

  * They introduce a new evaluation metrics called depth-aware Video Panoptic Quality (DVPQ) (datasets hard to collect: need special depth sensors and huge amount of labeling efforts)

  * Create a way to convert existing datasets into DVPS (Depth-aware Video Panoptic Segmentation) datasets. They produceed two datasets: 

    * Cityscapes-DVPS derived from [Cityscapes-VPS](https://paperswithcode.com/dataset/cityscapes-vps) with depth annotations 

    * SemKITTI-DVPS derived from [SemanticKITTI](http://www.semantic-kitti.org/) with a projection of its annotated 3D point clouds to the image plane

  * First sub-task video panoptic segmentation: unifies semantic segmentation and instance segmentation (assign a semantic label and an ID to each pixel). Each instance should have the same ID throughout the video sequence (model should be able to track objects)

    * The video panoptic segmentation is also divided into 3 parts: semantic segmentation, center prediction and center regression

  * This model also outperform in MOTS challenge (Multi-Object TRacking and Segmentation)

  * The second sub-task is monocular depth estimation

  * Model takes image t and t+1 concatenated horizontally as input

    * Both images get encoded by the same encoder

    * depth estimation, semantic prediction, instance center prediction are performed on image t

    * instance center regression is done on image t+1 wrt image t

  * mask IoU between regression pairs are used to propagate IDs to the next inference (largest mask IoU to propagate and new ID if an object has 0 IoU with a previous inference)

  * monocular depth estimation: 

    * dense regression problem with an estimated depth for each pixel

    * depth = Max Depth x Sigmoid and max depth = 88 for the range KITTI dataset

    * special training loss which combines scale invariant logarithmic error and relative squared error

  * Dataset:

    * annotation for each pixel (c, id, d) for semantic class, instance ID and depth (the model outputs the same format for each pixel)


* [Patchwork](https://arxiv.org/pdf/1904.01784.pdf): A Patch-wise Attention Network for
Efficient Object Detection and Segmentation in Video Streams

  * they explore the idea of hard attention aimed for latency-sensitive applications
  
  * their method selects and only processes a small sub-window of the frame
  
  * the technique then makes predictions for the full frame based on the subwindows from previous frames and the update from the current sub-window
  
  *  use of Q-learning based policy training strategy that enables our approach to intelligently select the sub-windows such that the staleness in the memory hurts the performance the least
  
  *  Q-learningbased policy training strategy that enables our approach to intelligently select the sub-windows such that the staleness in the memory hurts the performance the least
  
  * attention window is parameterized by its center and size

* [Monodepth2](https://arxiv.org/pdf/1806.01260.pdf): Digging Into Self-Supervised Monocular Depth Estimation

  * a minimum reprojection loss (designed to robustly handle occlusions)
  
  * a full-resolution multi-scale sampling method to reduce visual artifacts
  
  * auto-masking loss to ignore training pixels that violate camera motion assumption
  
  * Demonstrate the added value of each of the 3 previous components separatly
