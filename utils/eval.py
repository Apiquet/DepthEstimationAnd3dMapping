#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Function to evaluation segmentation module
"""

import cv2
from glob import glob
import imageio
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import random
import tensorflow as tf
from tqdm import tqdm


nb_colors = 100
COLORS = [(random.randint(50, 200),
           random.randint(50, 200),
           random.randint(50, 200)) for i in range(nb_colors)]


def overlapImgWithSegMap(img, module_output):
    """
    Function to overlap segmentation map with image

    Args:
        - (cv2.image) Raw input image in RGB
        - (numpy array) module_output : output of the model
        - (int) legend_size: factor to multiply legend sized calculated
    """
    origin_height, origin_width, _ = img.shape
    im_pil = Image.fromarray(img)

    depth_min = module_output.min()
    depth_max = module_output.max()
    depth_rescaled = (255 * (module_output - depth_min) / (depth_max - depth_min)).astype("uint8")
    depth_rescaled_3chn = gray = cv2.cvtColor(depth_rescaled, cv2.COLOR_GRAY2RGB)
    module_output_3chn = cv2.applyColorMap(depth_rescaled_3chn, cv2.COLORMAP_RAINBOW)
    module_output_3chn = cv2.resize(module_output_3chn,
                               (origin_width, origin_height),
                               interpolation=cv2.INTER_CUBIC)
    seg_pil = Image.fromarray(module_output_3chn.astype('uint8'), 'RGB')

    overlap = Image.blend(im_pil, seg_pil, alpha=0.6)

    return overlap

def preprocessImage(rgb_img, resize_shape=[256,256]):
    """
    Function to preprocess an image
    Normalization and resized to resize_shape

    Args:
        - (str) Path to the image
        - (numpy array) module_output : output of the model
        - (int) legend_size: factor to multiply legend sized calculated
    """
    normalized_img = rgb_img / 255.0

    img_resized = tf.image.resize(normalized_img, resize_shape,
                                  method='bicubic',
                                  preserve_aspect_ratio=False)
    img_resized = tf.transpose(img_resized, [2, 0, 1])
    img_input = img_resized.numpy()
    reshape_img = img_input.reshape(1, 3, resize_shape[0], resize_shape[1])
    tensor = tf.convert_to_tensor(reshape_img, dtype=tf.float32)
    return rgb_img, tensor


def pltPredOnImg(module, image, signature='serving_default',
                 save_path=None, plot_img=True):
    """
    Function to infer a module on an image
    Display overlap original image + segmentation map

    Args:
        - module from TensorFlow Hub
        - (str) image path
        - (str) path to save image result
    """
    fig = plt.figure(figsize=(8, 8))

    rgb_img, module_input = preprocessImage(image)
    module_output = module.signatures[signature](module_input)
    np_module_output = module_output['default'].numpy().squeeze()
    overlap = overlapImgWithSegMap(rgb_img, np_module_output)

    if plot_img:
        plt.show(overlap)

    if save_path is not None:
        plt.savefig(save_path)
    plt.close()
    return overlap
