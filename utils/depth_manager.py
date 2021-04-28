#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Function to evaluation segmentation module
"""

import cv2
import imageio
from matplotlib import pyplot as plt
from PIL import Image
import tensorflow as tf
from tqdm import tqdm
import os
import urllib.request


def get_tflite_interpreter(tflite_model_url, path_to_save, verbose=False):
    """
    Function to load a tflite model from URL

    Args:
        - (str) tflite_model_url: URL of the model to download
        - (str) path_to_save: path to save the model
        - (bool) verbose: print model input and output details
    Return:
        - (tf.lite.Interpreter) tflite interpreter
    """
    if not os.path.isfile(path_to_save):
        urllib.request.urlretrieve(tflite_model_url, path_to_save)

    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=path_to_save)
    interpreter.allocate_tensors()

    if verbose:
        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

    return interpreter


def run_tflite_interpreter(rgb_img, interpreter):
    """
    Function to infer tflite interpreter

    Args:
        - (cv2 image) image in rgb format
        - (tf.lite.Interpreter) tflite interpreter
    Return:
        - (np.array) interpreter output (width, height, channel)
    """
    # preprocess input image
    input_data = preprocess_image(rgb_img, [256, 256])

    # reshape data according to input_details
    input_data = tf.transpose(input_data, [0, 2, 3, 1])

    # Get result
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'],
                           input_data)
    interpreter.invoke()
    output_data = tf.squeeze(interpreter.get_tensor(
        interpreter.get_output_details()[0]['index']), axis=0)

    return output_data.numpy()


def overlap_img_with_segmap(img, module_output):
    """
    Function to overlap segmentation map with image

    Args:
        - (cv2.image) Raw input image in RGB
        - (numpy array) module_output : output of the model
        - (int) legend_size: factor to multiply legend sized calculated
    Return:
        - (PIL image) overlap between img and module_output
    """
    origin_height, origin_width, _ = img.shape
    im_pil = Image.fromarray(img)

    depth_min = module_output.min()
    depth_max = module_output.max()

    # rescale depthmap between 0-255
    depth_rescaled = (255 * (module_output - depth_min) /
                      (depth_max - depth_min)).astype("uint8")
    depth_rescaled_3chn = cv2.cvtColor(depth_rescaled,
                                       cv2.COLOR_GRAY2RGB)
    module_output_3chn = cv2.applyColorMap(depth_rescaled_3chn,
                                           cv2.COLORMAP_RAINBOW)
    module_output_3chn = cv2.resize(module_output_3chn,
                                    (origin_width, origin_height),
                                    interpolation=cv2.INTER_CUBIC)
    seg_pil = Image.fromarray(module_output_3chn.astype('uint8'), 'RGB')

    overlap = Image.blend(im_pil, seg_pil, alpha=0.6)

    return overlap


def preprocess_image(rgb_img, resize_shape):
    """
    Function to preprocess an image
    Normalization and resized to resize_shape

    Args:
        - (cv2 image) image in rgb format
        - (numpy array) module_output : output of the model
        - (int) legend_size: factor to multiply legend sized calculated
    Return:
        - (tf.Tensor) normalized and resized image in tensor
    """
    normalized_img = rgb_img / 255.0

    img_resized = tf.image.resize(normalized_img, resize_shape,
                                  method='bicubic',
                                  preserve_aspect_ratio=False)
    img_resized = tf.transpose(img_resized, [2, 0, 1])
    img_input = img_resized.numpy()
    reshape_img = img_input.reshape(1, 3, resize_shape[0], resize_shape[1])
    tensor = tf.convert_to_tensor(reshape_img, dtype=tf.float32)
    return tensor


def plt_pred_on_img(module, image, module_input_shape,
                    signature='serving_default',
                    save_path=None, plot_img=False):
    """
    Function to infer a tfhub module (from hub.load) on an image
    Display overlap original image + segmentation map

    Args:
        - module from TensorFlow Hub
        - (cv2 image) rgb image
        - (list) module_input_shape: expected shape for module
        - (str) signature: signature to get the module result
        - (str) save_path: path to save result overlap
        - (bool) plot_img: boolean for showing result
    Return:
        - (PIL image) overlap of rgb image with segmentation map
    """
    plt.figure(figsize=(8, 8))

    module_input = preprocess_image(image, module_input_shape)
    module_output = module.signatures[signature](module_input)
    np_module_output = module_output['default'].numpy().squeeze()
    overlap = overlap_img_with_segmap(image, np_module_output)

    if plot_img:
        plt.show(overlap)

    if save_path is not None:
        plt.savefig(save_path)
    plt.close()
    return overlap


def plot_pred_on_video(video_path, module, module_input_shape,
                       out_gif, signature='serving_default',
                       plot_img=False, fps=30, resize_fact=1, keep_every=1):
    """
    Function to infer a tfhub module (from hub.load) on a video
    Save result into a gif file

    Args:
        - (str) video_path: path to the video
        - module from TensorFlow Hub
        - (list) module_input_shape: expected shape for module
        - (str) out_gif: path to save image result
        - (str) signature: signature to get the module result
        - (bool) plot_img: boolean for showing result
        - (int) fps: output fps for the gif
        - (int) resize_fact: divise output resolution
        - (int) keep_every: keep frame if frame_idx%keep_every == 0
    """
    cap = cv2.VideoCapture(video_path)
    number_of_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    imgs = []
    for i in tqdm(range(number_of_frame)):
        if i % keep_every != 0 and keep_every != 1:
            continue
        _, image = cap.read()
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        overlap = plt_pred_on_img(module, rgb_img, module_input_shape,
                                  plot_img=plot_img,
                                  signature=signature)
        overlap = overlap.resize((int(overlap.size[0]*resize_fact),
                                  int(overlap.size[1]*resize_fact)))
        imgs.append(overlap)

    imgs[0].save(out_gif, format='GIF',
                 append_images=imgs[1:],
                 save_all=True, loop=0)

    gif = imageio.mimread(out_gif, memtest=False)
    imageio.mimsave(out_gif, gif, fps=fps)
