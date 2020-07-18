#!/usr/bin/env python
import numpy as np
import math
import cv2

"""
Generates an RGB image where each point corresponds to flow in that direction from the center,
as visualized by flow_viz_tf.
Output: color_wheel_rgb: [1, width, height, 3]
"""
def draw_color_wheel_np(width, height):
    color_wheel_x = np.linspace(-width / 2.,width / 2.,width)
    color_wheel_y = np.linspace(-height / 2.,height / 2.,height)
    color_wheel_X, color_wheel_Y = np.meshgrid(color_wheel_x, color_wheel_y)
    color_wheel_rgb = flow_viz_np(color_wheel_X, color_wheel_Y)
    return color_wheel_rgb


"""
Visualizes optical flow in HSV space using TensorFlow, with orientation as H, magnitude as V.
Returned as RGB.
Input: flow: [batch_size, width, height, 2]
Output: flow_rgb: [batch_size, width, height, 3]
"""
def flow_viz_np(flow_x, flow_y):
    import cv2
    flows = np.stack((flow_x, flow_y), axis=2)
    mag = np.linalg.norm(flows, axis=2)

    ang = np.arctan2(flow_y, flow_x)
    ang += np.pi
    ang *= 180. / np.pi / 2.
    ang = ang.astype(np.uint8)
    hsv = np.zeros([flow_x.shape[0], flow_x.shape[1], 3], dtype=np.uint8)
    hsv[:, :, 0] = ang
    hsv[:, :, 1] = 255
    hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return flow_rgb