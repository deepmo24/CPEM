# coding: utf-8
'''
reference:
https://github.com/cleardusk/3DDFA_V2
'''

__author__ = 'cleardusk'

import sys

sys.path.append('..')

import cv2
import numpy as np

from Sim3DR import RenderPipeline


cfg = {
    'intensity_ambient': 0.3,
    'color_ambient': (1, 1, 1),
    'intensity_directional': 0.6,
    'color_directional': (1, 1, 1),
    'intensity_specular': 0.1,
    'specular_exp': 5,
    'light_pos': (0, 0, 5),
    'view_pos': (0, 0, 5)
}

render_app = RenderPipeline(**cfg)

'''
Be careful of the input format of render_app(vertices, triangle, image):
1. all inputs must be numpy array.
2. vertices and triangle must be C-continuous
3. Type: vertices(np.float32), triangle(np.int32), image(np.uint8)
'''
def render(img, vertices, tri, alpha=0.6, wfp=None, with_bg_flag=True):
    if with_bg_flag:
        overlap = img.copy()
    else:
        overlap = np.zeros_like(img)


    overlap = render_app(vertices, tri, overlap)

    if with_bg_flag:
        res = cv2.addWeighted(img, 1 - alpha, overlap, alpha, 0)
    else:
        res = overlap

    if wfp is not None:
        cv2.imwrite(wfp, res)
        print(f'Save visualization result to {wfp}')

    return res

