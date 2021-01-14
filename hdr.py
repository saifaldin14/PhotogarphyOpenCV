import argparse
from matplotlib import cm
import itertools
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import cv2
from common import load_image

import exifread

MARKERS = ['o', '+', 'x', '*', '.', '1', 'v', 'D']

def thumbnail(img_rgb, long_edge=400):
    original_long_edge = max(img_rgb.shape[:2])
    dimensions = tuple([int(x / original_long_edge * long_edge) for x in img_rgb.shape[:2][::-1]])
    print('dimensions', dimensions)
    return cv2.resize(img_rgb, dimensions, interpolation=cv2.INTER_AREA)

def exposure_strength(path, iso_ref=100, f_stop_ref=6.375):
    with open(path, 'rb') as infile:
        tags = exifread.process_file(infile)
    [f_stop] = tags['EXIF ApertureValue'].values
    [iso_speed] = tags['EXIF ISOSpeedRatings'].values
    [exposure_time] = tags['EXIF ExposureTime'].values

    rel_aperature_area = 1 / (f_stop.num / f_stop.den / f_stop_ref) ** 2
    exposure_time_float = exposure_time.num / exposure_time.den

    score = rel_aperature_area * exposure_time_float * iso_speed / iso_ref
    return score, np.log2(score)

def lowe_match(descriptors1, descriptors2):
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # discard bad matches, ratio test as ler Lowe's paper
    good_matches = [m for m, n in matches
                    if m.distance < 0.7 * n.distance]
    return good_matches

def save_8bit(img, name):
    img_8bit = np.clip(img * 255, 0, 255).astype('uint8')
    cv2.imwrite(name, img_8bit)
    return img_8bit

OPEN_CV_COLORS = 'bgr'

def plot_crf(crf, colors=OPEN_CV_COLORS):
    for i, c in enumerate(colors):
        plt.plot(crf_debevec[:, 0, i], color=c)
