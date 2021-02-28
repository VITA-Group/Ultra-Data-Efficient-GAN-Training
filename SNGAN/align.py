# Usage:
#
# python3 script.py --input original.png --output modified.png
# Based on: https://github.com/mostafaGwely/Structural-Similarity-Index-SSIM-

# 1. Import the necessary packages
from skimage.measure import compare_ssim
import argparse
import cv2
import os
from glob import glob
from tqdm import tqdm

import warnings
from skimage.transform import resize
from scipy.stats import wasserstein_distance
from imageio import imread, imsave
import numpy as np 

# 2. Construct the argument parse and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dir1', type=str)
parser.add_argument('--dir2', type=str)
parser.add_argument('--num', type=int, default=16)
args  = parser.parse_args()
filelist_1 = sorted(glob(os.path.join(args.dir1, 'fid_buffer/*.png')))
filelist_2 = sorted(glob(os.path.join(args.dir2, 'fid_buffer/*.png')))
height = 32
width = 32
def get_img(path, norm_size=True, norm_exposure=False):
    '''
    Prepare an image for image processing tasks
    '''
    # flatten returns a 2d grayscale array
    img = imread(path, as_gray = True).astype(int)
    # resizing returns float vals 0:255; convert to ints for downstream tasks
    if norm_size:
        img = resize(img, (height, width), anti_aliasing=True, preserve_range=True)
    if norm_exposure:
        img = normalize_exposure(img)
    return img
def get_histogram(img):
    '''
    Get the histogram of an image. For an 8-bit, grayscale image, the
    histogram will be a 256 unit vector in which the nth value indicates
    the percent of the pixels in the image with the given darkness level.
    The histogram's values sum to 1.
    '''
    h, w = img.shape
    hist = [0.0] * 256
    for i in range(h):
        for j in range(w):
            hist[img[i, j]] += 1
    return np.array(hist) / (h * w) 


def normalize_exposure(img):
    '''
    Normalize the exposure of an image.
    '''
    img = img.astype(int)
    hist = get_histogram(img)
    # get the sum of vals accumulated by each position in hist
    cdf = np.array([sum(hist[:i+1]) for i in range(len(hist))])
    # determine the normalization values for each unit of the cdf
    sk = np.uint8(255 * cdf)
    # normalize each position in the output image
    height, width = img.shape
    normalized = np.zeros_like(img)
    for i in range(0, height):
        for j in range(0, width):
            normalized[i, j] = sk[img[i, j]]
    return normalized.astype(int)


def earth_movers_distance(path_a, path_b):
    '''
    Measure the Earth Mover's distance between two images
    @args:
        {str} path_a: the path to an image file
        {str} path_b: the path to an image file
    @returns:
        TODO
    '''
    img_a = get_img(path_a, norm_exposure=True)
    img_b = get_img(path_b, norm_exposure=True)
    hist_a = get_histogram(img_a)
    hist_b = get_histogram(img_b)
    return wasserstein_distance(hist_a, hist_b)


def structural_sim(path_a, path_b):
    '''
    Measure the structural similarity between two images
    @args:
        {str} path_a: the path to an image file
        {str} path_b: the path to an image file
    @returns:
        {float} a float {-1:1} that measures structural similarity
        between the input images
    '''
    img_a = get_img(path_a)
    img_b = get_img(path_b)
    sim, diff = compare_ssim(img_a, img_b, full=True)
    return sim


def pixel_sim(path_a, path_b):
    '''
    Measure the pixel-level similarity between two images
    @args:
        {str} path_a: the path to an image file
        {str} path_b: the path to an image file
    @returns:
        {float} a float {-1:1} that measures structural similarity
        between the input images
    '''
    img_a = get_img(path_a, norm_exposure=True)
    img_b = get_img(path_b, norm_exposure=True)
    return np.sum(np.absolute(img_a - img_b)) / (height*width) / 255


def sift_sim(path_a, path_b):
    '''
    Use SIFT features to measure image similarity
    @args:
        {str} path_a: the path to an image file
        {str} path_b: the path to an image file
    @returns:
        TODO
    '''
    # initialize the sift feature detector
    orb = cv2.ORB_create()

    # get the images
    img_a = cv2.imread(path_a)
    img_b = cv2.imread(path_b)

    # find the keypoints and descriptors with SIFT
    kp_a, desc_a = orb.detectAndCompute(img_a, None)
    kp_b, desc_b = orb.detectAndCompute(img_b, None)

    # initialize the bruteforce matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # match.distance is a float between {0:100} - lower means more similar
    matches = bf.match(desc_a, desc_b)
    similar_regions = [i for i in matches if i.distance < 70]
    if len(matches) == 0:
        return 0
    return len(similar_regions) / len(matches)


a1 = []
a2 = []
print(len(filelist_1))
print(len(filelist_2))
for i in tqdm(range(args.num)):
    img_a = filelist_1[i]
    max_score = 0
    max_index = 0
    for j in tqdm(range(len(filelist_2))):
        img_b = filelist_2[j]
        structural_sim_score = structural_sim(img_a, img_b)
        pixel_sim_score = pixel_sim(img_a, img_b)
        sift_sim_score = sift_sim(img_a, img_b)
        emd = earth_movers_distance(img_a, img_b)
        
        score = structural_sim_score + pixel_sim_score + sift_sim_score + emd
        if (score > max_score):
            max_score = score
            max_index = j
    
    a1.append(filelist_1[i])
    a2.append(filelist_2[max_index])
    
    tqdm.write("Max score for {}: {}".format(filelist_1[i], max_score))
    

print(a1)
print(a2)
    
    