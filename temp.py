# -*- coding: utf-8 -*-
import cv2
from commonfunctions import *
from skimage import morphology
from skimage.measure import find_contours
from skimage.morphology import binary_erosion, binary_dilation, binary_closing,skeletonize, thin
import numpy as np
from staffLine import *

from skimage.draw import polygon_perimeter
from skimage.util import crop
import matplotlib.patches as patches



def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
        key=lambda b:b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)




img = rgb2gray(io.imread('02.PNG'))

if img.dtype != np.uint8:
    noisy_img = (img * 255).astype(np.uint8)
img_median_filtered = hybridMedian(noisy_img)
img_median_filtered = img_median_filtered.astype(np.uint8)
# img_median_filtered = median(noisy_img)
# gaussian filtering
img_gaussian_filtered = gaussian(img_median_filtered, sigma=0.2)
img_gaussian_filtered = (img_gaussian_filtered * 255).astype(np.uint8)

# image rotation
image_rotated = skew_angle_hough_transform(img_gaussian_filtered)
image_rotated = (image_rotated * 255).astype(np.uint8)

# image binarization
binary = adaptiveThresh(image_rotated, t=15, div=8)

img_object , staffLines = staffLineRemoval(binary, 1)
show_images([img_object])
height, width = img_object.shape
count_images = len(staffLines) // 5
if count_images > 1: 
    padding_up = (staffLines[5] - staffLines[4]) // 2 
    padding_down = padding_up 
else:
    padding_up = staffLines[0]
    padding_down =  height - staffLines[4]

blocks = []
blocks_orginal = []
for i in range(0 , count_images):
    if i == 0:
        blocks.append(img_object[0: staffLines[i * 5 + 4] + padding_down,:])
        blocks_orginal.append(img[0: staffLines[i * 5 + 4] + padding_down,:])
    elif i == count_images - 1:
        blocks.append(img_object[staffLines[i * 5] - padding_up: height ,:])            
        blocks_orginal.append(img[staffLines[i * 5] - padding_up: height ,:])            
    else:
        blocks.append(img_object[staffLines[i * 5] - padding_up: staffLines[i * 5 + 4] + padding_down,:])
        blocks_orginal.append(img[staffLines[i * 5] - padding_up: staffLines[i * 5 + 4] + padding_down,:])

bounding_boxes = []
OMR_objects = []
#get the bounding rectangle for each 4 digits    
for i  in  range(0,  count_images):
    show_images([blocks[i]])
    contours, hier = cv2.findContours(blocks[i].astype(
        np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours, boxes = sort_contours(contours)
    # contours = find_contours(blocks[i], 0.8)
    fig, ax = plt.subplots()
    #ax.imshow(blocks[i], cmap=plt.cm.gray)
    #for contour in contours:
     #   ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
    for c in contours:
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        box = order_box(box)
        OMR_objects.append(blocks_orginal[i][int(box[0][1]):int(box[2][1]),int(box[0][0]):int(box[2][0])])
# for im in OMR_objects:
#     show_images([im])
