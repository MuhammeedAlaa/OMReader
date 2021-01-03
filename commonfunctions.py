import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
from skimage.exposure import histogram
from matplotlib.pyplot import bar
from skimage.color import rgb2gray, rgb2hsv
import os
from staffLine import staffLineRemoval

# Convolution:
from scipy.signal import convolve2d
from scipy.stats import mode

from scipy import fftpack
import math

from skimage.util import random_noise, pad
from skimage.filters import median, gaussian
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks
from skimage.transform import rotate

import cv2
# import imutils
from skimage import exposure
from skimage.morphology import binary_closing, binary_erosion, disk
from skimage.draw import ellipse
from skimage.transform import hough_circle, hough_circle_peaks



def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    rect[0, 0] -= 20
    rect[0, 1] -= 20

    rect[1, 0] += 20
    rect[1, 1] -= 20

    rect[2, 0] += 20
    rect[2, 1] += 20

    rect[3, 0] -= 20
    rect[3, 1] += 20

    return rect


def order_box(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)

    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def deskew_projection(gray_img):
    img = cv2.GaussianBlur(gray_img.copy(), (3, 3), 1)
    edged_img = cv2.Canny(img, 30, 200)
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 60))
    dilated_img = cv2.dilate(edged_img, se, 5)
    contours, hier = cv2.findContours(dilated_img.astype(
        np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    max_contour = contours[0]
    pts = np.zeros((max_contour.shape[0], 2), dtype=max_contour.dtype)
    for i in range(max_contour.shape[0]):
        pts[i] = max_contour[i, 0]
    gray_img = np.pad(gray_img, 100, mode='edge')
    wrapped_img = four_point_transform(gray_img, order_points(pts)+100)
    return wrapped_img


def projection_correction(img):
    img2 = deskew_projection(img)
    img3 = deskew_projection(img2)
    return img3

#image = rgb2gray(io.imread('scanned_sheet_low.jpg'))
# if image.dtype != "uint8":
#    image = (image * 255).astype("uint8")
#img2 = projection_correction(image)
# io.imshow(img2)

# Show the figures / plots inside the notebook


def show_images(images, titles=None):
    # This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None:
        titles = ['(%d)' % i for i in range(1, n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image, title in zip(images, titles):
        a = fig.add_subplot(1, n_ims, n)
        if image.ndim == 2:
            plt.gray()
        max = 1
        if image.dtype == "uint8":
            max = 255
        plt.imshow(image, vmin=0, vmax=max)
        a.set_title(title)
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show()


def showHist(img):
    # An "interface" to matplotlib.axes.Axes.hist() method
    plt.figure()
    imgHist = histogram(img, nbins=256)

    bar(imgHist[1].astype(np.uint8), imgHist[0], width=0.8, align='center')


def adaptiveThresh(img, t, div):

    height, width = img.shape

    s = width // div
    intImg = np.zeros([height + 2 * s, width + 2 * s])
    out = img.copy()
    for i in range(0, height, 1):
        for j in range(0, width, 1):
            intImg[i, j] = img[i, j]
            if i != 0:
                intImg[i, j] += intImg[i - 1, j]
            if j != 0:
                intImg[i, j] += intImg[i, j - 1]
            if i != 0 or j != 0:
                intImg[i, j] -= intImg[i - 1, j - 1]

    for i in range(0, height, 1):
        for j in range(0, width, 1):
            x1 = max(round(i - s / 2), 0)
            x2 = min(round(i + s / 2), height - 1)
            y1 = max(round(j - s / 2), 0)
            y2 = min(round(j + s / 2), width - 1)
            count = (x2 - x1) * (y2 - y1)
            sum = intImg[x2, y2] - intImg[x2, y1 - 1] - \
                intImg[x1 - 1, y2] + intImg[x1 - 1, y1 - 1]
            if img[i, j] * count <= (sum * (100-t)/100):
                out[i, j] = 0
            else:
                out[i, j] = 255
    return out

# this function to read the data set from the folder


def readDataSet():
    directory = os.fsencode("./dataset")
    dataset = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".png") or filename.endswith(".jpg"):
            image = rgb2gray(io.imread(os.path.join('./dataset/', filename)))
            if image.dtype != "uint8":
                image = (image * 255).astype("uint8")
            dataset.append(image)
    return dataset


def hybridMedian(img):
    # applying the 3 * 3 hybrid filter
    # define filter shapes for different spatial directions
    img_grayscale = np.copy(img)
    img_grayscale = img_grayscale.astype(np.uint8)
    cross_filter = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    plus_filter = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    fil_cross = median(img_grayscale, cross_filter)
    fil_plus = median(img_grayscale, plus_filter)

    # calculate the median of the three images for each pixel
    combined_images = np.array([fil_cross, fil_plus, img_grayscale])
    filtered_img_hybrid = np.median(combined_images, axis=0)
    return filtered_img_hybrid


def skew_angle_hough_transform(image):
    edges = canny(image)
    tested_angles = np.deg2rad(np.arange(0.1, 180.0))
    h, theta, d = hough_line(edges, theta=tested_angles)
    accum, angles, dists = hough_line_peaks(h, theta, d)
    most_common_angle = mode(np.around(angles, decimals=2))[0]
    skew_angle = np.rad2deg(most_common_angle - np.pi/2)
    img_rotated = rotate(image, skew_angle, resize=True, mode='constant', cval=255)
    return img_rotated

def sort_contours_horizontally(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
        key=lambda b:b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)


def split_objects(img_thresh):
    img_objects , staffLines = staffLineRemoval(img_thresh, 1)
    height = img_objects.shape[0]
    count_blocks = len(staffLines) // 5
    
    if count_blocks > 1: 
        padding_up = (staffLines[5] - staffLines[4]) // 2 
        padding_down = padding_up 
    else:
        padding_up = staffLines[0]
        padding_down =  height - staffLines[4]

    blocks = []
    blocks_orginal = []
    for i in range(0 , count_blocks):
        if i == 0:
            blocks.append(img_objects[0: staffLines[i * 5 + 4] + padding_down,:])
            blocks_orginal.append(img_thresh[0: staffLines[i * 5 + 4] + padding_down,:])
        elif i == count_blocks - 1:
            blocks.append(img_objects[staffLines[i * 5] - padding_up: height ,:])            
            blocks_orginal.append(img_thresh[staffLines[i * 5] - padding_up: height ,:])            
        else:
            blocks.append(img_objects[staffLines[i * 5] - padding_up: staffLines[i * 5 + 4] + padding_down,:])
            blocks_orginal.append(img_thresh[staffLines[i * 5] - padding_up: staffLines[i * 5 + 4] + padding_down,:])
    
    staffHeight = staffLines[4] - staffLines[3]
    
    objects = []
    
    for i  in  range(count_blocks):
        contours, hier = cv2.findContours(255 - blocks[i], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt = []
        for j in range (0, len(contours)):
            if hier[0,j,3] == -1:
                cnt.append(contours[j])
        cnt, boxes_sorted = sort_contours_horizontally(cnt)        
        for c in cnt:
            Xmin = int(np.min(c[:,0, 0]))
            Xmax = int(np.max(c[:,0, 0]))
            Ymin = int(np.min(c[:,0, 1]))
            Ymax = int(np.max(c[:,0, 1]))

            object_width = Xmax - Xmin
            object_height = Ymax - Ymin
            if object_width > staffHeight//2: 
                current_obj = blocks[i][0:blocks[i].shape[0], Xmin:Xmax]
                objects.append(current_obj)  
            elif object_height <= staffHeight//2:
                point_img = np.ones((blocks[i].shape[0],15))
                point_img[blocks[i].shape[0]//2-3:blocks[i].shape[0]//2+3, 5:10] = 0
                objects.append(point_img)   
    return objects


