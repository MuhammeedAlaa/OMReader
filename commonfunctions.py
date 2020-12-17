

import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
from skimage.exposure import histogram
from matplotlib.pyplot import bar
from skimage.color import rgb2gray, rgb2hsv
import os

# Convolution:
from scipy.signal import convolve2d
from scipy import fftpack
import math

from skimage.util import random_noise, pad
from skimage.filters import median, gaussian
from skimage.feature import canny

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



