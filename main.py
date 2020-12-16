from commonfunctions import *


img = rgb2gray(io.imread('dataset/scanned_sheet.jpg'))
img_filtered = gaussian(img, sigma=1)
show_images([img, img_filtered], ["original image", "filtered image"])
