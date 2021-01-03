from commonfunctions import *


img = rgb2gray(io.imread('cam/12.jpg'))
# noisy_img = random_noise(img, mode='s&p', amount=0.1)
# noisy_img = (noisy_img * 255).astype(np.uint8)
# Median filtering using the hybrid Median filter
img_gray = img
if img_gray.dtype != np.uint8:
    img_gray = (img_gray * 255).astype(np.uint8)

img_median_filtered = (hybridMedian(img_gray)).astype(np.uint8)

# gaussian filtering
img_gaussian_filtered = (gaussian(img_median_filtered, sigma=0.2)* 255).astype(np.uint8)

# image binarization
binary = adaptiveThresh(img_gaussian_filtered, t=15, div=8)
# image rotation
image_rotated = (skew_angle_hough_transform(binary)* 255).astype(np.uint8)
show_images([img, img_gray, img_median_filtered, img_gaussian_filtered, binary, image_rotated])

# staff line removal
image_no_staff, stuff = staffLineRemoval(image_rotated, 1)