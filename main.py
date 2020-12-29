from commonfunctions import *


img = rgb2gray(io.imread('dataset/multiple_lines_scanned.png'))
# noisy_img = random_noise(img, mode='s&p', amount=0.1)
# noisy_img = (noisy_img * 255).astype(np.uint8)
# Median filtering using the hybrid Median filter
noisy_img = img
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

# staff line removal
image_no_staff = staffLineRemoval(binary, 1)

io.imshow(img)
io.show()
io.imshow(noisy_img)
io.show()
io.imshow(img_median_filtered)
io.show()
io.imshow(img_gaussian_filtered)
io.show()
io.imshow(image_rotated)
io.show()
io.imshow(binary)
io.show()
io.imshow(image_no_staff)
io.show()
# images = [img, noisy_img, img_median_filtered, img_gaussian_filtered, binary, image_rotated, image_no_staff]
# titles = ["original", "s&p noise", "median", "median & gaussian", "binary", "Rotated", "no staff"]
# show_images(images, titles)



