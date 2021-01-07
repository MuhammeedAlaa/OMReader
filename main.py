from commonfunctions import *
from classifiers import *
from stemRemoval import stemRemoval
import numpy as np
from staffLine import *


img = rgb2gray(io.imread('dataset/scanned/16.PNG'))

noisy_img = img_as_ubyte(img)
img_median_filtered = hybridMedian(noisy_img)
img_median_filtered = img_median_filtered.astype(np.uint8)
# img_median_filtered = median(noisy_img)
# gaussian filtering
img_gaussian_filtered = gaussian(img_median_filtered, sigma=0.2)
img_gaussian_filtered = (img_gaussian_filtered * 255).astype(np.uint8)

# image rotation
image_rotated = skew_angle_hough_transform(img_gaussian_filtered)
show_images([image_rotated])
image_rotated = (image_rotated * 255).astype(np.uint8)

# image binarization
binary = adaptiveThresh(image_rotated, t=15, div=8)
show_images([binary])
img_staffLines_removed, staffLines, staffLineSpacing, staffHeight = staffLineRemoval(
    binary, 1)

objects = split_objects(binary, img_staffLines_removed, staffLines)

templates = read_all_templates()
point_img = np.ones((20, 20))
point_img[7:12, 7:12] = 0

sameBlock = objects[0][2]
pitches = []
pitches_coord = []
pitches, pitches_coord = getPitchesCoordinates(
    staffLineSpacing, staffLines, sameBlock)

print(staffLines)
for object, top, blockNumber in objects:
    if sameBlock != blockNumber:
        sameBlock = blockNumber
        pitches, pitches_coord = getPitchesCoordinates(
            staffLineSpacing, staffLines, blockNumber)
    if len(object) < 3.5*staffLineSpacing:
        show_images([object])
        if np.array_equal(object, point_img):
            # print(".")
            lbl = '.'
        else:
            lbl = classify_accidentals(
                (object, top, blockNumber), templates, staffLineSpacing)
            if lbl == 'full_note':
                lbl = pitches[find_nearest(
                    pitches_coord, top + len(object)/2)] + '/1'
        print(lbl)
        continue
    testNoStem, stems = stemRemoval(object, staffLineSpacing)
    print('stems: ', stems)
    show_images([object, testNoStem])
    if len(stems) == 0:
        continue
    elif len(stems) == 1:
        # show_images([testNoStem])
        if ChordsClassifier(testNoStem, top, staffLineSpacing, pitches, pitches_coord):
            continue
        classifierA(testNoStem, stems, staffLineSpacing,
                    staffHeight, top, pitches, pitches_coord)
    else:
        beamClassifier(object, testNoStem, staffLineSpacing,
                       staffHeight, top, pitches, pitches_coord)
