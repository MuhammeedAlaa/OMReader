from commonfunctions import *
from classifiers import *
from stemRemoval import stemRemoval
import numpy as np
from staffLine import *
import shutil
import sys
import argparse

# reading input/output folders
parser = argparse.ArgumentParser()
parser.add_argument("inputfolder", help="Input File")
parser.add_argument("outputfolder", help="Output File")
args = parser.parse_args()

inputImagesDirectory = args.inputfolder
directory = os.fsencode(inputImagesDirectory)
inputImages = []
filenames = []
outputDirectory = args.outputfolder

# templates to be used to classify the reltively short symbols with SIFT
templates = read_all_templates()

# reading all images in input folder  
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    filenames.append(os.path.splitext(filename)[0])
    image = img_as_ubyte(io.imread(os.path.join(
        inputImagesDirectory, filename), as_gray=True))
    inputImages.append(image)


# applying pipeline to all images  
for imageIndex in range(len(inputImages)):
    imgOutput = []
    img = inputImages[imageIndex]
    try:
        
        ############################ Pre-Analysis ###############################
        height, width = img.shape
        scaleFactor = min(1800/height, 1800/width)
        if scaleFactor < 1:
            img = rescale(img, scaleFactor, anti_aliasing=False)

        if img.dtype != 'uint8':
            img = (img*255).astype('uint8')
        img_median_filtered = hybridMedian(img).astype('uint8')
        # img_median_filtered = median(noisy_img)
        # gaussian filtering
        img_gaussian_filtered = gaussian(img_median_filtered, sigma=0.2)
        img_gaussian_filtered = (img_gaussian_filtered * 255).astype(np.uint8)

        # image rotation
        image_rotated = skew_angle_hough_transform(img_gaussian_filtered)
        image_rotated = (image_rotated * 255).astype(np.uint8)


        ############################ Segmentation ###############################
        # image binarization
        binary = adaptiveThresh(image_rotated, t=15, div=8)

        # removing stafflines
        img_staffLines_removed, staffLines, staffLineSpacing, staffHeight = staffLineRemoval(
            binary, 1)

        # clipping the unnecessary part before the real music score
        start_x = get_start_x(binary, len(staffLines), staffHeight)
        binary_clipped = binary[:, start_x:img_staffLines_removed.shape[1]]
        img_staffLines_removed_clipped = img_staffLines_removed[:,
                                                                start_x:img_staffLines_removed.shape[1]]

        # split each object in the score to be identified
        objects = split_objects(
            binary_clipped, img_staffLines_removed_clipped, staffLines)


        # if the score had multiple groups we divide them into blocks that have 5 stafflines
        sameBlock = objects[0][2]

        # each block has different pitches coordinates
        pitches, pitches_coord = getPitchesCoordinates(
            staffLineSpacing, staffLines, sameBlock)
        imgOutput.append([])
        accidentals = ""
        number = 0
        two = False

        
        ############################# Classification #############################
        for object, top, blockNumber, dots in objects:
            try:
                # if the we entered a new block recalculte the pitches coordinates
                if sameBlock != blockNumber:
                    imgOutput.append([])
                    sameBlock = blockNumber
                    pitches, pitches_coord = getPitchesCoordinates(
                        staffLineSpacing, staffLines, blockNumber)

                # classify the relatively short symbols using SIFT
                if len(object) < 3.5*staffLineSpacing:
                    objectLabel, objectType = classify_accidentals(
                        (object, top, blockNumber), templates, staffLineSpacing)
                    if objectType == "accidental":
                        accidentals += objectLabel
                    if objectLabel == 'full_note':
                        pitch = pitches[find_nearest(
                            pitches_coord, top + len(object)/2)]
                        objectLabel = pitch[0] + accidentals + pitch[1] + '/1'
                        accidentals = ''
                        objectLabel = objectLabel + '.' * dots
                        imgOutput[-1].append(objectLabel)
                    if objectType == "number":
                        number = number + 1
                        if objectLabel == "2":
                            two = True
                    if number == 2:
                        if two == True:
                            imgOutput[-1].append('\meter<"4/2">')
                        else:
                            imgOutput[-1].append('\meter<"4/4">')
                        two = False
                        number = 0
                    continue
                objectWithouStem, stems = stemRemoval(object, staffLineSpacing)
                if len(stems) == 0:
                    continue
                elif len(stems) == 1:
                    note = ChordsClassifier(
                        objectWithouStem, top, staffLineSpacing, pitches, pitches_coord)
                    if note != '':
                        imgOutput[-1].append(note)
                        continue
                    note = classifierA(objectWithouStem, stems, staffLineSpacing,
                                    staffHeight, top, pitches, pitches_coord, dots, accidentals)
                    if note != '':
                        imgOutput[-1].append(note)
                        accidentals = ""
                else:
                    if chordOrBeamCheck(objectWithouStem) == 'chord':
                        note = ChordsClassifier(
                            object, top, staffLineSpacing, pitches, pitches_coord)
                        if note != '':
                            imgOutput[-1].append(note)
                    else:
                        notes = beamClassifier(object, objectWithouStem, staffLineSpacing,
                                            staffHeight, top, pitches, pitches_coord, stems)
                        imgOutput[-1].extend(notes)
            except:
                pass
    except:
        pass
    if outputDirectory[-1] == '/':
      outputFileName = outputDirectory + filenames[imageIndex] + '.txt'
    else:
        outputFileName = outputDirectory + '/' + filenames[imageIndex] + '.txt'
    writeOutput(outputFileName, imgOutput)
