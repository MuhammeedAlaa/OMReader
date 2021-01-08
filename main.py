from commonfunctions import *
from classifiers import *
from stemRemoval import stemRemoval
import numpy as np
from staffLine import *
import shutil


inputImagesDirectory = "./dataset/scanned"
directory = os.fsencode(inputImagesDirectory)
inputImages = []
filenames = []
outputDirectory = './outputs/'
try:
    if os.path.exists(outputDirectory) and os.path.isdir(outputDirectory):
        shutil.rmtree(outputDirectory)
    os.mkdir(outputDirectory)
except:
    print("output file is not empty")

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    filenames.append(os.path.splitext(filename)[0])
    image = img_as_ubyte(io.imread(os.path.join(
        inputImagesDirectory, filename), as_gray=True))
    inputImages.append(image)

for imageIndex in range(len(inputImages)):
    imgOutput = []
    img = inputImages[imageIndex]

    img_median_filtered = hybridMedian(img).astype('uint8')
    # img_median_filtered = median(noisy_img)
    # gaussian filtering
    img_gaussian_filtered = img_as_ubyte(
        gaussian(img_median_filtered, sigma=0.2))

    # image rotation
    image_rotated = img_as_ubyte(
        skew_angle_hough_transform(img_gaussian_filtered))

    # image binarization
    binary = adaptiveThresh(image_rotated, t=15, div=8)

    # removing stafflines
    img_staffLines_removed, staffLines, staffLineSpacing, staffHeight = staffLineRemoval(
        binary, 1)

    # split each object in the score to be identified
    objects = split_objects(binary, img_staffLines_removed, staffLines)

    # templates to be used to classify the reltively short symbols with SIFT
    templates = read_all_templates()

    # if the score had multiple groups we divide them into blocks that have 5 stafflines
    sameBlock = objects[0][2]

    # each block has different pitches coordinates
    pitches, pitches_coord = getPitchesCoordinates(
        staffLineSpacing, staffLines, sameBlock)
    imgOutput.append([])
    accidentals = ""
    number = 0
    two = False
    for object, top, blockNumber, dots in objects:
        # if the we entered a new block recalculte the pitches coordinates
        if sameBlock != blockNumber:
            imgOutput.append([])
            sameBlock = blockNumber
            pitches, pitches_coord = getPitchesCoordinates(
                staffLineSpacing, staffLines, blockNumber)

        # classify the relatively short symbols using SIFT
        if len(object) < 3.5*staffLineSpacing:
            # show_images([object])
            objectLabel, objectType = classify_accidentals(
                (object, top, blockNumber), templates, staffLineSpacing)
            if objectType == "accidental":
                accidentals += objectLabel
            if objectLabel == 'full_note':
                objectLabel = pitches[find_nearest(
                    pitches_coord, top + len(object)/2)] + '/1'
                objectLabel = objectLabel + '.' * dots
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
        # show_images([object, objectWithouStem])
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
                beamClassifier(object, objectWithouStem, staffLineSpacing,
                               staffHeight, top, pitches, pitches_coord)
                notes = beamClassifier(object, objectWithouStem, staffLineSpacing,
                                       staffHeight, top, pitches, pitches_coord)
                imgOutput[-1].extend(notes)

    outputFileName = outputDirectory + filenames[imageIndex] + '.txt'
    writeOutput(outputFileName, imgOutput)
