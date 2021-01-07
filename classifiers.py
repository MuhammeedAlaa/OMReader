from skimage.transform import hough_line, hough_line_peaks
from skimage.morphology import disk, binary_erosion
from skimage.measure import label, regionprops
import numpy as np
from scipy import stats
from skimage.draw import line
from itertools import groupby
import math
import cv2
from commonfunctions import show_images


def ChordsClassifier(objectWithoutStem, objectTop, staffLineSpacing, pitches, pitches_coord):
    se = disk((staffLineSpacing-2) // 2 - 1)
    objectWithoutStem = np.copy(objectWithoutStem)
    eroded = binary_erosion((255 - objectWithoutStem) / 255, se)
    label_img, num = label(eroded, background=0,
                           return_num=True, connectivity=2)
    props = regionprops(label_img)
    heads = []
    for prop in props:
        if(prop.area != 1):
            heads.append(prop)
    if len(heads) > 1:
        show_images([eroded], ['chord'])
        for head in heads:
            # print('head centroid',head.centroid)
            headPosition = head.centroid[0] + objectTop
            # print('objecttop:', objectTop)
            print(pitches[find_nearest(pitches_coord, headPosition)])
            # print('pitches', pitches)
            # print('pitches_coordinates', pitches_coord)
            # print('headposition: ', headPosition)
            # print('````````````````````````````')
    return len(heads) > 1


def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx


def rle(bits):
    n = len(bits)
    if n == 0:
        return (None, None, None)
    else:
        # pairwise unequal (string safe)
        y = np.array(bits[1:] != bits[:-1])
        i = np.append(np.where(y), n - 1)   # must include last element posi
        lengths = np.diff(np.append(-1, i))       # run lengths
        positions = np.cumsum(np.append(0, lengths))[:-1]  # positions
        return(lengths, positions, bits[i])

# returns the run lengths of the ones in the input array


def runs_of_ones_list(bits):
    return [sum(g) for b, g in groupby(bits) if b]


def getPitchesCoordinates(staffLineSpacing, staffLines, blockNumber):
    maxLedgers = 2  # TODO: if you increase max ledgers modify the pitches array
    pitches_coord = []
    pitches_coord.append(-maxLedgers*staffLineSpacing +
                         staffLines[blockNumber * 5])
    for i in range(2 * maxLedgers - 1):
        pitches_coord.append(pitches_coord[-1]+staffLineSpacing/2)
    for i in range(5):
        currentStaffIndex = blockNumber * 5 + i
        currentStaffPosition = staffLines[currentStaffIndex]
        pitches_coord.append(currentStaffPosition)
        staffSpacing = staffLineSpacing
        if i != 4:
            staffSpacing = staffLines[currentStaffIndex +
                                      1] - currentStaffPosition
        pitches_coord.append(currentStaffPosition + staffSpacing/2)
    for i in range(2 * maxLedgers - 1):
        pitches_coord.append(pitches_coord[-1]+staffLineSpacing/2)
    pitches = ['c3', 'b2', 'a2', 'g2', 'f2', 'e2', 'd2', 'c2',
               'b1', 'a1', 'g1', 'f1', 'e1', 'd1', 'c1', 'b0', 'a0']
    return pitches, pitches_coord


def getHeads(staffLineSpacing, objectWithoutStem):
    se = disk((staffLineSpacing-2) // 2 - 1)
    eroded = binary_erosion(objectWithoutStem, se)
    show_images([eroded], ['Heads'])
    label_img, num = label(eroded, background=0,
                           return_num=True, connectivity=2)
    props = regionprops(label_img)
    heads = []
    for prop in props:
        if(prop.area != 1):
            heads.append(prop)
    return heads


def classifierB(objectWithoutStem, staffLineSpacing, objectTop, pitches, pitches_coord):
    vertical = objectWithoutStem[:, objectWithoutStem.shape[1] // 2 + 1]
    runlengths, startpositions, values = rle(vertical)
    whiteRunHeight = runlengths[np.nonzero(values)[0]]
    whiteRunPositions = startpositions[np.nonzero(values)[0]]
    if len(whiteRunHeight) == 2:
        headPosition = (whiteRunPositions[0] + (whiteRunPositions[1] + whiteRunHeight[1])) // 2 + objectTop
        print(str(pitches[find_nearest(pitches_coord, headPosition)])+'/2')
    elif len(whiteRunHeight) == 1:
        headPosition = whiteRunPositions[0] + whiteRunHeight[0] // 2 + objectTop
        print(str(pitches[find_nearest(pitches_coord, headPosition)])+'/4')


def classifierC(objectWithoutStem, stems, staffLineSpacing, objectTop, pitches, pitches_coord):
    stemPos, stemWidth = stems[0]
    verticalLeftStem = np.zeros((objectWithoutStem.shape[0], 1))
    verticalRightStem = np.zeros((objectWithoutStem.shape[0], 1))
    if stemPos != 0:
        verticalLeftStem = objectWithoutStem[:, stemPos - 1]
    if stemPos + stemWidth + 1 < objectWithoutStem.shape[1]:
        verticalRightStem = objectWithoutStem[:, stemPos + stemWidth + 1]
    runlengthsLeftStem, startpositionsLeftStem, valuesLeftStem = rle(verticalLeftStem)
    runlengthsRightStem, startpositionsRightStem, valuesRightStem = rle(verticalRightStem)
    whiteRunHeightLeftStem = runlengthsLeftStem[np.nonzero(valuesLeftStem)[0]]
    whiteRunPositionsLeftStem = startpositionsLeftStem[np.nonzero(valuesLeftStem)[0]]
    whiteRunHeightRightStem = runlengthsRightStem[np.nonzero(valuesRightStem)[0]]
    whiteRunPositionsRightStem = startpositionsRightStem[np.nonzero(valuesRightStem)[0]]
    oneRuns = max(len(whiteRunHeightLeftStem), len(whiteRunHeightRightStem))
    headPosition = 0
    if len(whiteRunHeightLeftStem) == 0:
        oneRuns = oneRuns - 1
        headPosition = whiteRunPositionsRightStem[0] + whiteRunHeightRightStem[0] // 2 + objectTop
    else:
        headPosition = whiteRunPositionsLeftStem[-1] + whiteRunHeightLeftStem[-1] // 2 + objectTop
    if oneRuns == 1:
        print(str(pitches[find_nearest(pitches_coord, headPosition)])+'/8')
    elif oneRuns == 2:
        print(str(pitches[find_nearest(pitches_coord, headPosition)])+'/16')
    elif oneRuns == 3:
        print(str(pitches[find_nearest(pitches_coord, headPosition)])+'/32')


def classifierA(objectWithoutStem, stems, staffLineSpacing, staffHeight, objectTop, pitches, pitches_coord):
    objectWithoutStem = (255-objectWithoutStem)/255
    
    # make structuring element with height a little more than staffheight and width 
    # of three pixels with ones in the middle column to remove the ledgers
    se = np.zeros((staffHeight+2, 3))
    se[:, se.shape[1]//2] = 1
    objectWithoutStem = binary_erosion(objectWithoutStem, se)
    show_images([objectWithoutStem], ['eroded'])
    horizontal = objectWithoutStem[objectWithoutStem.shape[0] // 2 + 1, :]
    oneRuns = runs_of_ones_list(horizontal)
    if len(oneRuns) == 0:
        classifierB(objectWithoutStem, staffLineSpacing,
                    objectTop, pitches, pitches_coord)
    else:
        classifierC(objectWithoutStem, stems, staffLineSpacing,
                    objectTop, pitches, pitches_coord)
