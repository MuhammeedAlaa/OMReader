import numpy as np
from skimage.draw import line
from itertools import groupby
from commonfunctions import rle


def stemRemoval(img, staffLineSpacing):
    outputImg = np.copy(img)
    height, width = outputImg.shape
    stemsPositions, stemsWidths = StemDetection(img, staffLineSpacing)

    # margin of error of the stem width as it may have some small thick part
    stemWidthErrorMargin = 1.5
    maxStemWidth = int(max(stemsWidths, default=0)*stemWidthErrorMargin)

    for stem in stemsPositions:
        for y in range(0, height, 1):
            if outputImg[y, stem] != 0:
                for j in range(1, maxStemWidth//2):
                    if stem + j < width and outputImg[y, stem + j] == 0:
                        stem = stem + j
                        break
                    if stem - j > -1 and outputImg[y, stem - j] == 0:
                        stem = stem - j
                        break
            horizontalThresholdResult = testHorizontalThreshold(
                outputImg, stem, y, maxStemWidth)
            if(horizontalThresholdResult[0]):
                rr, cc = line(
                    y, horizontalThresholdResult[1], y, horizontalThresholdResult[2])
                outputImg[rr, cc] = 255
    return outputImg, list(zip(stemsPositions, stemsWidths))


# returns the run lengths of the ones in the input array
def runs_of_ones_list(bits):
    return [sum(g) for b, g in groupby(bits) if b]


def StemDetection(img, staffLineSpacing):
    height, width = img.shape
    maxVertical = 7 * staffLineSpacing / 2 
    RLEImg = (255 - np.copy(img))/255
    candidateStemsPos = []
    stemsWidths = []
    stemsPositions = []
    for i in range(width):
        runlengths, startpositions, values = rle(RLEImg[:, i])
        objectRuns = runlengths[np.nonzero(values)[0]]
        if any(y >= maxVertical for y in objectRuns):
            if len(candidateStemsPos) > 0 and candidateStemsPos[-1] + 1 == i:
                stemsWidths[-1] += 1
            else:
                stemsWidths.append(1)
                stemsPositions.append(i)
            candidateStemsPos.append(i)
    return (stemsPositions, stemsWidths)



def testHorizontalThreshold(img, x, y, threshold):
    width = img.shape[1]
    leftX = x
    rightX = x
    while (leftX > 0):
        if (img[y, leftX-1] == 0):
            leftX -= 1
        else:
            break
    while (rightX < width-1):
        if (img[y, rightX+1] == 0):
            rightX += 1
        else:
            break
    return (rightX - leftX <= threshold), leftX, rightX
