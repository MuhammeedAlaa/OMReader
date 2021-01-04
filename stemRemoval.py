from skimage.transform import hough_line, hough_line_peaks
import numpy as np
from scipy import stats
from skimage.draw import line
from itertools import groupby


def stemRemoval(img, staffLineSpacing):
    outputImg = np.copy(img)
    height, width = outputImg.shape
    verticalProjection = (
        np.sum((255 - outputImg)/255, axis=0)).astype('uint64')
    maxVertical = 3 * staffLineSpacing
    mask = np.where(verticalProjection <= maxVertical, 0, 1)
    runlengths, startpositions, values = rle(mask)
    stemsWidths = runlengths[np.nonzero(values)[0]]
    stemsPositions = startpositions[np.nonzero(values)[0]]

    # assume that the stem is almost vertical (bounding box of the stem width = 5)
    maxStemSkew = 5
    maxStemWidth = np.max(stemsWidths, initial=0)
    threshold = maxStemSkew / 2

    for stem in stemsPositions:
        for y in range(0, height, 1):
            if outputImg[y, stem] != 0:
                for j in range(1, round(threshold/2)):
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

# returns: tuple (runlengths, startpositions, values)


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
