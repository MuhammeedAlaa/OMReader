from skimage.transform import hough_line, hough_line_peaks
import numpy as np
from scipy import stats
from skimage.draw import line
import skimage.io as io


def staffLineRemoval(thresholdedImg, thicknessThresholdFraction):
    img = np.copy(thresholdedImg)  # for convinience :)
    height, width = img.shape
    # Invert the input binary image
    imgBinary = 255 - img

    # apply hough lines to detect stafflines
    hspace, angles, dists = hough_line(imgBinary)
    hspace, angles, dists = hough_line_peaks(hspace, angles, dists)
    filtered_dists = []
    # ---------------------------------------------------------------------------
    temp_cnt = -1
    for a in angles:
        temp_cnt = temp_cnt + 1
        if a >= 1.56 and a <= 1.58:
            filtered_dists.append(dists[temp_cnt])
    # ---------------------------------------------------------------------------
    staffLines = np.sort(np.round(dists).astype('int32'))
    # remove negative staffLines 
    staffLines = staffLines[staffLines >= 0]
    # find most common black pixel run length (white pixel run length in binary image due to inversion)
    # This should correspond to staff line thickness
    staffLineThickness = verticalRunLengthMode(imgBinary, 255, width, height)
    # TODO: check whether the addition is needed
    staffLineSpacing = verticalRunLengthMode(imgBinary, 0, width, height)

    threshold = staffLineSpacing / 2  # TODO: check whether needed
    for staffLineRow in staffLines:
        for x in range(width-1, 0, -1):
            if img[staffLineRow, x] != 0:
                for j in range(1, round(threshold/2)):
                    if img[staffLineRow + j, x] == 0:
                        staffLineRow = staffLineRow + j
                        break
                    if img[staffLineRow - j, x] == 0:
                        staffLineRow = staffLineRow - j
                        break
            verticalThresholdResult = testVerticalThreshold(
                img, x, staffLineRow, staffLineThickness*thicknessThresholdFraction)
            if(verticalThresholdResult[0]):
                rr, cc = line(
                    verticalThresholdResult[1], x, verticalThresholdResult[2], x)
                img[rr, cc] = 255
        # TODO: fix with morphology the broken objects
    return (img, staffLines)


# Returns the mode vertical run length of the given colour in the input image
def verticalRunLengthMode(img, colour, width, height):
    runLengths = []
    for x in range(0, width):
        inColour = False
        currentRun = 0
        for y in range(0, height):
            if (img[y, x] == colour):
                if (inColour):
                    currentRun = currentRun + 1
                else:
                    currentRun = 1
                    inColour = True
            else:
                if (inColour):
                    runLengths.append(currentRun)
                    inColour = False
    return int(stats.mode(runLengths)[0][0])


# to check whether the point in the staff line coincide with a note
# providing a threshold for the thickness of the stafflines
def testVerticalThreshold(img, x, y, threshold):
    upperY = y
    lowerY = y
    while (upperY >= 0):
        if (img[upperY-1, x] == 0):
            upperY -= 1
        else:
            break
    while (lowerY <= len(img)):
        if (img[lowerY+1, x] == 0):
            lowerY += 1
        else:
            break
    return (lowerY - upperY <= threshold), upperY, lowerY
