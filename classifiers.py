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
    # show_images([eroded], ['Heads'])
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
        headPosition = (
            whiteRunPositions[0] + (whiteRunPositions[1] + whiteRunHeight[1])) // 2 + objectTop
        return pitches[find_nearest(pitches_coord, headPosition)] + '/2'
    elif len(whiteRunHeight) == 1:
        headPosition = whiteRunPositions[0] + \
            whiteRunHeight[0] // 2 + objectTop
        return pitches[find_nearest(pitches_coord, headPosition)] + '/4'
    return ''

def classifierC(objectWithoutStem, stems, staffLineSpacing, objectTop, pitches, pitches_coord):
    stemPos, stemWidth = stems[0]
    verticalLeftStem = np.zeros((objectWithoutStem.shape[0], 1))
    verticalRightStem = np.zeros((objectWithoutStem.shape[0], 1))
    if stemPos != 0:
        verticalLeftStem = objectWithoutStem[:, stemPos - 1]
    if stemPos + stemWidth + 1 < objectWithoutStem.shape[1]:
        verticalRightStem = objectWithoutStem[:, stemPos + stemWidth + 1]
    runlengthsLeftStem, startpositionsLeftStem, valuesLeftStem = rle(
        verticalLeftStem)
    runlengthsRightStem, startpositionsRightStem, valuesRightStem = rle(
        verticalRightStem)
    whiteRunHeightLeftStem = runlengthsLeftStem[np.nonzero(valuesLeftStem)[0]]
    whiteRunPositionsLeftStem = startpositionsLeftStem[np.nonzero(valuesLeftStem)[
        0]]
    whiteRunHeightRightStem = runlengthsRightStem[np.nonzero(valuesRightStem)[
        0]]
    whiteRunPositionsRightStem = startpositionsRightStem[np.nonzero(valuesRightStem)[
        0]]
    oneRuns = max(len(whiteRunHeightLeftStem), len(whiteRunHeightRightStem))
    headPosition = 0
    if len(whiteRunHeightLeftStem) == 0:
        oneRuns = oneRuns - 1
        headPosition = whiteRunPositionsRightStem[0] + \
            whiteRunHeightRightStem[0] // 2 + objectTop
    else:
        headPosition = whiteRunPositionsLeftStem[-1] + \
            whiteRunHeightLeftStem[-1] // 2 + objectTop
    if oneRuns == 1:
        return pitches[find_nearest(pitches_coord, headPosition)] + '/8'
    elif oneRuns == 2:
        return pitches[find_nearest(pitches_coord, headPosition)] + '/16'
    elif oneRuns == 3:
        return pitches[find_nearest(pitches_coord, headPosition)] + '/32'
    return ''


def classifierA(objectWithoutStem, stems, staffLineSpacing, staffHeight, objectTop, pitches, pitches_coord, dots):
    objectWithoutStem = (255-objectWithoutStem)/255

    # make structuring element with height a little more than staffheight and width
    # of three pixels with ones in the middle column to remove the ledgers
    se = np.zeros((2*staffHeight, 3))
    se[:, se.shape[1]//2] = 1
    objectWithoutStem = binary_erosion(objectWithoutStem, se)
    # show_images([objectWithoutStem], ['eroded'])
    horizontal = objectWithoutStem[objectWithoutStem.shape[0] // 2 + 1, :]
    oneRuns = runs_of_ones_list(horizontal)
    note = ''
    if len(oneRuns) == 0:
        note = classifierB(objectWithoutStem, staffLineSpacing,
                           objectTop, pitches, pitches_coord)
    else:
        note = classifierC(objectWithoutStem, stems, staffLineSpacing,
                           objectTop, pitches, pitches_coord)
    if note != '':
        note = note + ('.' * dots)
    return note


def beamClassifier(object, objectWithoutStem, staffLineSpacing, staffHeight, objectTop, pitches, pitches_coord):
    objectWithoutStem = (255-objectWithoutStem)/255
    # check if it the beam is above or below the note heads
    height, width = objectWithoutStem.shape
    object = 255 - object
    rectL_sum = np.sum(object[:, :staffLineSpacing//2])
    rectR_sum = np.sum(object[:, width-(staffLineSpacing//2): width])
    # remove ledgers
    # make structuring element with height a little more than staffheight and width
    # of three pixels with ones in the middle column to remove the ledgers
    se = np.zeros((2*staffHeight, 3))
    se[:, se.shape[1]//2] = 1
    # se = np.ones((2 * staffHeight, staffHeight))
    objectWithoutStem = binary_erosion(objectWithoutStem, se)
    labeled_img, num_labels = label(
        objectWithoutStem, background=0, return_num=True, connectivity=2)
    regions = regionprops(labeled_img)
    bboxes = []
    bboxes_centroids = []
    for region in regions:
        rect_endpoints = region['bbox']
        # get bounding box coordinates
        min_row = rect_endpoints[0]
        min_col = rect_endpoints[1]
        max_row = rect_endpoints[2]
        max_col = rect_endpoints[3]
        bbox_width = max_col - min_col
        bbox_height = max_row - min_row
        if (bbox_height < 1.5 * staffLineSpacing) and (bbox_width < 1.5 * staffLineSpacing):
            bboxes.append(rect_endpoints)
            bboxes_centroids.append(region['centroid'])
        bboxes_centroids.sort(key=lambda x: x[1])
        bboxes.sort(key=lambda x: x[1])
    notes = []
    for centroid in bboxes_centroids:
        cent_y = centroid[0]
        cent_x = centroid[1]
        # top_staff_y = staffLines[top_block_staffLine[cnt_obj]]
        # print('pos : ' + str((cent_y - top_staff_y) // staffLineSpacing))
        headPosition = cent_y + objectTop
        notes.append(pitches[find_nearest(pitches_coord, headPosition)])
        if rectR_sum < rectL_sum:
            duration = calc_duration(cent_y, cent_x, objectWithoutStem, 'top', staffLineSpacing)
            if duration == '':
                notes = notes[:-1]
            else:
                notes[-1] = notes[-1] + '/' + duration
        else:
            duration = calc_duration(cent_y, cent_x, objectWithoutStem, 'bottom', staffLineSpacing)
            if duration == '':
                notes = notes[:-1]
            else:
                notes[-1] = notes[-1] + '/' + duration
    return notes


def calc_duration(cent_y, cent_x, object, note_pos, staffLineSpacing):
    if note_pos == 'bottom':
        # to get to the point above the note head
        min_y = int(cent_y - staffLineSpacing)
        if int(cent_x + 1.5 * staffLineSpacing) >= object.shape[1]:
            detection_line_col = int(cent_x - 1.5 * staffLineSpacing)
        else:
            detection_line_col = int(cent_x + 1.5 * staffLineSpacing)
        detection_line = np.array(object[0:min_y, detection_line_col], ndmin=1)

    else:
        # to get to the point below the note head
        max_y = int(cent_y + 1.5 * staffLineSpacing)
        if int(cent_x - staffLineSpacing) <= 0:
            detection_line_col = int(cent_x + 1.5 * staffLineSpacing)
        else:
            detection_line_col = int(cent_x - 1.5 * staffLineSpacing)
        detection_line = np.array(
            object[max_y:object.shape[0], detection_line_col])
    mask = np.where(detection_line > 0, 1, 0)
    if len(mask) == 0:
        mask=np.zeros((1, 5)) # arbitrary empty mask
    runlengths, startpositions, values = rle(mask)
    num_startpositions = len(startpositions[np.nonzero(values)[0]])
    if num_startpositions == 1:
        note_duration = '8'
    elif num_startpositions == 2:
        note_duration = '16'
    elif num_startpositions == 3:
        note_duration = '32'
    else:
        note_duration = ''
    return note_duration


def ChordsClassifier(objectWithoutStem, objectTop, staffLineSpacing, pitches, pitches_coord):
    se = disk((staffLineSpacing) // 2)
    objectWithoutStem = np.copy(objectWithoutStem)
    eroded = binary_erosion((255 - objectWithoutStem) / 255, se)
    label_img, num = label(eroded, background=0,
                           return_num=True, connectivity=2)
    props = regionprops(label_img)
    heads = []
    note = ''
    for prop in props:
        if(prop.area != 1):
            heads.append(prop)
    if len(heads) > 1:
        note = '{'
        # show_images([eroded], ['chord'])
        for head in heads:
            # print('head centroid',head.centroid)
            headPosition = head.centroid[0] + objectTop
            # print('objecttop:', objectTop)
            note = note + pitches[find_nearest(pitches_coord, headPosition)] + '/4,' 
            # print('pitches', pitches)
            # print('pitches_coordinates', pitches_coord)
            # print('headposition: ', headPosition)
            # print('````````````````````````````')
    if note != '':
        note = note[:-1] + '}'
    return note


def chordOrBeamCheck(objectWithouStems):
    height, width = objectWithouStems.shape
    objectWithouStems = (255 - objectWithouStems)/255
    upperRect = objectWithouStems[0:height//4,:]
    lowerRect = objectWithouStems[:,3*height//4:]
    if min(np.sum(upperRect), np.sum(lowerRect)) == 0:
        return 'chord'
    else:
        return 'beam'
