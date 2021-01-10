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
import os
from skimage.color import rgb2gray, rgb2hsv
import skimage.io as io
import operator


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


def classifierB(objectWithoutStem, staffLineSpacing, objectTop, pitches, pitches_coord, accidentals):
    vertical = objectWithoutStem[:, objectWithoutStem.shape[1] // 2 + 1]
    runlengths, startpositions, values = rle(vertical)
    whiteRunHeight = runlengths[np.nonzero(values)[0]]
    whiteRunPositions = startpositions[np.nonzero(values)[0]]
    if len(whiteRunHeight) == 2:
        headPosition = (
            whiteRunPositions[0] + (whiteRunPositions[1] + whiteRunHeight[1])) // 2 + objectTop
        pitch = pitches[find_nearest(
            pitches_coord, headPosition)]
        p = pitch[0]
        d = pitch[1]

        note = p + accidentals + d + '/2'
        return note
    elif len(whiteRunHeight) == 1:
        headPosition = whiteRunPositions[0] + \
            whiteRunHeight[0] // 2 + objectTop
        pitch = pitches[find_nearest(
            pitches_coord, headPosition)]
        p = pitch[0]
        d = pitch[1]
        note = p + accidentals + d + '/4'
        return note
    return ''


def classifierC(objectWithoutStem, stems, staffLineSpacing, objectTop, pitches, pitches_coord, accidentals):
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

    pitch = pitches[find_nearest(
        pitches_coord, headPosition)]
    p = pitch[0]
    d = pitch[1]
    if oneRuns == 1:
        note = p + accidentals + d + '/8'
        return note
    elif oneRuns == 2:
        note = p + accidentals + d + '/16'
        return note
    elif oneRuns == 3:
        note = p + accidentals + d + '/32'
        return note
    return ''


def classifierA(objectWithoutStem, stems, staffLineSpacing, staffHeight, objectTop, pitches, pitches_coord, dots, accidentals):
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
                           objectTop, pitches, pitches_coord, accidentals)
    else:
        note = classifierC(objectWithoutStem, stems, staffLineSpacing,
                           objectTop, pitches, pitches_coord, accidentals)
    if note != '':
        note = note + ('.' * dots)
    return note


def beamClassifier(object, objectWithoutStem, staffLineSpacing, staffHeight, objectTop, pitches, pitches_coord, stems):
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
    for i in range(len(bboxes_centroids)):
        cent_y = bboxes_centroids[i][0]
        stem_x = stems[i][0]
        # top_staff_y = staffLines[top_block_staffLine[cnt_obj]]
        # print('pos : ' + str((cent_y - top_staff_y) // staffLineSpacing))
        headPosition = cent_y + objectTop
        notes.append(pitches[find_nearest(pitches_coord, headPosition)])
        if rectR_sum < rectL_sum:
            # treat the last note specially
            if i == len(bboxes_centroids) - 1:
                duration = calc_duration(
                    cent_y, stem_x, objectWithoutStem, 'top', 1, staffLineSpacing)
            else:
                duration = calc_duration(
                    cent_y, stem_x, objectWithoutStem, 'top', 0, staffLineSpacing)
            if duration == '':
                notes = notes[:-1]
            else:
                notes[-1] = notes[-1] + '/' + duration
        else:
            # treat the first note specially
            if i == 0:
                duration = calc_duration(
                    cent_y, stem_x, objectWithoutStem, 'bottom', 1, staffLineSpacing)
            else:
                duration = calc_duration(
                    cent_y, stem_x, objectWithoutStem, 'bottom', 0, staffLineSpacing)
            if duration == '':
                notes = notes[:-1]
            else:
                notes[-1] = notes[-1] + '/' + duration
    return notes


def calc_duration(cent_y, stem_x, object, note_pos, special, staffLineSpacing):
    if note_pos == 'bottom':
        # treat the first note specially
        min_y = int(cent_y - staffLineSpacing)
        if special == 1:
            detection_line_col = int(stem_x + 2)
        else:
            # to get to the point above the note head
            detection_line_col = int(stem_x - 2)
        detection_line = np.array(object[0:min_y, detection_line_col], ndmin=1)

    else:
        # to get to the point below the note head
        max_y = int(cent_y + 1.5 * staffLineSpacing)
        # treat the last note specially
        if special == 1:
            detection_line_col = int(stem_x - 2)
        else:
            detection_line_col = int(stem_x + 2)

        detection_line = np.array(
            object[max_y:object.shape[0], detection_line_col])

    mask = np.where(detection_line > 0, 1, 0)
    if len(mask) == 0:
        mask = np.zeros((1, 5))  # arbitrary empty mask
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
    note = []
    for prop in props:
        if(prop.area != 1):
            heads.append(prop)
    if len(heads) > 1:
        # show_images([eroded], ['chord'])
        for head in heads:
            # print('head centroid',head.centroid)
            headPosition = head.centroid[0] + objectTop
            # print('objecttop:', objectTop)
            note.append(pitches[find_nearest(
                pitches_coord, headPosition)] + '/4')
            # print('pitches', pitches)
            # print('pitches_coordinates', pitches_coord)
            # print('headposition: ', headPosition)
            # print('````````````````````````````')
    if len(note) != 0:
        note = sorted(note)
        out = "{"
        for n in note:
            out = out + n + ","
        note = out[:-1] + "}"
    else:
        note = ''
    return note


def chordOrBeamCheck(objectWithouStems):
    height, width = objectWithouStems.shape
    objectWithouStems = (255 - objectWithouStems)/255
    upperRect = objectWithouStems[0:height//4, :]
    lowerRect = objectWithouStems[3*height//4:,:]
    if min(np.sum(upperRect), np.sum(lowerRect)) == 0:
        return 'chord'
    else:
        return 'beam'

############################################################################

def read_temp():
    directory = os.fsencode("../temp")
    templates = {}
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".png") or filename.endswith(".jpg"):
            image = rgb2gray(io.imread(os.path.join('../temp/', filename)))
            if image.dtype != "uint8":
                image = (image * 255).astype("uint8")
            templates[filename[0:-4]] = image
    return templates


def check_temp(obj, tmp, accuracy):
    #show_images([tmp, obj], ["Template", "Object"])
    try:
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(tmp, None)
        kp2, des2 = sift.detectAndCompute(obj, None)
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < accuracy*n.distance:
                good.append([m])
        p = len(good)/(min(len(kp1), len(kp2)))
        #print("percentage = {}".format(p*100))
        return p*100
    except:
        #print("percentage = 0")
        return 0


def check_all_templates(obj, templates):
    show_images([obj])
    best_solution = 0
    label = None
    for tmp in templates:
        result = check_temp(obj, templates[tmp], 0.3)
        if result > best_solution:
            best_solution = result
            label = tmp
    if (best_solution > 0) and (label is not None):
        print("Matched with {}".format(label))
        return label
    else:
        print("Unmatched")
        return None


def read_temps_versions(tmp_name):
    directory = os.fsencode("temp/"+tmp_name)
    templates = {}
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".png") or filename.endswith(".jpg"):
            image = rgb2gray(
                io.imread(os.path.join('temp/'+tmp_name, filename)))
            if image.dtype != "uint8":
                image = (image * 255).astype("uint8")
            templates[filename[0:-4]] = image
    return templates


def check_match(dictionary_temp, img):
    count = 0
    for im_temp in dictionary_temp:
        if check_temp(dictionary_temp[im_temp], img, 0.5) != 0:
            count += 1
    return count


def classify_accidentals(obj, templates, staffHeight):
    dictionary_matches = {}
    dictionary_matches["2"] = check_match(templates[0], obj[0])
    dictionary_matches["4"] = check_match(templates[1], obj[0])
    dictionary_matches["&&"] = check_match(templates[2], obj[0])
    dictionary_matches["##"] = check_match(templates[3], obj[0])
    dictionary_matches["&"] = check_match(templates[4], obj[0])
    dictionary_matches["full_note"] = check_match(templates[5], obj[0])
    dictionary_matches[""] = check_match(templates[6], obj[0])
    dictionary_matches["#"] = check_match(templates[7], obj[0])
    # for mat in dictionary_matches:
    #    print(mat + " = {}".format(dictionary_matches[mat]))
    best_match = max(dictionary_matches.items(), key=operator.itemgetter(1))[0]
    t = "accidental"
    if best_match == "2" or best_match == "4":
        t = "number"
    if best_match == "full_note":
        t = "full_note"
    if dictionary_matches[best_match] == 0:
        return ("full_note", "full_note")
    if best_match == "&&" or best_match == "&":
        if dictionary_matches["&&"] >= dictionary_matches["&"]:
            return ("&&", t)
        return ("&", t)
    return (best_match, t)


def read_all_templates():
    temps_2 = read_temps_versions("2")
    temps_4 = read_temps_versions("4")
    temps_double_flat = read_temps_versions("double_flat")
    temps_double_sharp = read_temps_versions("double_sharp")
    temps_flat = read_temps_versions("flat")
    temps_full_note = read_temps_versions("full_note")
    temps_natural = read_temps_versions("natural")
    temps_sharp = read_temps_versions("sharp")

    templates = [temps_2, temps_4, temps_double_flat, temps_double_sharp,
                 temps_flat, temps_full_note, temps_natural, temps_sharp]
    return templates
