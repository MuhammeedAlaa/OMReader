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


def ChordsClassifier(objectWithoutStem, objectTop, blockNumber, staffLines, staffLineSpacing):
  pitches, pitches_coord = getPitchesCoordinates(staffLineSpacing, staffLines, blockNumber)
  se = disk((staffLineSpacing-2) // 2 - 1)
  eroded = binary_erosion((255 - objectWithoutStem) / 255, se)
  label_img, num = label(eroded, background=0, return_num=True, connectivity=2)
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
      # print('staffLines: ', staffLines)
      print(pitches[find_nearest(pitches_coord, headPosition)])
      # print('pitches', pitches)
      # print('pitches_coordinates', pitches_coord)
      # print('headposition: ', headPosition)
  return len(heads) > 1

def find_nearest(array,value):
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
  maxLedgers = 2 #TODO: if you increase max ledgers modify the pitches array
  pitches_coord = []
  pitches_coord.append(-maxLedgers*staffLineSpacing + staffLines[blockNumber * 5])
  for i in range(2 * maxLedgers - 1):
    pitches_coord.append(pitches_coord[-1]+staffLineSpacing/2)
  for i in range(5):
    currentStaffIndex = blockNumber * 5 + i
    currentStaffPosition = staffLines[currentStaffIndex]
    pitches_coord.append(currentStaffPosition)
    staffSpacing = staffLineSpacing
    if i != 4:
      staffSpacing = staffLines[currentStaffIndex + 1] - currentStaffPosition
    pitches_coord.append(currentStaffPosition + staffSpacing/2)
  for i in range(2 * maxLedgers - 1):
    pitches_coord.append(pitches_coord[-1]+staffLineSpacing/2)
  pitches = ['c3', 'b2', 'a2', 'g2', 'f2', 'e2', 'd2', 'c2', 'b1', 'a1', 'g1', 'f1', 'e1', 'd1', 'c1', 'b0', 'a0']
  return pitches, pitches_coord

def classifierB(objectWithoutStem, staffLineSpacing,  staffLines, blockNumber, objectTop):
  pitches, pitches_coord = getPitchesCoordinates(staffLineSpacing, staffLines, blockNumber)
  se = disk((staffLineSpacing-2) // 2 - 1)
  eroded = binary_erosion(objectWithoutStem, se)
  show_images([eroded], ['classifier B'])
  label_img, num = label(eroded, background=0, return_num=True, connectivity=2)
  props = regionprops(label_img)
  heads = []
  for prop in props:
    if(prop.area != 1):
      heads.append(prop)
  if len(heads) == 0:
    return
  headPosition = heads[0].centroid[0] + objectTop

  vertical = objectWithoutStem[:,objectWithoutStem.shape[1] // 2 + 1]
  oneRuns = runs_of_ones_list(vertical)   
  if len(oneRuns) == 2:
    print(str(pitches[find_nearest(pitches_coord, headPosition)])+'/2')
  elif len(oneRuns) == 1:
    print(str(pitches[find_nearest(pitches_coord, headPosition)])+'/4')

def classifierC(objectWithoutStem, stems, staffLineSpacing, staffLines, blockNumber, objectTop):
  pitches, pitches_coord = getPitchesCoordinates(staffLineSpacing, staffLines, blockNumber)
  se = disk((staffLineSpacing-2) // 2 - 1)
  eroded = binary_erosion(objectWithoutStem, se)
  show_images([eroded], ['classifier C'])
  label_img, num = label(eroded, background=0, return_num=True, connectivity=2)
  props = regionprops(label_img)
  heads = []
  for prop in props:
    if(prop.area != 1):
      heads.append(prop)
  if len(heads) == 0:
    return
  headPosition = heads[0].centroid[0] + objectTop

  stemPos, stemWidth = stems[0]
  verticalLeftStem = objectWithoutStem[:,stemPos - 1]
  verticalRightStem = objectWithoutStem[:,stemPos + stemWidth + 1]
  oneRunsLeftStem = runs_of_ones_list(verticalLeftStem)
  oneRunsRightStem = runs_of_ones_list(verticalRightStem)
  oneRuns = max(len(oneRunsLeftStem), len(oneRunsRightStem))  
  if oneRuns == 1:
    print(str(pitches[find_nearest(pitches_coord, headPosition)])+'/8')
  elif oneRuns == 2:
    print(str(pitches[find_nearest(pitches_coord, headPosition)])+'/16')
  elif oneRuns == 3:          
    print(str(pitches[find_nearest(pitches_coord, headPosition)])+'/32')

def classifierA(objectWithoutStem, stems, staffLineSpacing, staffHeight, staffLines, blockNumber, objectTop):
  objectWithoutStem = (255-objectWithoutStem)/255
  se = np.zeros((staffHeight, staffHeight))
  se[:,staffHeight//2] = 1
  objectWithoutStem = binary_erosion(objectWithoutStem,se)
  horizontal = objectWithoutStem[objectWithoutStem.shape[0] // 2 + 1,:]
  oneRuns = runs_of_ones_list(horizontal)
  if len(oneRuns) == 0:
    classifierB(objectWithoutStem, staffLineSpacing,  staffLines, blockNumber, objectTop)
  else:
    classifierC(objectWithoutStem, stems, staffLineSpacing, staffLines, blockNumber, objectTop)



