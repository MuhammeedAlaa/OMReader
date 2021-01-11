import difflib
import os

#function to calculate difference of code output with expected output
filenames = []
TAFilesPath = './dataset/gt' 
ourOutputPath = './outputs'
for file in os.listdir(TAFilesPath):
    filename = os.fsdecode(file)
    filenames.append(filename)
    f = open(os.path.join(TAFilesPath, filename), "r")
    expected = f.readlines()
    f.close()
    f = open(os.path.join(ourOutputPath, filename), "r")
    result = f.readlines()
    f.close()
    differ = difflib.ndiff(expected, result, charjunk=difflib.IS_CHARACTER_JUNK)
    for line in differ:
        print(line)

    print('``````````````````````````````````````````````````')
