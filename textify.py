import numpy as np
import glob


def textify(x):
    names = []
    index = 0
    for file in glob.glob("./flags/*.jpg"):
        names.append(file[8: -4])
        print(file[8: -4], ": ", x[index])
        index = index+1
