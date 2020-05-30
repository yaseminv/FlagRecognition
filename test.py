import numpy as np
import functions as f
import pickle
import glob

pickle_in = open("rick.pickle", "rb")
wh, bh, wo, bo = pickle.load(pickle_in)
flagNames = f.getFlagNames()

for file in glob.glob("./tests/*.jpg"):
    fileName = file[8:-4]
    np.set_printoptions(suppress=True)
    zh = np.dot(np.vstack([f.getPixels(file)]), wh) + bh
    ah = f.sigmoid(zh)
    zo = np.dot(ah, wo) + bo
    ao = f.softmax(zo)
    flagIndex = np.where(ao == np.amax(ao))[1][0]
    print(fileName, flagNames[flagIndex], int(
        round(np.amax(ao), 2)*100), "percent")
