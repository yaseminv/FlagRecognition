import numpy as np
import readflag
import pickle
import glob

def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)

def sigmoid(x):
    return 1/(1+np.exp(-x))

files = glob.glob("./tests/*.jpg")

pickle_in = open("rick.pickle","rb")
wh, bh, wo, bo = pickle.load(pickle_in)

z = readflag.readFlag(6)[1]

for file in files:
    fileName = file[8:-4]
    np.set_printoptions(suppress=True)
    zh = np.dot(np.vstack([readflag.getPixels(4, file)]), wh) + bh
    ah = sigmoid(zh)
    zo = np.dot(ah, wo) + bo
    ao = softmax(zo)
    flagIndex = np.where(ao == np.amax(ao))[1][0]
    print(z[flagIndex], round(np.amax(ao),2)*100, "percent")