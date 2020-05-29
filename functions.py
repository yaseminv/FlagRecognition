#!/usr/bin/python
# -*- coding: utf-8 -*-
from PIL import Image
import numpy as np
import glob


def sigmoid(x):
    return 1/(1+np.exp(-x))


def sigmoid_der(x):
    return sigmoid(x) * (1-sigmoid(x))


def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)


def getFlagNames():
    k = []
    for file in glob.glob('./flags/*.jpg'):
        k.append(file[8:-4])
    return k


def createEye(x):
    eye = np.zeros((x, x))
    for i in range(x):
        eye[i, i] = 1
    return eye


def getPixels(inRangeOf, file):
    im = Image.open(file)
    rgb_im = im.convert('RGB')
    pix = rgb_im.getpixel

    (width, height) = im.size
    y = np.ceil(height / (inRangeOf + 1) - 1)
    x = width / (inRangeOf + 1)

    pixels = []

    for i in range(inRangeOf):
        for j in range(inRangeOf):
            pixels.append(pix((x * (i + 1), y * (j + 1))))

    return [item / 255 for t in pixels for item in t]


def readFlags(lemme):
    normalizedPixels = []
    for file in glob.glob('./flags/*.jpg'):
        normalizedPixels.append(getPixels(lemme, file))
    return normalizedPixels
