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
    for file in glob.glob('./flags/real/*.jpg'):
        k.append(file.split("\\")[1][0:-4])
    return k


def createEye(x, label):
    eye = np.zeros((len(label), x))
    for i in range(len(label)):
        eye[i, label[i]] = 1

    return eye


def getPixels_OLD(inRangeOf, file):
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


def getPixels(file):
    im = Image.open(file)
    rgb_im = im.convert('RGB')
    pix = rgb_im.getpixel

    (width, height) = im.size

    points = [[1 / 12, 1 / 12], [1 / 12, 2 / 12], [1 / 12, 3 / 12],
              [2 / 12, 1 / 12], [2 / 12, 2 / 12], [2 / 12, 3 / 12],
              [3 / 12, 1 / 12], [3 / 12, 2 / 12], [3 / 12, 3 / 12],
              [2 / 12, 6 / 12], [2 / 12, 10 / 12], [6 / 12, 2 / 12],
              [5 / 12, 5 / 12], [5 / 12, 6 / 12], [5 / 12, 7 / 12],
              [6 / 12, 5 / 12], [6 / 12, 6 / 12], [6 / 12, 7 / 12],
              [7 / 12, 5 / 12], [7 / 12, 6 / 12], [7 / 12, 7 / 12],
              [10 / 12, 6 / 12], [6 / 12, 10 / 12], [10 / 12, 2 / 12],
              [10 / 12, 10 / 12]]

    pixels = []

    for i in range(len(points)):
        pixels.append(pix((points[i][0]*width, points[i][1]*height)))

    return [item / 255 for t in pixels for item in t]


def readFlags():
    normalizedPixels = []
    for file in glob.glob('./flags/*/*.jpg'):
        normalizedPixels.append(getPixels(file))
    return normalizedPixels
