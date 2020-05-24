from PIL import Image
import numpy as np
import glob

def getPixels(inRangeOf, file):
    im = Image.open(file)
    rgb_im = im.convert('RGB')
    pix = rgb_im.getpixel

    width, height = im.size
    y = np.ceil(height/(inRangeOf+1)-1)
    x = width/(inRangeOf+1)

    pixels = []  

    for i in range(inRangeOf):
        for j in range(inRangeOf):
            pixels.append(pix((x*(i+1), y*(j+1))))

    return ([item/255 for t in pixels for item in t])

def readFlag(lemme):
    normalizedPixels = []
    print("naber")

    k = []

    files = glob.glob("./flags/*.jpg")

    for file in files:
        k.append(file[8:-4])
        normalizedPixels.append(getPixels(lemme, file))

    return [normalizedPixels, k]