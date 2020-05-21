from PIL import Image
import numpy as np
import glob


def readFlag():
    # 1,1 pixeldeki rgb yi okuma

    normalizedPixels = []

    for file in glob.glob("./flags/*.png"):

        im = Image.open(file)
        rgb_im = im.convert('RGB')
        pix = rgb_im.getpixel

        width, height = im.size
        #print(file, "width,height", width, height)
        y = np.ceil(height/5-1)
        x = width/5

        pixels = [
            pix((x,   y)), pix((2*x,  y)),  pix((3*x,  y)),  pix((4*x,  y)),
            pix((x, 2*y)), pix((2*x, 2*y)), pix((3*x, 2*y)), pix((4*x, 2*y)),
            pix((x, 3*y)), pix((2*x, 3*y)), pix((3*x, 3*y)), pix((4*x, 3*y)),
            pix((x, 4*y)), pix((2*x, 4*y)), pix((3*x, 4*y)), pix((4*x, 4*y))
        ]

        for x in pixels:
            for y in x:
                normalizedPixels.append(y/255)

    #print("Total RGBs", len(normalizedPixels))
    return normalizedPixels


readFlag()
