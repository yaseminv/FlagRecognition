# 16 bayrak
# her bayrak icin 16 pixel
# her pixel icin birer r,g ve b degerleri
# feature_set 16 satir 48 sutun
# labels[0]=1 1.bayrak icin diger indexler 0 (satir)

import numpy as np
# sigara içer, obezdir, egzersiz yapar


# pip install Pillow
from PIL import Image

# 1,1 pixeldeki rgb yi okuma
im = Image.open('ad.png')
pix = im.load()
width, height = im.size
print("width,height", width, height)
y = np.ceil(height/5-1)
x = width/5
pixels = [
    pix[x, y], pix[2*x, y], pix[3*x, y], pix[4*x, y],
    pix[x, 2*y], pix[2*x, 2*y], pix[3*x, 2*y], pix[4*x, 2*y],
    pix[x, 3*y], pix[2*x, 3*y], pix[3*x, 3*y], pix[4*x, 3*y],
    pix[x, 4*y], pix[2*x, 4*y], pix[3*x, 4*y], pix[4*x, 4*y]
]

normalizedPixels = []

for x in pixels:
    for y in x:
        normalizedPixels.append(y/255)


print(normalizedPixels)

feature_set = normalizedPixels


# diyabettir (çıktı)

labels = np.eye(16)
print("label", labels)
# dik konuma getiriyoruz
labels = labels.reshape(16, 1)

# numpy için randomize seed oluştur
# böylece seed'e bağlı her seferinde
# aynı random sayıyı üretir.
np.random.seed(1337)

# ağırlıklar, bias ve öğrenme oranı
weights = np.random.rand(48, 1)
bias = np.random.rand(1)
lr = 0.01

print("Weights:", weights)
print("Bias:", bias)
print("Learning Rate:", lr)


# sigmoid fonksiyonu
# 0'da 0.5, negatifte 0'a,
# pozitifte 1'e yaklaşır.
def sigmoid(x):
    return 1/(1+np.exp(-x))


# sigmoid'in türevi
def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))


# iterasyon sayısı
for epoch in range(10000):
    inputs = feature_set

    # ileri besleme aşama 1
    XW = np.dot(feature_set, weights) + bias

    # ileri besleme aşama 2
    z = sigmoid(XW)

    # geri yayılım aşama 1
    error = z - labels

    print(error.sum())

    # geri yayılım aşama 2
    dcost_dpred = error
    dpred_dz = sigmoid_der(z)

    z_delta = dcost_dpred * dpred_dz

    inputs = feature_set.T
    weights -= lr * np.dot(inputs, z_delta)

    for num in z_delta:
        bias -= lr * num
