# 16 bayrak
# her bayrak icin 16 pixel
# her pixel icin birer r,g ve b degerleri
# feature_set 16 satir 48 sutun
# labels[0]=1 1.bayrak icin diger indexler 0 (satir)

import numpy as np
# sigara içer, obezdir, egzersiz yapar
import readflag
# pip install Pillow


feature_set = np.asarray(readflag.readFlag())
# diyabettir (çıktı)

print(feature_set)

labels = np.zeros(4944)
labels[0] = 1
#print("label", labels)
# dik konuma getiriyoruz
labels = labels.reshape(4944, 1)

# numpy için randomize seed oluştur
# böylece seed'e bağlı her seferinde
# aynı random sayıyı üretir.
np.random.seed(1337)

# ağırlıklar, bias ve öğrenme oranı
weights = np.random.rand(4944, 1)
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
for epoch in range(100):
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
