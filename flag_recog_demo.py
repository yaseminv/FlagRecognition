import numpy as np
import readflag
# sigmoid function


def nonlin(x, deriv=False):
    if(deriv == True):
        return x*(1-x)
    return 1/(1+np.exp(-x))  # SIGMOID AKTIVASYON


# input dataset
X = np.array(readflag.readFlag())
# print(X)

# output dataset
y = np.eye(5)

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1337)

# initialize weights randomly with mean 0
syn0 = 2 * np.random.random((48, 5)) - 1
l1 = []

lr = 0.01


for i in range(5):
    for iter in range(100):

        # Input
        l0 = X  # 103x48
        # print(len(l0))

        # Hata oranı
        l1 = nonlin(np.dot(l0, syn0))  # 103x1

        # Hata oranının gerçek değerlerden çıkartılması
        l1_error = y[i] - l1  # 103x103

        # Ne kadar değişeceği
        l1_delta = l1_error * nonlin(l1, True)  # 103x103

        #
        syn0 += lr * np.dot(l0.T, l1_delta)

    # print(i, "tamamlandı. (", np.where(l1[i] == np.amax(l1[i]))[0], ")")
    # print("Önerilen Çıktı:", l1[i])
    #print("Bir sonraki adıma geçiliyor...")

print(nonlin(np.dot(X[0], syn0)))

"""
z = X[2]  # Inputlardan AD olanını al. 48 adet eleman var içinde.
a = np.dot(z, syn0)  # Bu 48 elemanı çarp
print(a)


print("Output After Training:")
print(len(l1), len(l1[0]))

print("Veri:", X[0])
print(":", l1[0])

k = np.zeros(103)

for i in range(48):
    for j in range(103):
        k[j] += X[0][i] * syn0[0][j]

print(np.where(k == np.amax(k))[0])
"""

"""
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

labels = np.zeros(103)
labels[0] = 1
# print("label", labels)
# dik konuma getiriyoruz
labels = labels.reshape(103, 1)

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
for epoch in range(1):
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

"""
