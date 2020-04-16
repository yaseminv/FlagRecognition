# sigara içmesine, obez olmasına ve egzersiz yapmasına
# göre diyabet olup olmadığını öğrenen yapay sinir ağı

import numpy as np
# sigara içer, obezdir, egzersiz yapar
feature_set = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]])
# diyabettir (çıktı)
labels = np.array([[1, 0, 0, 1, 1]])
# dik konuma getiriyoruz
labels = labels.reshape(5, 1)

# numpy için randomize seed oluştur
# böylece seed'e bağlı her seferinde
# aynı random sayıyı üretir.
np.random.seed(1337)

# ağırlıklar, bias ve öğrenme oranı
weights = np.random.rand(3, 1)
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
