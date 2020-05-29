import numpy as np
import functions as f
import pickle

# Ayarlar
epoch_count = 10000
hidden_nodes = 16
print_count = 20
flag_division = 4
learning_rate = 0.01

flagNames = f.getFlagNames()
flagCount = len(flagNames)

# Rastgele bir seed oluştur.
# Bu işlem, rastgele sayı oluşturmanın tahmin
# edilebilir olmasını sağlar. Uygulama her
# başladığında, ilk rastgele ürettiği sayı
# 0.37454011884 olacaktır.
np.random.seed(42)

# Giriş Nöronları
feature_set = np.array(f.readFlags(flag_division))

# Çıkış Nöronlarının kontrolü için birim matris üretir
one_hot_labels = np.array(f.createEye(flagCount))

instances = feature_set.shape[0]  # 100, Giriş olarak verilen kaç bayrak var?
attributes = feature_set.shape[1]  # 48, Her bir bayrak için verilen veri
output_labels = flagCount  # 100, Çıkış olarak istenen kaç bayrak var?

# Hidden layer için rastgele weight ve bias üretir
wh = np.random.rand(attributes, hidden_nodes)
bh = np.random.randn(hidden_nodes)

# Output layer için rastgele weight ve bias üretir
wo = np.random.rand(hidden_nodes, output_labels)
bo = np.random.randn(output_labels)

for epoch in range(epoch_count):

    # Kullanılan karakterler:
    # w: Weight
    # b: Bias
    # d: Derivative
    # h: Hidden Layer
    # o: Output Layer
    # a: Activation
    # z: Value
    # cost: Cost
    # _: Division

    # İleri Besleme (Feed Forward)

    # Faz 1
    # Giriş değerlerimizi weight ile çarpıp bias ekliyoruz.
    # Böylece her bir bayrak için hidden node kadar değer
    # elde etmiş oluyoruz.
    zh = np.dot(feature_set, wh) + bh  # 100x48 . 48x16 = 100x16

    # Elde ettiğimiz bu değerin aktivasyon fonksiyonunu ne
    # kadar tetiklediğini buluyoruz.
    ah = f.sigmoid(zh)  # 100x16

    # Faz 2
    # Hidden node'lardan elde ettiğimiz değerleri weight ile
    # çarpıp bias ekliyoruz. Böylece her bir bayrak için output
    # node kadar değer elde etmiş oluyoruz.
    zo = np.dot(ah, wo) + bo  # 100x16 . 16x100 = 100x100

    # Elde ettiğimiz bu değerin aktivasyon fonksiyonunu ne
    # kadar tetiklediğini buluyoruz.
    ao = f.softmax(zo)  # 100x100

    # Geri Yayılım (Back Propagation)

    # Faz 1
    # Elde ettiğimiz sonucun, gerçek sonuç ile arasındaki
    # fark bulunur.
    dcost_dzo = ao - one_hot_labels  # 100x100
    # Not: Burada değer negatifse bulunması gereken değerden az,
    # pozitif ise bulunması gereken değerden fazla bulduğumuzu
    # anlayabiliriz.

    # Ara katmandaki nöronların ne kadar aktive olduğu
    # alınır. Burada türev, gizli katmandan gelen çıktılardır.
    dzo_dwo = ah  # 100x16

    # Bu iki matris çarpılarak, hidden layerdan çıkan
    # weightlerin ne kadar değişmesi gerektiği bulunur.
    # dcost/dwo = ao - y
    dcost_wo = np.dot(dzo_dwo.T, dcost_dzo)  # 16x100 . 100x100 = 16x100

    # Son olarak biaslar, elde ettiğimiz sonucun ne kadar
    # farklı çıktığına göre düzenlenir.
    # dcost/dbo = dcost/dao * dao/dzo * dzo/dbo
    # dcost/dbo = ao - y
    dcost_bo = dcost_dzo  # 100x100

    # Faz 2
    # Hidden layer - Output layer arasındaki weightler alınır
    # dzo/dah = wo
    dzo_dah = wo  # 16x100

    # Gerçek sonuç ve elde ettiğimiz değerler aradaki
    # farkın, çıktılara ne kadar etki ettiği hesaplanır
    # dcost/dah = dcost/dzo * dzo/dah
    dcost_dah = np.dot(dcost_dzo, dzo_dah.T)  # 100x100 . 100x16 = 100x16

    # Aktivasyon fonksiyonuna giren değerlerin, bu gizli
    # katmandaki nöronlara ne kadar etkisi olduğu hesaplanır
    # dah/dzh = sigmoid(zh) * (1-sigmoid(zh))
    dah_dzh = f.sigmoid_der(zh)  # 100x16

    # Hidden layer'a yapılan etki, aktivasyon
    # fonksiyonunu çağıran input*weight olduğundan,
    # dzh/dwh = input features
    dzh_dwh = feature_set  # 100x48

    # Her bir bayrak değerinin gizli katmana ne kadar
    # etki ettiği bulunur
    dcost_wh = np.dot(dzh_dwh.T, dah_dzh * dcost_dah)
    # 48x100 . 100x16 = 48x16

    # dcost/dbh = dcost/dah * dah/dzh
    # Çünkü, dzh/dbh = 1
    dcost_bh = dcost_dah * dah_dzh  # 100x16

    # Son olarak ağırlıkları güncelliyoruz

    wh -= learning_rate * dcost_wh  # 48x16
    bh -= learning_rate * dcost_bh.sum(axis=0)  # 16

    wo -= learning_rate * dcost_wo  # 16x100
    bo -= learning_rate * dcost_bo.sum(axis=0)  # 100

    # Loss fonksiyonu tasarlanan modelin hata oranını, aynı
    # zamanda başarımını ölçen fonksiyondur. Derin ağların
    # son katmanı loss fonksiyonun tanımlandığı katmandır.
    # Loss fonksiyonu, hata hesaplama işini problemi bir
    # optimizasyon problemine dönüştürerek yaptığı için
    # optimizasyon terminolojinde kullanılan cost function
    # olarak da tanımlanmaktadır.

    # Biz bu projede hata fonksiyonu olarak min square error
    # yerine çarpraz entropi kullandık.

    # Bunu kullanmamızdaki amaç, bizimki gibi çok sayıda
    # veriyi sınıflandırmak için kullanılan softmax ve
    # sigmoid fonksiyonlarının, çarpraz entropi fonksiyonu
    # ile daha hızlı çalışması ve daha doğru sonuç vermesiydi.

    if epoch % (epoch_count/print_count) == 0:
        loss = np.sum(-one_hot_labels * np.log(ao))
        print('Kayıp fonksiyon değeri: ', loss)

# Çıktı verilerini daha sonra okumak için dosyaya yaz
pickle_out = open("rick.pickle", "wb")
pickle.dump([wh, bh, wo, bo], pickle_out)
pickle_out.close()
