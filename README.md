# Flag Recognition with Neural Networks

Bayrakların yapısı gereği tam bir uniformity halinde olmaları zordur. Ayrıca dünya üzerindeki çoğu bayrak yatay veya dikey üç renk ile ifade ediliyor. Yapacağımız yapay zeka ile öncelikle edge detection kullanarak bayrağın olup olmadığını bulacak, sonrasında renk analizi yaparak hangi bayrak olduğunu kestirecegiz. Bunları kamera yardımıyla, elimizde bayrak tutarak gerçekleştireceğiz. Veri setini, baraklari belirli koşullar altında düzenleyerek (blur, noise, oversaturation, low-light gibi) oluşturacağız. Böylece birden fazla yöntem kullanarak bayrağı tanıyacağız.

## Kaynaklar
Edge Detection Giriş: https://www.youtube.com/watch?v=P19jOyFMuwM
Edge Detection Devam: https://www.youtube.com/watch?v=UueauK_VbfA
Python Neural Network From Scratch: https://stackabuse.com/creating-a-neural-network-from-scratch-in-python/
How to build your own Neural Network from scratch in Python: https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6