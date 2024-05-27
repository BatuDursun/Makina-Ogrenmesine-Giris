import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv('musteriler.csv')
print(veriler)

X = veriler.iloc[:,3:].values

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 3, init = 'k-means++')
kmeans.fit(X)
print(kmeans.cluster_centers_)
# hacim ve maaş cinsinden orta noktaların koordinatlarını verdi

# k için optimum değeri bulma
sonuclar = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 123)
    kmeans.fit(X)
    sonuclar.append(kmeans.inertia_)    
#WSS değerleri
 
plt.plot(range(1,11), sonuclar)
# 4 güzel- 2 de olabilir(Dirsek noktası)

    





