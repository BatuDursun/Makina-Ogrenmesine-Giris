#!/usr/bin/env python

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

#ADIM 1: VERİYİ OKUMA VE ANLAMA

df = pd.read_csv("other_files/images_analyzed_productivity1.csv")
print(df.head())

#Productivity değerlerini sayarak Good ve Bad arasındaki bölünmeyi görün
sizes = df['Productivity'].value_counts(sort = 1)
print(sizes)

#plt.pie(sizes, autopct='%1.1f%%')
#Etiketlerin her birinin oranını bilmek iyi olur


#ADIM 2: İLGİSİZ VERİYİ ÇIKARMA
#Örneğimizde, Images_Analyzed iyi analiz veya kötü analizi yansıtıyor
#bu yüzden bunu dahil etmemeliyiz. Ayrıca, Kullanıcı numarası sadece bir sayı ve 
#verimliliğe etkisi yok, bu yüzden bunu da çıkarabiliriz.

df.drop(['Images_Analyzed'], axis=1, inplace=True)
df.drop(['User'], axis=1, inplace=True)


#ADIM 3: Gerekirse eksik değerleri işleyin
#df = df.dropna()  #En az bir eksik değere sahip tüm satırları siler. 


#ADIM 4: Gerekirse sayısal olmayan verileri sayısala çevirin.
#Bazen sayısal olmayan verilerimiz olabilir, örneğin parti adı, kullanıcı adı, şehir adı vb.
#örneğin, veriler EVET ve HAYIR şeklindeyse 1 ve 2'ye çevirin

df.Productivity[df.Productivity == 'Good'] = 1
df.Productivity[df.Productivity == 'Bad'] = 2
print(df.head())


#ADIM 5: VERİYİ HAZIRLAYIN.

#Y bağımlı değişken içeren veridir, bu Productivity sütunudur
Y = df["Productivity"].values  #Bu noktada Y bir nesnedir, int türünde değil
#Y'yi int'e dönüştür
Y=Y.astype('int')

#X bağımsız değişkenleri içeren veridir, Productivity sütunu hariç her şey
#Etiket sütununu X'ten çıkarın çünkü bunu özelliklerden biri olarak dahil etmek istemiyorsunuz
X = df.drop(labels = ["Productivity"], axis=1)  
#print(X.head())

#ADIM 6: VERİYİ EĞİTİM VE TEST VERİSİNE BÖLME.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=20)
#random_state herhangi bir tamsayı olabilir ve veri setini rastgele bölmek için bir tohum olarak kullanılır.
#Bunu yaparak her seferinde aynı test veri seti ile çalışırız, bu önemliyse.
#random_state=None her seferinde veri setini rastgele böler

#print(X_train)

#ADIM 7: Modeli tanımlama ve eğitme.

# Kullanacağımız modeli içe aktar

# Kullanacağımız modeli içe aktar
#RandomForestRegressor, regresyon tipi problemler içindir. 
#Sınıflandırma için RandomForestClassifier kullanırız.
#Her ikisi de benzer sonuçlar verir, ancak regressor için sonuç float,
#ve sınıflandırıcı için bir tamsayıdır. 
#Bu bir sınıflandırma problemi olduğu için sınıflandırıcıyı kullanalım

from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import RandomForestRegressor

# 10 karar ağacı ile modeli başlat
model = RandomForestClassifier(n_estimators = 10, random_state = 30)
# Eğitim verisi üzerinde modeli eğit
model.fit(X_train, y_train)


#ADIM 8: MODELİ TEST VERİSİNDE TAHMİN EDEREK TEST ETME
#VE DOĞRULUK SKORUNU HESAPLAMA

prediction_test = model.predict(X_test)
#print(y_test, prediction_test)

from sklearn import metrics
#Tahmin doğruluğunu yazdır
print ("Doğruluk = ", metrics.accuracy_score(y_test, prediction_test))
#Test doğruluğunu çeşitli test boyutları için deneyin ve daha fazla eğitim verisi ile nasıl iyileştiğini görün

#Random forest'ın inanılmaz özelliklerinden biri, bize özellik önemlerini sağlayabilmesidir
# Sayısal özellik önemlerini alın
#importances = list(model.feature_importances_)

#Bunları güzel bir formatta yazdıralım.

feature_list = list(X.columns)
feature_imp = pd.Series(model.feature_importances_,index=feature_list).sort_values(ascending=False)
print(feature_imp)
