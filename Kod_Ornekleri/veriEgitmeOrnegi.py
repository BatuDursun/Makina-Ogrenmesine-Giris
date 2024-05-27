# Gerekli kütüphaneleri import etme
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Veri kümesini yükleme (örneğin Iris veri kümesi)
iris = load_iris()
X = iris.data
y = iris.target

# Veriyi eğitim ve test setlerine bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Veriyi ölçeklendirme
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Modeli oluşturma ve eğitme
model = LogisticRegression()
model.fit(X_train, y_train)

# Modeli test etme
y_pred = model.predict(X_test)

# Model performansını değerlendirme
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
