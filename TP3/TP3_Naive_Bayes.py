# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 13:20:11 2019

@author: Adrian
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


df = pd.read_csv('datasets/train.csv')
df.head()

df.shape
df.columns

df['price_range'].values
X = df.drop(['price_range'],axis=1)
y = df['price_range']

# Separación entre train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

#from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

modelo_NB = MultinomialNB()
modelo_NB.fit(X_train, y_train)
prediccion = modelo_NB.predict(X_test)

# Primero calculamos el accuracy general del modelo
accuracy_score(y_test, prediccion)
mat = confusion_matrix(y_test, prediccion)
mat.shape

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
#xticklabels=['negativo','positivo'], yticklabels=['negativo','positivo']
plt.xlabel('Etiquetas verdaderas')
plt.ylabel('Etiquetas predichas');

#SVC
#estandarizar
scaler = StandardScaler().fit(X_train)
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)

#convertir a df
X_train_std_df = pd.DataFrame(X_train_std, index=X_train.index, columns=X_train.columns)
X_test_std_df = pd.DataFrame(X_test_std, index=X_test.index, columns=X_test.columns) 

#modelos
svm_lin = SVC(kernel='sigmoid',degree=3,C=10, gamma='auto')
svm_rbf = SVC(kernel='rbf',C=10, gamma='auto')
svm_rbf.fit(X_train_std_df, y_train)
svm_lin.fit(X_train_std_df, y_train)

#predicciones
prediccion_svm_lin = svm_lin.predict(X_test_std_df)
prediccion_svm_rbf = svm_rbf.predict(X_test_std_df)

#accuracy
print(accuracy_score(y_test, prediccion_svm_lin))
print(accuracy_score(y_test, prediccion_svm_rbf))

print("SVM-lin score:", svm_lin.score(X_test_std_df, y_test))
print("SVM-rbf score:", svm_rbf.score(X_test_std_df, y_test))

