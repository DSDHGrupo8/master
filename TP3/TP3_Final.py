# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 13:36:44 2019

@author: Adrian
"""

import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
#import matplotlib.pyplot as plt
#from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('datasets/train.csv')

X = df.drop(['price_range'],axis=1)
y = df['price_range']

# Separación entre train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

#estandarizar
scaler = Normalizer().fit(X_train)
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)

#convertir a df
X_train_std_df = pd.DataFrame(X_train_std, index=X_train.index, columns=X_train.columns)
X_test_std_df = pd.DataFrame(X_test_std, index=X_test.index, columns=X_test.columns) 

modelo_NB = MultinomialNB()
#print("X_train_std:", X_train_std)

modelo_NB.fit(X_train_std_df, y_train)

prediccion = modelo_NB.predict(X_test_std)

# Primero calculamos el accuracy general del modelo
print("accuracy NB:" , accuracy_score(y_test, prediccion))

#estandarizar
scaler2 = StandardScaler().fit(X_train)
X_train_std = scaler2.transform(X_train)
X_test_std = scaler2.transform(X_test)

#convertir a df
X_train_std_df = pd.DataFrame(X_train_std, index=X_train.index, columns=X_train.columns)
X_test_std_df = pd.DataFrame(X_test_std, index=X_test.index, columns=X_test.columns) 


mat = confusion_matrix(y_test, prediccion)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
#xticklabels=['negativo','positivo'], yticklabels=['negativo','positivo']
plt.xlabel('Etiquetas verdaderas')
plt.ylabel('Etiquetas predichas');

##elijo componentes principales
#pca = PCA(n_components = 3) 
#X_train_pca = pca.fit_transform(X_train_std_df)
#X_test_pca = pca.transform(X_test_std_df)

#modelos
svm_lin = SVC(kernel='sigmoid',degree=3,C=10, gamma='auto')
svm_rbf = SVC(kernel='rbf',C=10, gamma='auto')
svm_rbf.fit(X_train_std_df, y_train)
svm_lin.fit(X_train_std_df, y_train)

knn = KNeighborsClassifier()

best_k = int(round(math.sqrt(len(X)),0))
print("K selected:" , best_k)
knn = KNeighborsClassifier(n_neighbors=best_k)

knn.fit(X_train_std_df, y_train)
#score= knn_gscv.score(X_test_std_df, y_test)
score= knn.score(X_test_std_df, y_test)
print("Score de KNN:", score)

#Applying grid search for optimal parameters and model after k-fold validation

#parameters = [{'C':[0.01,0.1,1,10,50,100,500,1000], 'kernel':['rbf'], 'gamma': [0.1,0.125,0.15,0.17,0.2]}]
#grid_search = GridSearchCV(estimator=svm_rbf, param_grid=parameters, scoring ='accuracy',cv=10,n_jobs=-1)
#grid_search = grid_search.fit(X_train_std,y_train)
#
#best_accuracy = grid_search.best_score_
#opt_param = grid_search.best_params_
#print("best accuracy:" , best_accuracy)
#print("opt_param:" , opt_param)

#predicciones
#prediccion_svm_lin = svm_lin.predict(X_test_std_df)
#prediccion_svm_rbf = svm_rbf.predict(X_test_std_df)

#accuracy
#print(accuracy_score(y_test, prediccion_svm_lin))
#print(accuracy_score(y_test, prediccion_svm_rbf))

print("SVM-lin score:", svm_lin.score(X_test_std_df, y_test))
print("SVM-rbf score:", svm_rbf.score(X_test_std_df, y_test))










