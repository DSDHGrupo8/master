# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 16:24:32 2019

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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

print("pandas version:", pd.__version__)
warnings.filterwarnings('ignore')

data = pd.read_csv('datasets/train.csv')
#print(data.head())

#Verificar datos faltantes
#print("datos faltantes:" ,data.isnull().sum().max())
#verificar balanceo de datos
#print(data['price_range'].value_counts())

#print(data.describe().T)

#corr = data.corr()
#plt.figure(figsize=(15,10))
#sns.heatmap(corr, square=True, annot=True, annot_kws={'size':8})

X = data.drop(['price_range'],axis=1)

# Remove non-ordinal
X = X.drop(['blue', 'dual_sim', 'four_g', 'three_g', 'touch_screen', 'wifi'],axis=1)
# Remove colinearity
X = X.drop(['fc', 'px_width', 'sc_w'],axis=1)
# Remove low variance
X = X.drop(['m_dep', 'clock_speed'],axis=1)


y = data['price_range']

#print(X.var())


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

scaler = StandardScaler().fit(X_train)
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)

#Converting numpy array to dataframe
X_train_std_df = pd.DataFrame(X_train_std, index=X_train.index, columns=X_train.columns)
X_test_std_df = pd.DataFrame(X_test_std, index=X_test.index, columns=X_test.columns) 

knn = KNeighborsClassifier()

best_k = int(round(math.sqrt(len(X)),0))
print("K selected:" , best_k)
knn = KNeighborsClassifier(n_neighbors=best_k)

#create a dictionary of all values we want to test for n_neighbors
#param_grid = {'n_neighbors': np.arange(1, 25)}
##use gridsearch to test all values for n_neighbors
#knn_gscv = GridSearchCV(knn, param_grid, cv=5)
#
#knn_gscv.fit(X_train_std_df, y_train)
knn.fit(X_train_std_df, y_train)
#score= knn_gscv.score(X_test_std_df, y_test)
score= knn.score(X_test_std_df, y_test)
print("Score de KNN:", score)


