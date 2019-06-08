# -*- coding: utf-8 -*-
"""
Created on Sat May 25 16:28:25 2019

@author: Adrian
"""

from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = 10, 10

import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, Ridge, RidgeCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('datasets\\properati_caballito.csv')
cols = ['Unnamed: 0', 'Unnamed: 0.1','place_name', 'precio_m2_usd', 'expenses', 'description', 'title', 'property_type']
df.drop(cols, axis=1, inplace=True)

# DF
df = df.dropna()
print(df.columns) 
print(df.shape)
df.head(5)

dummycols = [col for col in df if col.startswith('dummy_')]
distcols=[col for col in df if col.startswith('dist')]
cols=['lat', 'lon', 'surface_total_in_m2'] + dummycols + distcols
dfX = df[cols]

y = df['price_usd_per_m2']

X = StandardScaler().fit_transform(dfX)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=53)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# Generamos un grid de $\alpha$ para probar e instanciamos un particionador del Training Set 
# en K partes para realizar la validación cruzada

al_ridge = np.linspace(0.001, 2, 300)
al_lasso = np.linspace(0.001, 2, 300)
kf = KFold(n_splits=5, shuffle=True, random_state=12)

# Instanciamos los modelos
lm = LinearRegression()
lmRidgeCV = RidgeCV(alphas=[0.1], cv=kf, normalize=False)
lmLassoCV = LassoCV(alphas=al_lasso, cv=kf, normalize=False)

#lmRidgeCV = RidgeCV(fit_intercept=False, alphas=[0.1], cv=kf, normalize=False)
# Hacemos los fits respectivos

lm.fit(X_train, y_train)
lmRidgeCV.fit(X_train, y_train)
lmLassoCV.fit(X_train, y_train)


print('Alpha Ridge:',lmRidgeCV.alpha_,'\n'
      'Alpha LASSO:',lmLassoCV.alpha_,'\n')

# Calculamos el R2

print("Score Train Lineal:", lm.score(X_train, y_train),"\n"
      "Score Train Ridge:",  lmRidgeCV.score(X_train, y_train),"\n"
      "Score Train Lasso:",  lmLassoCV.score(X_train, y_train))

# Calculamos el MSE

lmpred_Tr = lm.predict(X_train)
lmRidgepred_Tr = lmRidgeCV.predict(X_train)
lmLassoepred_Tr = lmLassoCV.predict(X_train)

print("Train MSE lineal=", mean_squared_error(y_train,lmpred_Tr), "\n"
      "Train MSE Ridge=",  mean_squared_error(y_train,lmRidgepred_Tr), "\n"
      "Train MSE Lasso=",  mean_squared_error(y_train,lmLassoepred_Tr))



