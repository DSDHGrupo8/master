# -*- coding: utf-8 -*-
"""
Created on Thu May 23 19:07:33 2019

@author: Adrian
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistics import mode 
from matplotlib import cm as cm
from sklearn import svm
import seaborn as sns
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import sys

def sum_mod(model, X):
    a = pd.DataFrame(model.coef_ , X.columns.values)
    a = a.append(pd.DataFrame([model.intercept_, model.score(X, y)], index=['Intecept','R2']))
    return(a)

pd.set_option('display.max_columns', 500)

df = pd.read_csv('properati_caballito_train.csv')
cols = ['Unnamed: 0', 'Unnamed: 0.1','place_name', 'precio_m2_usd']
df.drop(cols, axis=1, inplace=True)
df.head(5)

print(df.info())

df.describe()

sns.distplot(df.loc[:,'price_usd_per_m2'], kde=False, bins= 100);
plt.axvline(0, color="k", linestyle="--");

sns.distplot(df.loc[:,'surface_total_in_m2'], kde=False, bins= 100);
plt.axvline(0, color="k", linestyle="--");

print(df.shape)

k = 36
cols = df.corr().nlargest(k,'price_usd_per_m2')['price_usd_per_m2'].index
corr=df.corr()
#print("cols:", cols)
print("corr:", corr)
#sys.exit(0)
#exit()   

cm = df[cols].corr()
#plt.figure(figsize=(20,15))
#sns.heatmap(cm, annot=True, cmap = 'OrRd');

#DF MINI
df_mini_Y = df.dropna()

#df_mini_Y.isna().sum()
print(df_mini_Y.shape)

cols = ['property_type','price_aprox_usd',
       'price_usd_per_m2', 'description', 'title']
df_mini_X = df_mini_Y.drop(cols, axis=1)
df_mini_X.columns

k = 36
cols = df_mini_Y.corr().nlargest(k,'price_usd_per_m2')['price_usd_per_m2'].index
cm = df_mini_Y[cols].corr()
#plt.figure(figsize=(20,15))
#sns.heatmap(cm, annot=True, cmap = 'OrRd');

# Seleccionar modelo
# Elegir hiperparámetros
model = LinearRegression(fit_intercept=True)
# Preparar los datos en una matriz de features
# Crear X e y
X = df_mini_X #2 dimensiones
y = df_mini_Y['price_usd_per_m2'] #1 dimensión

# Split entrenamiento / testeo para CV

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=1)
#Ajustar el modelo a los datos
model.fit(Xtrain, ytrain)
print (model.coef_)
print (model.intercept_)

#Evaluar
ypred = model.predict(Xtest)

print ('MAE:', metrics.mean_absolute_error(ytest, ypred))
print ('MSE:', metrics.mean_squared_error(ytest, ypred))
print ('RMSE:', np.sqrt(metrics.mean_squared_error(ytest, ypred)))
print ('R2:', metrics.r2_score(ytest, ypred))



lm = linear_model.LinearRegression()
X = df_mini_X
y = df_mini_Y['price_usd_per_m2']
model = lm.fit(X, y)
predictions = model.predict(X)

# Plot the model
plt.plot(y,y, '-.',c='grey')
plt.scatter(predictions, y, s=30, c='r', marker='+', zorder=10)
plt.xlabel("Predicted Values from all values")
plt.ylabel("Actual Values")
plt.show()
print ("MSE:", mean_squared_error(y, predictions))
print (sum_mod(model, X))

cols = ['property_type','price_aprox_usd',
       'price_usd_per_m2', 'description', 'title','expenses']
df_mini_X_sin_expensas = df_mini_Y.drop(cols, axis=1)

# Seleccionar modelo

# Elegir hiperparámetros
model = LinearRegression(fit_intercept=True)
# Preparar los datos en una matriz de features
# Crear X e y
Xb = df_mini_X_sin_expensas #2 dimensiones
yb = df_mini_Y['price_usd_per_m2'] #1 dimensión

# Split entrenamiento / testeo para CV

Xtrain, Xtest, ytrain, ytest = train_test_split(Xb, yb, random_state=1)
#Ajustar el modelo a los datos
model.fit(Xtrain, ytrain)
print (model.coef_)
print (model.intercept_)

#Evaluar
ypred = model.predict(Xtest)

print ('MAE:', metrics.mean_absolute_error(ytest, ypred))
print ('MSE:', metrics.mean_squared_error(ytest, ypred))
print ('RMSE:', np.sqrt(metrics.mean_squared_error(ytest, ypred)))
print ('R2:', metrics.r2_score(ytest, ypred))

# Y
cols = ['property_type', 'description', 'title', 'expenses']
df_no_expenses_Y = df.drop(cols, axis=1)
df_no_expenses_Y = df_no_expenses_Y.dropna()
print(df_no_expenses_Y.columns) 
print(df_no_expenses_Y.shape)

# X
cols = ['price_aprox_usd','price_usd_per_m2']
df_no_expenses_X = df_no_expenses_Y.drop(cols, axis=1)
print(df_no_expenses_X.columns)
print(df_no_expenses_X.shape)

k = 36
cols = df_no_expenses_Y.corr().nlargest(k,'price_usd_per_m2')['price_usd_per_m2'].index
cm = df_no_expenses_Y[cols].corr()
#plt.figure(figsize=(20,15))
#sns.heatmap(cm, annot=True, cmap = 'OrRd');

# Seleccionar modelo
from sklearn.linear_model import LinearRegression
# Elegir hiperparámetros
model_1 = LinearRegression(fit_intercept=True)
# Preparar los datos en una matriz de features
# Crear X e y
X1 = df_no_expenses_X #2 dimensiones
y1 = df_no_expenses_Y['price_aprox_usd'] #1 dimensión

# Split entrenamiento / testeo para CV
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X1, y1, random_state=1)
#Ajustar el modelo a los datos
model_1.fit(Xtrain, ytrain)
print (model_1.coef_)
print (model_1.intercept_)

#Evaluar
ypred = model_1.predict(Xtest)
from sklearn import metrics
print ('MAE:', metrics.mean_absolute_error(ytest, ypred))
print ('MSE:', metrics.mean_squared_error(ytest, ypred))
print ('RMSE:', np.sqrt(metrics.mean_squared_error(ytest, ypred)))
print ('R2:', metrics.r2_score(ytest, ypred))

lm = linear_model.LinearRegression()
Xa = df_no_expenses_X
ya = df_no_expenses_Y['price_aprox_usd']
model = lm.fit(Xa, ya)
predictions = model.predict(Xa)

# Plot the model
plt.plot(ya,ya, '-.',c='grey')
plt.scatter(predictions, ya, s=30, c='r', marker='+', zorder=10)
plt.xlabel("Predicted Values from all values")
plt.ylabel("Actual Values")
plt.show()
print ("MSE:", mean_squared_error(ya, predictions))
#print (sum_mod(model, Xa))

#LASSO sobre DS sin expensas
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
model = Lasso()
model

df_random = df.sample(frac=1,random_state = 42)
cols = ['property_type', 'description', 'title', 'expenses']
df_no_expenses_Y_lasso = df_random.drop(cols, axis=1)
df_no_expenses_Y_lasso = df_no_expenses_Y_lasso.dropna()
print (df_no_expenses_Y_lasso.shape)

cols = ['price_aprox_usd','price_usd_per_m2']
df_no_expenses_X_lasso = df_no_expenses_Y_lasso.drop(cols, axis=1)
print (df_no_expenses_X_lasso.shape)

Y_lasso = df_no_expenses_Y_lasso
X_lasso = df_no_expenses_X_lasso

results = cross_val_score(model,X_lasso,Y_lasso,cv=5)
print(results)
print(np.mean(results))
print(np.std(results))

model.fit(X_lasso,Y_lasso)
model.coef_
X_lasso.columns
model.alpha_









