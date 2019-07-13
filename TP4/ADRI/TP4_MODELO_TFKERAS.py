# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 12:24:45 2019
@author: Adrian
"""
import datetime
#import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report
from sklearn.metrics import roc_auc_score

from keras import optimizers
from sklearn.metrics import accuracy_score,recall_score,confusion_matrix
import warnings
import gc

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras import backend as K
# your code here



warnings.filterwarnings("ignore")    
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

startTime=datetime.datetime.now()

#fields=["AvSigVersion","AVProductsInstalled","IsProtected","Firewall","HasDetections"]
#df=pd.read_csv("train_etl_etapa2.csv",usecols=fields,encoding="utf-8")
df=pd.read_csv("train_etl_etapa2.csv",encoding="utf-8")
input_read_start=datetime.datetime.now()
#df=pd.read_hdf("train_etl_final.h5",start=0,stop=20000)
input_read_end=datetime.datetime.now()

print("input read time (secs):", str((input_read_end-input_read_start).total_seconds()))

print("Archivo de datos leido OK")
print("Cant. de registros:", len(df))

Y=df["HasDetections"]
X=df.drop("HasDetections",axis=1)
del df

#IMPUTACIONES
#X.loc[X.AVProductStatesIdentifier.isnull(),"AVProductStatesIdentifier"]=0
#X.loc[X.AVProductsInstalled.isnull(),"AVProductsInstalled"]=0
#X.loc[X.AVProductsEnabled.isnull(),"AVProductsEnabled"]=0
#X.loc[X.RtpStateBitfield.isnull(),"RtpStateBitfield"]=0

#NORMALIZACION/ESCALAMIENTO
scaler = StandardScaler().fit(X)
scaler.transform(X)

print("X e Y listos")

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2, random_state=30)
#del X
#del Y


print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)


print("Split train-test listo")

#with tf.device('/gpu:0'):
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=1733))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          epochs=20,
          batch_size=128)
    
y_pred=model.predict(X_test)


print("y_pred shape:", y_pred.shape)
print("y_test shape:",y_test.shape)

#y_pred=grid.best_estimator_.predict(X_test)
scores=model.evaluate(X_test,y_test)
print("Score:",round(scores[1]*100,2),"%")


#auc=roc_auc_score(y_test, y_pred)
#print("AUC: %.3f" % auc)

#print("Classification report  \n %s" %(classification_report(y_test, y_pred)))

endTime=datetime.datetime.now()
print("Proceso terminado en:" + str((endTime-startTime).total_seconds()) + " segundos")

del X_train, X_test, y_train, y_test
gc.collect()
print("READY!")
