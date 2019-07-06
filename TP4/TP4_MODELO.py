# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 12:24:45 2019
@author: Adrian
"""
import datetime
#import numpy as np
import pandas as pd
#from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report
from sklearn.neural_network import MLPClassifier
#import xgboost as xgb
import warnings


from sklearn.metrics import accuracy_score,recall_score,confusion_matrix

 
class myScaler(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        scaler = Normalizer().fit(X)
        scaler.transform(X)
        return X

class Imputer(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on text columns in the data
    """
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        #X.dropna(thresh=10,inplace=True)
        X.loc[X.IsProtected.isnull(),"IsProtected"]=0
        #Dropear rows donde HasDetections no sea 0 o 1
        
        return X

warnings.filterwarnings("ignore")    
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

startTime=datetime.datetime.now()

df=pd.read_csv("train_clean2.csv",encoding="utf-8")

print("Archivo de datos leido OK")
print("Cant. de registros:", len(df))

Y=df["HasDetections"]
X=df.drop("HasDetections",axis=1)
del df

print("X e Y listos")

#print("X.info=", X.info())
#print(X.describe())
#print(X.isna().sum().sort_values(ascending=False))


X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2, random_state=30)
del X
del Y

print("Split train-test listo")

#xgb_model=xgb.XGBClassifier(objective="binary:logistic", random_state=42)

steps = [("imputer",Imputer("")),("scaler", myScaler("")), ("CM",RandomForestClassifier(n_estimators=150,max_depth=50,max_features="log2"))]
#steps = [("imputer",Imputer("")),("scaler", myScaler("")), ("CM",xgb_model)]
#steps = [("imputer",Imputer("")),("scaler", myScaler("")), ("CM",MLPClassifier(hidden_layer_sizes=(100,100,100),max_iter=1000))]

print("Preparando pipeline..")

pipeline = Pipeline(steps) # define the pipeline object.

#param_dist = {'max_depth': [2, 3, 4],
#              'bootstrap': [True, False],
#              'max_features': ['auto', 'sqrt', 'log2', None],
#              'criterion': ['gini', 'entropy']}
 
   
param_dist={}

grid = GridSearchCV(pipeline, param_grid=param_dist, cv=5,scoring="f1")


print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)

print("Pipeline listo para fittear")

pipe_fitstart=datetime.datetime.now()
#pipeline.fit(X_train, y_train)
grid.fit(X_train,y_train)
pipe_fitend=datetime.datetime.now()

print("Pipeline fitteado")


print("Mejor estimador:", round(grid.best_score_,2))
#print("Importancia de features:", grid.best_estimator_.feature_importances_)

#pipe_predictstart=datetime.datetime.now()
#preds = pipeline.predict(X_train)
#pipe_predictend=datetime.datetime.now()
#print("Predicciones listas recién salidas del pipeline")

pipe_scoringstart=datetime.datetime.now()

#print("score RandomForest on test:" ,round(pipeline.score(X_train,y_train),2))
#print("score RandomForest on test:" ,round(pipeline.score(X_test,y_test),2))

#y_pred=pipeline.predict(X_test)
y_pred=grid.predict(X_test)
print("Accuracy:" ,round(accuracy_score(y_test,y_pred),2))

print("Classification report  \n %s" %(classification_report(y_test, y_pred)))

pipe_scoringend=datetime.datetime.now()

endTime=datetime.datetime.now()
print("Proceso terminado en:" + str((endTime-startTime).total_seconds()) + " segundos")
print("Pipeline fit:" + str((pipe_fitend-pipe_fitstart).total_seconds()) + " segundos")
#print("Pipeline predict:" + str((pipe_predictend-pipe_predictstart).total_seconds()) + " segundos")
print("Pipeline scoring:" + str((pipe_scoringend-pipe_scoringstart).total_seconds()) + " segundos")
