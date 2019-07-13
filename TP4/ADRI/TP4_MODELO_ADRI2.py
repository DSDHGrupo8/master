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
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures
import xgboost as xgb
import warnings
import gc


from sklearn.metrics import accuracy_score,recall_score,confusion_matrix

 
class myScaler(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        scaler = StandardScaler().fit(X)
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
#        X.loc[X.AVProductStatesIdentifier.isnull(),"AVProductStatesIdentifier"]=0
#        X.loc[X.AVProductsInstalled.isnull(),"AVProductsInstalled"]=0
#        X.loc[X.AVProductsEnabled.isnull(),"AVProductsEnabled"]=0
#        X.loc[X.RtpStateBitfield.isnull(),"RtpStateBitfield"]=0
        #X.loc[X.UacLuaenable.isnull(),"UacLuaenable"]=0
        #X.loc[X.IsProtected.isnull(),"IsProtected"]=0
        #X.loc[X.Firewall.isnull(),"Firewall"]=0
        #X.loc[X.SMode.isnull(),"SMode"]=0
        #X.loc[X.IsProtected.isnull(),"IsProtected"]=0
        #Dropear rows donde HasDetections no sea 0 o 1
        
        return X

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

print("X e Y listos")

#print("X.info=", X.info())
#print(X.describe())
#print(X.isna().sum().sort_values(ascending=False))


#
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2, random_state=30)
del X
del Y

poly_features = PolynomialFeatures(degree=2)

# transforms the existing features to higher degree features.
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly=poly_features.fit_transform(X_test)

print("Split train-test listo")

model=RandomForestClassifier(n_estimators=100,max_depth=25,max_features="log2")
#model=xgb.XGBClassifier(objective="binary:logistic", random_state=42)                             
#model=KNeighborsClassifier(n_neighbors=2)                        
#model=MultinomialNB()     
#model=MLPClassifier(hidden_layer_sizes=(100,50,50,1),max_iter=100)                             

steps = [("imputer",Imputer("")),("scaler", myScaler("")), ("CM",model)]


print("Preparando pipeline..")

pipeline = Pipeline(steps) # define the pipeline object.

#param_dist = {'max_depth': [2, 3, 4],
#              'bootstrap': [True, False],
#              'max_features': ['auto', 'sqrt', 'log2', None],
#              'criterion': ['gini', 'entropy']}
 
   
param_dist={}

grid = GridSearchCV(pipeline, param_grid=param_dist, cv=5,scoring="accuracy")


print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)

print("X_train_poly shape:", X_train_poly.shape)
print("y_train shape:", y_train.shape)
print("X_test_poly shape:", X_test_poly.shape)


print("Modelo:", type(model))
print("Pipeline listo para fittear")

pipe_fitstart=datetime.datetime.now()
#pipeline.fit(X_train, y_train)
#grid.fit(X_train,y_train)
grid.fit(X_train_poly,y_train)
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
#y_pred=grid.best_estimator_.predict(X_test)
y_pred=grid.best_estimator_.predict(X_test_poly)
print("Accuracy:" ,round(accuracy_score(y_test,y_pred),2))

auc=roc_auc_score(y_test, y_pred)
print("AUC: %.3f" % auc)

print("Classification report  \n %s" %(classification_report(y_test, y_pred)))

pipe_scoringend=datetime.datetime.now()

endTime=datetime.datetime.now()
print("Proceso terminado en:" + str((endTime-startTime).total_seconds()) + " segundos")
print("Pipeline fit:" + str((pipe_fitend-pipe_fitstart).total_seconds()) + " segundos")
#print("Pipeline predict:" + str((pipe_predictend-pipe_predictstart).total_seconds()) + " segundos")
print("Pipeline scoring:" + str((pipe_scoringend-pipe_scoringstart).total_seconds()) + " segundos")

del X_train, X_test, y_train, y_test
gc.collect()
print("READY!")
