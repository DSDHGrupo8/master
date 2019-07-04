# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 12:24:45 2019

@author: Adrian
"""
import datetime
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

class colPreProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        fields=["ProductName","EngineVersion","AvSigVersion",
        "AVProductStatesIdentifier","AVProductsInstalled","AVProductsEnabled","HasTpm","CountryIdentifier",
        "CityIdentifier","Platform","Processor","OsVer","OsPlatformSubRelease","IsProtected","SMode",
        "SmartScreen","Firewall","Census_DeviceFamily","RtpStateBitfield","IsSxsPassiveMode",
        "Census_OSArchitecture", "Census_OSWUAutoUpdateOptionsName","Census_IsPortableOperatingSystem",
        "Census_GenuineStateName","Census_IsSecureBootEnabled","UacLuaenable"]

        #Recortamos el dataset a los fields necesarios
        X=X[fields]
        
        
        #Dummificar las columnas de texto
        #Una vez dummificadas dropear las viejas columnas de texto
        
        #Dummificar ProductName
        dummies_productName=pd.get_dummies(X['ProductName'],prefix='dummy_ProductName',drop_first=True)
        X=pd.concat([X,dummies_productName],axis=1)
       
        X.drop("ProductName",axis=1,inplace=True)

        #Dummificar OsVer
        dummies_OsVer=pd.get_dummies(X['OsVer'],prefix='dummy_OsVer',drop_first=True)
        X=pd.concat([X,dummies_OsVer],axis=1)

        X.drop("OsVer",axis=1,inplace=True)

        #Dummificar EngineVersion
        dummies_EngineVersion=pd.get_dummies(X['EngineVersion'],prefix='dummy_EngineVersion',drop_first=True)
        X=pd.concat([X,dummies_EngineVersion],axis=1)

        X.drop("EngineVersion",axis=1,inplace=True)

        #Dummificar AvSigVersion
        dummies_AvSigVersion=pd.get_dummies(X['AvSigVersion'],prefix='dummy_AvSigVersion',drop_first=True)
        X=pd.concat([X,dummies_AvSigVersion],axis=1)

        X.drop("AvSigVersion",axis=1,inplace=True)

        #Dummificar Platform
        dummies_Platform=pd.get_dummies(X['Platform'],prefix='dummy_Platform',drop_first=True)
        X=pd.concat([X,dummies_Platform],axis=1)

        X.drop("Platform",axis=1,inplace=True)

        #Dummificar Processor
        dummies_Processor=pd.get_dummies(X['Processor'],prefix='dummy_Processor',drop_first=True)
        X=pd.concat([X,dummies_Processor],axis=1)

        X.drop("Processor",axis=1,inplace=True)

        #Dummificar OsPlatformSubRelease
        dummies_OsPlatformSubRelease=pd.get_dummies(X['OsPlatformSubRelease'],prefix='dummy_OsPlatformSubRelease',drop_first=True)
        X=pd.concat([X,dummies_OsPlatformSubRelease],axis=1)

        X.drop("OsPlatformSubRelease",axis=1,inplace=True)

        #Dummificar SmartScreen
        dummies_SmartScreen=pd.get_dummies(X['SmartScreen'],prefix='dummy_SmartScreen',drop_first=True)
        X=pd.concat([X,dummies_SmartScreen],axis=1)

        X.drop("SmartScreen",axis=1,inplace=True)
      
        #Dummificar Census_DeviceFamily
        dummies_Census_DeviceFamily=pd.get_dummies(X['Census_DeviceFamily'],prefix='dummy_Census_DeviceFamily',drop_first=True)
        X=pd.concat([X,dummies_Census_DeviceFamily],axis=1)

        X.drop("Census_DeviceFamily",axis=1,inplace=True)

        #Dummificar Census_OSArchitecture
        dummies_Census_OSArchitecture=pd.get_dummies(X['Census_OSArchitecture'],prefix='dummy_Census_OSArchitecture',drop_first=True)
        X=pd.concat([X,dummies_Census_OSArchitecture],axis=1)

        X.drop("Census_OSArchitecture",axis=1,inplace=True)

        #Dummificar Census_OSWUAutoUpdateOptionsName
        dummies_Census_OSWUAutoUpdateOptionsName=pd.get_dummies(X['Census_OSWUAutoUpdateOptionsName'],prefix='dummy_Census_OSWUAutoUpdateOptionsName',drop_first=True)
        X=pd.concat([X,dummies_Census_OSWUAutoUpdateOptionsName],axis=1)

        X.drop("Census_OSWUAutoUpdateOptionsName",axis=1,inplace=True)

        #Dummificar Census_GenuineStateName
        dummies_Census_GenuineStateName=pd.get_dummies(X['Census_GenuineStateName'],prefix='dummy_Census_GenuineStateName',drop_first=True)
        X=pd.concat([X,dummies_Census_GenuineStateName],axis=1)

        X.drop("Census_GenuineStateName",axis=1,inplace=True)

        return X


   
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
        X.dropna(thresh=10,inplace=True)
        #X.loc[X.SmartScreen.isnull(),"SmartScreen"]="Off"
        X.loc[X.Firewall.isnull(),"IsProtected"]=0
        X.loc[X.Firewall.isnull(),"Firewall"]=0
        #Dropear rows donde UacLuenable no sea 0 o 1
        X=X[X.UacLuaenable<=1]
        #Dropear rows donde HasDetections no sea 0 o 1
        #X=X[X.HasDetections<=1]
        

        X.loc[X.SMode.isnull(),"SMode"]=0
        return X
    
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

startTime=datetime.datetime.now()

df=pd.read_csv("train_clean.csv",encoding="utf-8",nrows=10000)
#df=pd.read_csv("C:\\temp\\train_clean.csv",encoding="utf-8")

print("Archivo de datos leido OK")

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

steps = [("colPreProcessor",colPreProcessor("")),("imputer",Imputer("")),("scaler", myScaler("")), ("SVC",SVC())]

print("Preparando pipeline..")

pipeline = Pipeline(steps) # define the pipeline object.
parametros = {'SVM__C':[0.001,0.1,10,100,10e5], 'SVM__gamma':[0.1,0.01]}
grid = GridSearchCV(pipeline, param_grid=parametros, cv=5)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

print("Pipeline listo para fittear")

pipe_fitstart=datetime.datetime.now()
pipeline.fit(X_train, y_train)
pipe_fitend=datetime.datetime.now()

print("Pipeline fitteado")

pipe_predictstart=datetime.datetime.now()

preds = pipeline.predict(X_train)
pipe_predictend=datetime.datetime.now()
print("Predicciones listas recién salidas del pipeline")

pipe_scoringstart=datetime.datetime.now()

print("score:" ,pipeline.score(X_train,y_train))

pipe_scoringend=datetime.datetime.now()

endTime=datetime.datetime.now()
print("Proceso terminado en:" + str((endTime-startTime).total_seconds()) + " segundos")
print("Pipeline fit:" + str((pipe_fitend-pipe_fitstart).total_seconds()) + " segundos")
print("Pipeline predict:" + str((pipe_predictend-pipe_predictstart).total_seconds()) + " segundos")
print("Pipeline scoring:" + str((pipe_scoringend-pipe_scoringstart).total_seconds()) + " segundos")





