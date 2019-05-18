import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistics import mode 
from matplotlib import cm as cm

import sklearn as sk
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler 
#from sklearn import svm

import datetime


class tp1_DS:
    
    
  def __init__(self):
    print("Version de SciKit:" ,sk.__version__)
    self.df=pd.read_csv("properati_caballito.csv",encoding = 'utf8')
    #print(self.df.head(5))
   
    print("Dataset cargado . Cantidad de registros del dataset:", len(self.df))
    #print("columnas:", self.df.columns)

  def predecir(self):

    dummy_cols = [col for col in self.df if col.startswith('dummy_')]
    #print("dummy columns:" , dummy_cols)
    campos_entrada=dummy_cols + ['surface_total_in_m2','price_usd_per_m2']
    print("campos entrada:" , campos_entrada)
    
    # Escalamos la data (normalización)
    scaler = StandardScaler() 
    
    inputDF=scaler.fit_transform(self.df[campos_entrada])
    #print("precio aprox usd:", self.df['price_aprox_usd'].shape)    
    targetDF=self.df['price_aprox_usd']
    
    
    #TODO:Falta la parte de regularización (se recomendó usar Lasso)
    
    
    Xtrn, Xtest, Ytrn, Ytest = train_test_split(inputDF, 
                                                targetDF,
                                                test_size=0.2)
    
    #print("type of price_aprox_usd:" , type(Ytrn))

    models = [LinearRegression(),
        RandomForestRegressor(n_estimators=100, max_features='sqrt'),
        KNeighborsRegressor(n_neighbors=6)
    ]

    
    model_names=['Linear','Random_Forest','KNeighbor']
    
    XY = {}
    counter=0

    #log startTime
    startTime=datetime.datetime.now()
    for model in models:
      # fit model on training dataset
      model.fit(Xtrn, Ytrn)
      # predict prices for test dataset and calculate r^2
      XY[model_names[counter]] =int(round(r2_score(Ytest, model.predict(Xtest)) * 100,0))
      print("XY:" , XY)
      counter+=1

    endTime=datetime.datetime.now()
    print("Analisis terminado en:", (endTime-startTime).total_seconds())
    y_pos = np.arange(len(XY))
    #print("XY.Values=", list(XY.values()))
    #print("XY.Keys=",list(XY.keys()))
    #print("y_pos=", y_pos)
    plt.bar(y_pos, list(XY.values()), align='center', alpha=0.5)
    plt.xticks(y_pos, XY.keys())
    plt.ylabel("Eficacia %")
    plt.title("% de eficacia de regresiones") 
    plt.show()

x=tp1_DS()
#x.analizarRegistrosRotos()
x.predecir()
