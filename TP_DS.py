import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistics import mode 
from matplotlib import cm as cm

#from pydrive.auth import GoogleAuth
#from pydrive.drive import GoogleDrive
#from google.colab import auth
#from oauth2client.client import GoogleCredentials

import sklearn as sk
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
#from sklearn import svm
from datetime import datetime
import math

class tp1_DS:
    
    
  def __init__(self):
    print("Version de SciKit:" ,sk.__version__)
    self.df=pd.read_csv("Properati_CABA_DS.csv",encoding = 'utf8')
    # Authenticate and create the PyDrive client.
#    auth.authenticate_user()
#    gauth = GoogleAuth()
#    gauth.credentials = GoogleCredentials.get_application_default()
#    drive = GoogleDrive(gauth)
#    link = 'https://drive.google.com/open?id=10yw7Aopax6xH572LiDe_pa-oHaBNOuag' # The shareable link
#    #link = 'https://drive.google.com/open?id=1GXhb9LJJshv_gFdiMS6PujIG8SlhaZqv' # The shareable link
#    fluff, id = link.split('=')
#    downloaded = drive.CreateFile({'id':id}) 
#    downloaded.GetContentFile('Properati_limpio.csv')  
    
    
    #print(dfStates.head(5))
    #print(dfPlaces.head(5))
    #print(dfPropertyTypes.head(5))
    
    print("Dataset cargado . Cantidad de registros del dataset:", len(self.df))
  def analizarRegistrosRotos(self):
    
    self.df=self.df.fillna(0)
    print("Cantidad registros:",len(self.df))
    dfNulos0=self.df[(self.df.surface_total_in_m2==0) & (self.df.price_aprox_usd==0) & (self.df.price_usd_per_m2==0)] 
    dfNulos1=self.df[(self.df.surface_total_in_m2==0) & (self.df.price_aprox_usd==0)] 
    dfNulos2=self.df[(self.df.price_usd_per_m2==0) & (self.df.surface_total_in_m2==0)] 
    dfNulos3=self.df[(self.df.price_usd_per_m2==0) & (self.df.price_aprox_usd==0)]
    
    dfNulos4=self.df[(self.df.price_usd_per_m2==0)]
    dfNulos5=self.df[(self.df.price_aprox_usd==0)]
    dfNulos6=self.df[(self.df.surface_total_in_m2==0)]
    
    print("Registros rotos tipo 0 (irrecuperable):" ,np.round(len(dfNulos0)/len(self.df)*100,2) , "%")
    
    print("Registros rotos tipo 1 (sup total. m2./ precio aprox. usd.):" ,np.round(len(dfNulos1)/len(self.df)*100,2), "%")
    print("Registros rotos tipo 2 (sup. total m2./ precio por m2):" ,np.round(len(dfNulos2)/len(self.df)*100,2), "%")
    print("Registros rotos tipo 3 (precio por m2 / precio en usd):" ,np.round(len(dfNulos3)/len(self.df)*100,2), "%")

    print("Registros rotos tipo 4 (precio por m2):" ,np.round(len(dfNulos4)/len(self.df)*100,2), "%")
    print("Registros rotos tipo 5 (precio aprox usd):" ,np.round(len(dfNulos5)/len(self.df)*100,2), "%")
    print("Registros rotos tipo 6 (superficie total):" ,np.round(len(dfNulos6)/len(self.df)*100,2), "%")
    
  def predecir(self):
      #prepare dataset  
      #....   
      #spilt dataset
      
      campos_entrada=['state_code','place_code','property_type_code','surface_total_in_m2','precio_m2_usd']
      
      
      Xtrn, Xtest, Ytrn, Ytest = train_test_split(self.df[campos_entrada], self.df['price_aprox_usd'],
                                                  test_size=0.2)
      
      models = [LinearRegression(),
                RandomForestRegressor(n_estimators=100, max_features='sqrt'),
                KNeighborsRegressor(n_neighbors=6),
                SVR(kernel='linear'),
                LogisticRegression()
                ]

      TestModels = pd.DataFrame()
      tmp = {}
 
      for model in models:
          # get model name
          m = str(model)
          tmp['Model'] = m[:m.index('(')]
          # fit model on training dataset
          model.fit(Xtrn, Ytrn['price_aprox_usd'])
          # predict prices for test dataset and calculate r^2
          tmp['R2_Price'] = r2_score(Ytest['price_aprox_usd'], model.predict(Xtest))
          # write obtained data
          TestModels = TestModels.append([tmp])

      TestModels.set_index('Model', inplace=True)

      fig, axes = plt.subplots(ncols=1, figsize=(10, 4))
      TestModels.R2_Price.plot(ax=axes, kind='bar', title='R2_Price')
      plt.show()

x=tp1_DS()
#x.analizarRegistrosRotos()
x.predecir()
