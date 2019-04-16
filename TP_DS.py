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


class tp1_DS:
    
    
  def __init__(self):
    print("Version de SciKit:" ,sk.__version__)
    self.df=pd.read_csv("Properati_CABA_DS_fixed.csv",encoding = 'utf8')
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

  def predecir(self):

    
    campos_entrada=['state_code','place_code','property_type_code','surface_total_in_m2','precio_m2_usd']
    Xtrn, Xtest, Ytrn, Ytest = train_test_split(self.df[campos_entrada], self.df['price_aprox_usd'],
                      test_size=0.2)
    
    print("type of price_aprox_usd:" , type(Ytrn))

    models = [LinearRegression(),
        RandomForestRegressor(n_estimators=100, max_features='sqrt'),
        KNeighborsRegressor(n_neighbors=6),
        SVR(kernel='linear'),
        LogisticRegression()
    ]

    TestModels = pd.DataFrame()
    tmp = {}

    model_names=['Linear','Random_Forest','KNeighbor','SVR','Logistic']
 
    counter=0
    for model in models:
      # get model name
      m = str(model)
      
      tmp['Model'] = model_names[counter]
      # fit model on training dataset
      model.fit(Xtrn, Ytrn)
      # predict prices for test dataset and calculate r^2
      tmp['R2_Price'] = r2_score(Ytest, model.predict(Xtest))
      # write obtained data
      TestModels = TestModels.append([tmp])
      #print("len(TestModels):", len(TestModels))

      TestModels.set_index('Model', inplace=True)
      print("TestModels:",TestModels)
    
      fig, axes = plt.subplots(ncols=1, figsize=(5, 5))
      #print("axes:",axes)
      TestModels.R2_Price.plot(ax=axes, kind='bar', title='R2_Price')
      
      counter+=1
      plt.show()  
    
     

x=tp1_DS()
#x.analizarRegistrosRotos()
x.predecir()
