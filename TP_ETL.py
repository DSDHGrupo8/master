import pandas as pd
import numpy as np
import math
import re
import matplotlib.pyplot as plt
from statistics import mode 
from matplotlib import cm as cm

#from pydrive.auth import GoogleAuth
#from pydrive.drive import GoogleDrive
#from google.colab import auth
#from oauth2client.client import GoogleCredentials
#
#from sklearn import svm

pd.set_option('display.expand_frame_repr', False)
pd.options.display.float_format = '{:.2f}'.format

class tp1_ETL:
  
    def __init__(self):
        #Authenticate and create the PyDrive client.
        #auth.authenticate_user()
        #gauth = GoogleAuth()
        #gauth.credentials = GoogleCredentials.get_application_default()
        #drive = GoogleDrive(gauth)
        #linkDS = 'https://drive.google.com/open?id=1GXhb9LJJshv_gFdiMS6PujIG8SlhaZqv' # The shareable link
        #fluff, id = linkDS.split('=')
        #downloaded = drive.CreateFile({'id':id}) 
        #downloaded.GetContentFile('properatti.csv')
        #        
        ##https://drive.google.com/open?id=1HXMeE6Endm3g4c3JeiYIggdNcoQk6IqR -> DataSet con Lookup table Precio X m2
        #link_lookupPxM2="https://drive.google.com/open?id=1HXMeE6Endm3g4c3JeiYIggdNcoQk6IqR"
        #fluff, id = link_lookupPxM2.split('=')
        #downloaded = drive.CreateFile({'id':id}) 
        #downloaded.GetContentFile('precioxm2_pais.csv')
        
        self.df=pd.read_csv("C:\\Users\\Public\\properati.csv", encoding = 'utf8')
        #self.df=pd.read_hdf("properati_CABA.hdf",key="table")
        self.df_m2=pd.read_csv("precioxm2_pais.csv", encoding = 'utf8')
        print("DataSet registros:", len(self.df))
        print("DataSet Lookup Precio x m2:", len(self.df_m2))
        
        valor_Dolar=17.8305
        
        #DROPEAMOS VARIABLES NO INTERESANTES
        cols=['price', 'currency', 'country_name', 'price_aprox_local_currency','operation','lat','lon','properati_url','place_with_parent_names','image_thumbnail','floor','rooms','geonames_id']
        #cols=['price', 'currency', 'price_aprox_local_currency']
        self.df.drop(cols, axis=1, inplace=True)
        
        #FIJAR SCOPE EN CABA
        self.df = self.df[self.df['state_name'] == 'Capital Federal']
        self.df.drop('state_name', axis=1, inplace=True)
           
        #CORRECCION DE M2 TOTALES
        df1 = self.df[self.df['surface_total_in_m2'].isnull()]
        aux = df1['title'].str.extract(r'( a )?(\.)?(x )?(\d+)\s?(m2|mt|m²)[^c](?!\w?cub)', re.IGNORECASE)
        aux.dropna(how='all', inplace=True)
        aux=aux[(aux[0].isnull()) & (aux[1].isnull()) & (aux[2].isnull())]
        aux=aux.drop([0, 1, 2, 4], axis=1)
        aux.columns=['surface_total_in_m2']
        aux['surface_total_in_m2']=aux['surface_total_in_m2'].astype('float64')
        self.df.loc[self.df['surface_total_in_m2'].isnull(),'surface_total_in_m2'] = aux['surface_total_in_m2']
        
        aux = df1['description'].str.extract(r'( a )?(\.)?(x )?(\d+)\s?(m2|mt|m²)[^c](?!\w?cub)', re.IGNORECASE)
        aux.dropna(how='all', inplace=True)
        aux=aux[(aux[0].isnull()) & (aux[1].isnull()) & (aux[2].isnull())]
        aux=aux.drop([0, 1, 2, 4], axis=1)
        aux.columns=['surface_total_in_m2']
        aux['surface_total_in_m2']=aux['surface_total_in_m2'].astype('float64')
        self.df.loc[self.df['surface_total_in_m2'].isnull(),'surface_total_in_m2'] = aux['surface_total_in_m2']
        
        #CORRECCION DE M2 CUBIERTOS
        df1 = self.df[self.df['surface_covered_in_m2'].isnull()]
        aux = df1['title'].str.extract(r'(\d+)\s?(m2|mt|m²)(c[^o]|\s?cub)', re.IGNORECASE)
        aux.dropna(how='all', inplace=True)
        aux=aux.drop([1, 2], axis=1)
        aux.columns=['surface_covered_in_m2']
        aux['surface_covered_in_m2']=aux['surface_covered_in_m2'].astype('float64')
        self.df.loc[self.df['surface_covered_in_m2'].isnull(),'surface_covered_in_m2'] = aux['surface_covered_in_m2']
        
        aux = df1['description'].str.extract(r'(\d+)\s?(m2|mt|m²)(c[^o]|\s?cub)', re.IGNORECASE)
        aux.dropna(how='all', inplace=True)
        aux=aux.drop([1, 2], axis=1)
        aux.columns=['surface_covered_in_m2']
        aux['surface_covered_in_m2']=aux['surface_covered_in_m2'].astype('float64')
        self.df.loc[self.df['surface_covered_in_m2'].isnull(),'surface_covered_in_m2'] = aux['surface_covered_in_m2']
              
        #CORRECCION DE PRECIOS
        df1 = self.df[self.df['price_aprox_usd'].isnull()]
        aux = df1['title'].str.extract(r'(U?u?\$[SDsd]?)\s?(\d+)\.?(\d*)\.?(\d*)')
        aux.dropna(inplace=True)
        aux[0]=aux[0].replace(to_replace='^\$$', value='ARS', regex=True)
        aux[0]=aux[0].replace(to_replace='^[^A].*$', value='USD', regex=True)
        aux['currency']=aux[0]
        aux['price']=aux[1]+aux[2]+aux[3]
        aux['price']=aux['price'].astype('float64')
        aux=aux[['currency','price']]
        aux.loc[aux['currency'] == 'ARS', 'price'] = aux.loc[:, 'price']/valor_Dolar
        self.df.loc[self.df['price_aprox_usd'].isnull(),'price_aprox_usd'] = aux.loc[:, 'price']
        
        aux = df1['description'].str.extract(r'(U?u?\$[SDsd]?)\s?(\d+)\.?(\d*)\.?(\d*)')
        aux=aux.dropna()
        aux[0]=aux[0].replace(to_replace='^\$$', value='ARS', regex=True)
        aux[0]=aux[0].replace(to_replace='^[^A].*$', value='USD', regex=True)        
        aux['currency']=aux[0]
        aux['price']=aux[1]+aux[2]+aux[3]
        aux['price']=aux['price'].astype('float64')
        aux=aux[['currency','price']]
        aux.loc[aux['currency'] == 'ARS', 'price'] = aux.loc[:, 'price']/valor_Dolar
        self.df.loc[self.df['price_aprox_usd'].isnull(),'price_aprox_usd'] = aux.loc[:, 'price']
        
        #COMPLETAR REGISTROS DESPUES DE LLENAR CON REGEX
        self.df.dropna(subset=['surface_total_in_m2', 'surface_covered_in_m2'], how='all', inplace=True)
        self.df.loc[(self.df['surface_total_in_m2'].isnull()) & (self.df['surface_covered_in_m2'].notnull()), 'surface_total_in_m2'] = self.df.loc[:, 'surface_covered_in_m2']
        self.df.loc[(self.df['price_usd_per_m2'].isnull()) & (self.df['price_aprox_usd'].notnull()) & (self.df['surface_total_in_m2'].notnull()), 'price_usd_per_m2'] = self.df.loc[:, 'price_aprox_usd']/self.df.loc[:, 'surface_total_in_m2']      
        
        
        #FILTRAR OUTLIERS
        qryFiltro="(price_aprox_usd >= 10000 and price_aprox_usd <= 1000000)"
        qryFiltro+=" and (surface_total_in_m2 >= 20 and surface_total_in_m2 <= 1000)"
        qryFiltro+=" and (surface_total_in_m2 >= surface_covered_in_m2)"
        
        self.df=self.df.query(qryFiltro)
        
        self.df.drop(['surface_covered_in_m2', 'price_per_m2'], axis=1, inplace=True)

        
        #ARREGLAR DATOS CORREGIBLES
        #Arreglar precio x m2 en dólares
        self.df['price_aprox_usd']=np.round(self.df['price_aprox_usd'],0).fillna(0).astype(np.int64)
        self.df['surface_total_in_m2']=np.round(self.df['surface_total_in_m2'],0).astype(np.int64)
        self.df['precio_m2_usd']=np.round(self.df['price_aprox_usd'] / self.df['surface_total_in_m2'],0)
        
        
        #ARREGLAR LATITUD Y LONGITUD A PARTIR DE LA COLUMNA LAT-LON
        latlongdf=self.df['lat-lon'].str.split(",",expand=True)
        self.df['lat']=latlongdf.loc[:,0]
        self.df['lon']=latlongdf.loc[:,1]
        self.df.drop('lat-lon',axis=1,inplace=True)
        #self.df.drop('price_per_m2',axis=1,inplace=True)
        
        #Convertir campos categoricos a valores enteros
        dfPlaces=pd.DataFrame(self.df["place_name"].unique(),columns=['name'])
        dfPlaces["ID"]=list(range(len(dfPlaces)))
        dfPropertyTypes=pd.DataFrame(self.df["property_type"].unique(),columns=['name'])
        dfPropertyTypes["ID"]=list(range(len(dfPropertyTypes)))
        
        #print(dfStates)
        #print(dfPlaces)
        #print(dfPropertyTypes)
        
        auxval=0
        self.df["state_code"]=0
        self.df["place_code"]=0
        self.df["property_type_code"]=0
        qryfiltro=""
        
        for index, row in self.df.iterrows():
          if (math.fmod(index,1000)==0):print("Processing row:", index)

          if not (row.property_type is ''):
            
            if (self.df.at[index,"price_usd_per_m2"] == 0):
                qryfiltro="place_name=='" + row.place_name + "'"
                qryfiltro+=" and (m2_Desde<=" + str(row.surface_total_in_m2) 
                qryfiltro+=" and m2_Hasta>=" + str(row.surface_total_in_m2) + ")"
                
                auxval=self.df_m2.query(qryfiltro).Valor_usd
                #print("auxval:" , auxval)
                #print("len(auxval):" , len(auxval))
                
                if (len(auxval)>=2):
                  self.df.at[index,"price_usd_per_m2"]=auxval[1]
                  self.df.at[index,"price_aprox_usd"]=self.df.at[index,"price_usd_per_m2"]*self.df.at[index,"surface_total_in_m2"]
                  
        #dummificar las variables
        dummies_state=pd.get_dummies(df['state_name'],prefix='dummy_state_',drop_first=True)
        dummies_place=pd.get_dummies(df['place_name'],prefix='dummy_place_',drop_first=True)
        dummies_property=pd.get_dummies(df['property_type'],prefix='dummy_property_',drop_first=True)
        
        self.df.join(dummies_state)
        self.df.join(dummies_place)
        self.df.join(dummies_property)
        
        self.df.to_csv("Properati_CABA_DS_fixed.csv",encoding='utf-8')
        #uploaded = drive.CreateFile({'Properati_fixed': 'Properati_fixed.csv'})
        #uploaded.SetContentFile("Properati_fixed.csv")
        #uploaded.Upload()
        
        print("All done!")
        

    def correrAnalisis(self):
        
        
        print("Conteo total de registros sin nulos en price y surface_total:" , str(len(self.df3)))
        
        #print(precioSerie)
        
        
        print("**********************DF2*******************************")
        print(self.df2.describe())
        print("*****************************************************")
        
        
        print("**********************DF3*******************************")
        print(self.df3.describe())
        print("*****************************************************")

x=tp1_ETL()

#x.conteo_por_grupos()

