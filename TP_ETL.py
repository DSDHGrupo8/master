import pandas as pd
import numpy as np
import math
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
  
     
  
    def conteo_por_grupos(self):
        totalregs=len(self.df)
        self.df['COUNTER'] =1       #initially, set that counter to 1.
        #g=self.df.groupby(['state_name','place_name'])['COUNTER'].size();
        g=self.df.groupby(['state_name','place_name'])['COUNTER'].size()/totalregs * 100;
        g=g.sort_values(ascending=False)
        print(g)
  
    def correlation_matrix(self):
        
            
        #plt.rcParams.update({'font.size': 12})
        fig = plt.figure()
        fig.set_figheight(10)
        fig.set_figwidth(40)
        ax1 = fig.add_subplot(111)
        cmap = cm.get_cmap('jet', 20)
        cax = ax1.imshow(self.df.corr(), interpolation="nearest", cmap=cmap)
        ax1.grid(True)
        plt.title('Correlación entre variables')
        labels=self.df.columns
        ax1.set_xticklabels(labels,fontsize=12)
        ax1.set_yticklabels(labels,fontsize=12)
        # Add colorbar, make sure to specify tick locations to match desired ticklabels
        fig.colorbar(cax, ticks=[0,0.25,0.50,0.75,1])
        plt.show()
    
    def mostrarHistogramas(self):
        
        precioSerie=self.df['price_aprox_usd']
        
        print("******************PRECIO**************************************")
        print("Promedio de Precio:" + str(np.round(np.mean(precioSerie),2)))
        print("Mediana del Precio:" + str(np.round(np.median(precioSerie),2)))
        print("Moda:" + str(mode(precioSerie)))
        
        plt.hist(precioSerie,bins=50)
        plt.xlabel("Precio en USD - propiedades")
        plt.ylabel("Frecuencia")
        plt.show()
        
        supSerie=self.df['surface_total_in_m2']
        
        print("******************SUPERFICIE**************************************")
        print("Promedio de superficie:" + str(np.round(np.mean(supSerie),2)))
        print("Mediana de superficie:" + str(np.round(np.median(supSerie),2)))
        print("Moda:" + str(mode(supSerie)))
        
        
        print("Cantidad de registros con price mayor > 0 sin nulos:" ,str(len(precioSerie)))
        
        plt.hist(supSerie,bins=50)
        plt.xlabel("Superficie m2 - propiedades")
        plt.ylabel("Frecuencia")
        plt.show()

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
        
        self.df=pd.read_csv("Properati_CABA_DS.csv", encoding = 'utf8')
        self.df_m2=pd.read_csv("precioxm2_pais.csv", encoding = 'utf8')
        print("DataSet registros:", len(self.df))
        print("DataSet Lookup Precio x m2:", len(self.df_m2))
        
        #FIJAR SCOPE EN CABA
        qryFiltro="(state_name=='Capital Federal') and (price_aprox_usd >= 10000 and price_aprox_usd <= 1000000)"
        qryFiltro+=" and (surface_total_in_m2 >= 20 and surface_total_in_m2 <= 1000)"
        qryFiltro+=" and (surface_total_in_m2 >= surface_covered_in_m2)"
        
        self.df=self.df.query(qryFiltro)
        
        #DROPEAMOS VARIABLES NO INTERESANTES
        #dropeo columnas que no son de interés
        cols=['country_name', 'currency','price', 'price_aprox_local_currency','operation','lat','lon','properati_url','description','title','place_with_parent_names','image_thumbnail','floor','rooms','geonames_id','price_usd_per_m2']
        self.df.drop(cols, axis=1, inplace=True)
        self.df.dropna(subset=['surface_total_in_m2','price_aprox_usd'],inplace=True)
        
            
        #CORRECCION DE PRECIOS y MONEDA
        df1 = self.df[self.df['price'].isnull()]
        aux = df1['title'].str.extract(r'(U?u?\$[SDsd]?)\s?(\d+)\.?(\d*)\.?(\d*)')
        #print("aux:", len(aux))
        aux.dropna(inplace=True)
        print("aux cantidad no nulos:", len(aux.dropna()))
        #aux[0]=aux[0].str.replace('^\$$', 'ARS', regex=True)
        aux[0]=aux[0].replace(to_replace='^\$$', value='ARS', regex=True)
        #aux[0]=aux[0].str.replace('^[^A].*$', 'USD', regex=True)
        aux[0]=aux[0].replace(to_replace='^[^A].*$', value='USD', regex=True)
        aux['currency']=aux[0]
        aux['price']=aux[1]+aux[2]+aux[3]
        aux['price']=aux['price'].astype('float64')
        aux=aux[['currency','price']]
        self.df.loc[self.df['price'].isnull(),'price'] = aux['price']
        self.df.loc[self.df['currency'].isnull(),'currency'] = aux['currency']
        df1 = self.df[self.df['price'].isnull()]
        aux = df1['description'].str.extract(r'(U?u?\$[SDsd]?)\s?(\d+)\.?(\d*)\.?(\d*)')
        aux=aux.dropna()
        #aux[0]=aux[0].str.replace('^\$$', 'ARS', regex=True)
        aux[0]=aux[0].replace(to_replace='^\$$', value='ARS', regex=True)
        #aux[0]=aux[0].str.replace('^[^A].*$', 'USD', regex=True)
        aux[0]=aux[0].replace(to_replace='^[^A].*$', value='USD', regex=True)
        aux['currency']=aux[0]
        aux['price']=aux[1]+aux[2]+aux[3]
        aux['price']=aux['price'].astype('float64')
        aux=aux[['currency','price']]
        self.df.loc[self.df['price'].isnull(),'price'] = aux['price']
        self.df.loc[self.df['currency'].isnull(),'currency'] = aux['currency']
        
        #ARREGLAR DATOS CORREGIBLES
        #Arreglar precio x m2 en dólares
        self.df['price_aprox_usd']=np.round(self.df['price_aprox_usd'],0).fillna(0).astype(np.int64)
        self.df['surface_total_in_m2']=np.round(self.df['surface_total_in_m2'],0).fillna(0).astype(np.int64)
        self.df['precio_m2_usd']=np.round(self.df['price_aprox_usd'] / self.df['surface_total_in_m2'],0)
        
        #auxdf=self.df.query('precio_m2_usd>10000').filter(items=['precio_m2_usd','price_aprox_usd','surface_total_in_m2'])
        #print(auxdf)
        #print(self.df['precio_m2_usd'])
        
        #ARREGLAR LATITUD Y LONGITUD A PARTIR DE LA COLUMNA LAT-LON
        latlongdf=self.df['lat-lon'].str.split(",",expand=True)
        self.df['lat']=latlongdf.loc[:,0]
        self.df['lon']=latlongdf.loc[:,1]
        self.df.drop('lat-lon',axis=1,inplace=True)
        self.df.drop('price_per_m2',axis=1,inplace=True)
        
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
          self.df.at[index,"state_code"]=0
          
          if not (row.place_name is ''):
            auxval=dfPlaces.query("name=='" + row.place_name + "'").ID
            self.df.at[index,"place_code"]=auxval
              
          if not (row.property_type is ''):
            auxval=dfPropertyTypes.query("name=='" + row.property_type + "'").ID
            #print("property_type", row.property_type)
            #print("auxval found:", auxval)
            self.df.at[index,"property_type_code"]=auxval
            #print("row.property_type:",row.property_type)
            
            qryfiltro="place_name=='" + row.place_name + "'"
            qryfiltro+=" and (m2_desde<=" + str(row.surface_total_m2) 
            qryfiltro+=" and m2_hasta>=" + str(row.surface_total_m2) + ")"
            
            self.df.at[index,"price_usd_per_m2"]=self.df_m2.query(qryfiltro).Valor_usd
        
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
#x.correlation_matrix()
x.conteo_por_grupos()
