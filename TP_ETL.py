import pandas as pd
import numpy as np
import math
import re
import matplotlib.pyplot as plt
from statistics import mode
from matplotlib import cm as cm
from shapely import wkt
from shapely.geometry import Point, Polygon, MultiPolygon, LinearRing
import geopandas as gpd

#from pydrive.auth import GoogleAuth
#from pydrive.drive import GoogleDrive
#from google.colab import auth
#from oauth2client.client import GoogleCredentials
#
#from sklearn import svm

pd.set_option('display.expand_frame_repr', False)
pd.options.display.float_format = '{:.2f}'.format

class tp1_ETL:
   
    def calcularDistancia(self,lat1, lon1, lat2, lon2):
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a =math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return self.R * c
       
    def distanciaMinimaParque(self, lat, lon):
        listaDistancias = []
        if ((lat is not None) and (lon is not None)):
            flat=float(lat)
            flon=float(lon)    
        for index, row in self.parques.iterrows():
            d = self.parques.at[index,"Arcos"].project(Point(flon,flat))
            p = self.parques.at[index,"Arcos"].interpolate(d)
            pmc = list(p.coords)[0]
            listaDistancias.append(self.calcularDistancia(math.radians(flat), math.radians(flon), math.radians(float(pmc[1])), math.radians(float(pmc[0]))))
       
        if (len(listaDistancias)>0):
            return min(listaDistancias)
        else:
            return math.nan

    def __init__(self):
        self.R = 6373.0
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
        self.df_m2=pd.read_csv("datasets\\precioxm2_pais.csv", encoding = 'utf8')
        self.subte=pd.read_csv("datasets\\estaciones-de-subte.csv", encoding = 'utf8')
        self.subtes=self.subte[['lon', 'lat']]
        self.hospitales=pd.read_csv("datasets\\hospitales.csv", encoding = 'utf8')
        self.escuelas=pd.read_csv("datasets\\escuelas.csv", encoding = 'utf8')
        self.parques=pd.read_csv("datasets\\arcos.csv", encoding = 'utf8')
        self.barrios=pd.read_csv('datasets\\barrios.csv', encoding = 'utf8')
        self.barrios['WKT'] = self.barrios['WKT'].apply(wkt.loads)
        self.barrios=gpd.GeoDataFrame(geometry=self.barrios.WKT)
        self.parques['Arcos'] = self.parques['Arcos'].apply(wkt.loads)
       
        self.subtes_gdf = gpd.GeoDataFrame(self.subtes, geometry=[Point(x, y) for x, y in zip(self.subtes.lon, self.subtes.lat)], crs={'init': 'epsg:4326'})
        self.subtes_gdf.to_crs(epsg=22196,inplace=True)
        self.hospitales_gdf = gpd.GeoDataFrame(self.hospitales, geometry=[Point(x, y) for x, y in zip(self.hospitales.lon, self.hospitales.lat)], crs={'init': 'epsg:4326'})
        self.hospitales_gdf.to_crs(epsg=22196,inplace=True)
        self.escuelas_gdf = gpd.GeoDataFrame(self.escuelas, geometry=[Point(x, y) for x, y in zip(self.escuelas.lon, self.escuelas.lat)], crs={'init': 'epsg:4326'})
        self.escuelas_gdf.to_crs(epsg=22196,inplace=True)
       
        print("DataSet registros:", len(self.df))
        print("DataSet Lookup Precio x m2:", len(self.df_m2))
        
       

        valor_Dolar=17.8305
       
        #DROPEAMOS VARIABLES NO INTERESANTES
        cols=['price', 'currency', 'country_name', 'price_aprox_local_currency','operation','properati_url','place_with_parent_names','image_thumbnail','rooms','geonames_id']
        #cols=['price', 'currency', 'price_aprox_local_currency']
        self.df.drop(cols, axis=1, inplace=True)

        #FIJAR SCOPE EN CABA - Caballito
        self.df = self.df[self.df['state_name'] == 'Capital Federal']

        self.df['lon'].fillna(0, inplace=True)
        self.df['lat'].fillna(0, inplace=True)
        self.df = gpd.GeoDataFrame(self.df, geometry=[Point(x, y) for x, y in zip(self.df.lon, self.df.lat)])
        self.df = gpd.sjoin(self.df, self.barrios, how='inner')
        self.df.loc[:, 'place_name'] = 'Caballito'
        
        self.df.drop('state_name', axis=1, inplace=True)
        print("cantidad de registros:", len(self.df))
       
        #dummificar las variables place_name y property_type
        #dummies_place=pd.get_dummies(self.df['place_name'],prefix='dummy_place_',drop_first=True)
        dummies_property=pd.get_dummies(self.df['property_type'],prefix='dummy_property_type_',drop_first=True)
        self.df=pd.concat([self.df,dummies_property],axis=1)
        #self.df=pd.concat([self.df,dummies_place],axis=1)
		

        #IMPUTAMOS EXPENSAS POR EL PROMEDIO
        promedio_exp=round(self.df['expenses'].mean(),2)
        print("promedio expensas:", promedio_exp)
        self.df['expenses']=self.df['expenses'].fillna(promedio_exp)
        
        #ARREGLAR LATITUD Y LONGITUD A PARTIR DE LA COLUMNA LAT-LON
        latlongdf=self.df['lat-lon'].str.split(",",expand=True)
        self.df['lat']=latlongdf.loc[:,0]
        self.df['lon']=latlongdf.loc[:,1]
        self.df.drop('lat-lon',axis=1,inplace=True)

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
        
        #ARREGLAR DATOS CORREGIBLES
        #Arreglar precio x m2 en dólares
        self.df['price_aprox_usd']=np.round(self.df['price_aprox_usd'],0).fillna(0).astype(np.int64)
        self.df['surface_total_in_m2']=np.round(self.df['surface_total_in_m2'],0).astype(np.int64)

        
     
        auxval=0
        qryfiltro=""
        rowcounter=0	
		
        self.df.reset_index(drop=True, inplace=True)
       
        for index, row in self.df.iterrows():
            rowcounter+=1
            if (math.fmod(rowcounter,100)==0):print("Processing row:", rowcounter)
            
            aux = pd.Series(Point(float(self.df.at[index,"lon"]), float(self.df.at[index,"lat"])))
            aux = gpd.GeoDataFrame(aux, geometry=aux, crs={'init':'epsg:4326'})
            aux.to_crs(epsg=22196,inplace=True)
            aux = aux.loc[0,'geometry']
    
            self.df.at[index,"distSubte"] = min(self.subtes_gdf.distance(aux))
            self.df.at[index,"distEscuela"] = min(self.escuelas_gdf.distance(aux))
            self.df.at[index,"distHospital"] = min(self.hospitales_gdf.distance(aux))
            self.df.at[index,"distParque"] = self.distanciaMinimaParque(self.df.at[index,"lat"], self.df.at[index,"lon"])

            

        vcols=["acondicionado","amenities","alarma","ascensor","balcon","baulera","blindada","calefaccion",
               "cancha","cine","cochera","contrafrente","crédito","electrógeno","estrenar","fitness","frente","frio-calor",
               "guardacoche","gimnasio","jacuzzi","hidromasaje","hospital",
               "jardin","lavarropas","lavadero","laundry","luminoso","living","metrobus","multisplit","parque",
               "patio","parrilla","pentahome","pileta","premium","piscina","policlínico","profesional",
               "quincho","refrigeración","residencial","reciclado","pozo","sauna",
               "spa","split","solarium","sum","S.U.M","subte","suite","seguridad","terraza","vigilancia"]

        for x in vcols:
            self.df["dummy_" + x]=self.df["description"].str.contains(x).astype(int)
            
    
        #HACEMOS EL RECORTE
        cant_regs_total=len(self.df)
        cant_regs_train=math.trunc((cant_regs_total/100)*80)
        print("cant. regs. totales:", cant_regs_total)
        print("cant. regs. train:", cant_regs_train)
        df_test=self.df.iloc[cant_regs_train:cant_regs_total,:]
        df_test['precio_m2_usd']=df_test['price_usd_per_m2']
        print("df_test:" ,len(df_test))
     
        self.df.loc[(self.df['price_usd_per_m2'].isnull()) & (self.df['price_aprox_usd'].notnull()) & (self.df['surface_total_in_m2'].notnull()), 'price_usd_per_m2'] = self.df.loc[:, 'price_aprox_usd']/self.df.loc[:, 'surface_total_in_m2']
        self.df.drop(['surface_covered_in_m2', 'price_per_m2'], axis=1, inplace=True)
        self.df['precio_m2_usd']=np.round(self.df['price_aprox_usd'] / self.df['surface_total_in_m2'],0)


        #LIMPIAR BASURA
        self.df=self.df[pd.to_numeric(self.df['dummy_property_type__store'], errors='coerce').notnull()]
        self.df=self.df[pd.to_numeric(self.df['dummy_property_type__apartment'], errors='coerce').notnull()]
        self.df=self.df[pd.to_numeric(self.df['dummy_property_type__house'], errors='coerce').notnull()]
        self.df=self.df[pd.to_numeric(self.df['lat'], errors='coerce').notnull()]
        self.df=self.df[pd.to_numeric(self.df['lon'], errors='coerce').notnull()]
        self.df=self.df[pd.to_numeric(self.df['distSubte'], errors='coerce').notnull()]

        #Guardamos el dataset antes del recorte
        self.df.to_csv("datasets\\properati_caballito.csv",encoding='utf-8')		
       
       
        #df_test["price_usd_per_m2"]=0
        
        self.df=self.df.iloc[:cant_regs_train,:]
        
        for index, row in self.df.iterrows():
            if not (row.property_type == ''):
            
                if (self.df.at[index,"price_usd_per_m2"] == 0):
                    qryfiltro="place_name=='" + row.place_name + "'"
                    qryfiltro+=" and (m2_Desde<=" + str(row.surface_total_in_m2)
                    qryfiltro+=" and m2_Hasta>=" + str(row.surface_total_in_m2) + ")"

                    auxval=self.df_m2.query(qryfiltro).Valor_usd
                    #print("auxval:" , auxval)
                    #print("len(auxval):" , len(auxval))
            
                    if (len(auxval)>=2):
                      self.df.at[index,"price_usd_per_m2"]=auxval[1]
        

        #FILTRAR OUTLIERS
        qryFiltro="(price_aprox_usd >= 50000 and price_aprox_usd <= 500000)"
        qryFiltro+=" and (surface_total_in_m2 >= 35 and surface_total_in_m2 <= 500)"
        #qryFiltro+=" and (surface_total_in_m2 >= surface_covered_in_m2)"
        qryFiltro+=" and (precio_m2_usd <= 7000 and precio_m2_usd >= 2000)"
        qryFiltro+=" and (price_usd_per_m2 <= 7000 and price_usd_per_m2 >= 2000)"
        
        self.df=self.df.query(qryFiltro)
        df_test=df_test.query(qryFiltro)
        
        
        self.df["rooms"]=round(self.df['surface_total_in_m2'] / 10,0)
        df_test["rooms"]=round(df_test['surface_total_in_m2'] / 10,0)
        

        df_test.to_csv("datasets\\properati_caballito_test.csv",encoding="utf8")
        self.df.to_csv("datasets\\properati_caballito_train.csv",encoding='utf-8')
        print("campos de salida:", self.df.columns)
        #uploaded = drive.CreateFile({'Properati_fixed': 'Properati_fixed.csv'})
        #uploaded.SetContentFile("Properati_fixed.csv")
        #uploaded.Upload()

        print("All done!")


x=tp1_ETL()
