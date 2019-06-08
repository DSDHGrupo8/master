# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 11:35:40 2019

@author: Adrian
"""
import pandas as pd
import numpy as np
from datetime import datetime
import math


df=pd.read_csv("Properati.csv",encoding = "utf8")
df_m2=pd.read_csv("precioxm2_pais.csv", encoding = 'utf8')

#DROPEAMOS VARIABLES NO INTERESANTES
#dropeo columnas que no son de interés
cols=['country_name', 'price_aprox_local_currency','operation','lat','lon','properati_url','place_with_parent_names','image_thumbnail','floor','rooms','price_usd_per_m2']
df.drop(cols, axis=1, inplace=True)
#df.dropna(subset=['surface_total_in_m2','price_aprox_usd'],inplace=True)

#FIJAR SCOPE EN CABA
qryFiltro="state_name=='Capital Federal' and (price_aprox_usd >= 10000 and price_aprox_usd <= 1000000) "
qryFiltro+="and (surface_total_in_m2 >= 20 and surface_total_in_m2 <= 1000) "
qryFiltro+="and (surface_total_in_m2 >= surface_covered_in_m2)"

df=df.query(qryFiltro)

startTime=datetime.utcnow()

print("Cant. registros del dataset:", len(df))

#print(df.info())

#ARREGLAR DATOS CORREGIBLES
#Arreglar precio x m2 en dólares
df['price_aprox_usd']=np.round(df['price_aprox_usd'],0).fillna(0).astype(np.int64)
df['surface_total_in_m2']=np.round(df['surface_total_in_m2'],0).fillna(0).astype(np.int64)
df['precio_m2_usd']=np.round(df['price_aprox_usd'] / df['surface_total_in_m2'],0)


#ARREGLAR LATITUD Y LONGITUD A PARTIR DE LA COLUMNA LAT-LON
latlongdf=df['lat-lon'].str.split(",",expand=True)
#print("latlongDF:", latlongdf.head(10))
df['lat']=latlongdf.loc[:,0]
df['lon']=latlongdf.loc[:,1]
#print("Columna lat:",df['lat'].head(10))
#print("Columna lon:", df['lon'].head(10))
df.drop('lat-lon',axis=1,inplace=True)
df.drop('price_per_m2',axis=1,inplace=True)



#Convertir campos categoricos a valores enteros
dfPlaces=pd.DataFrame(df["place_name"].unique(),columns=['name'])
dfPlaces["ID"]=list(range(len(dfPlaces)))
dfPropertyTypes=pd.DataFrame(df["property_type"].unique(),columns=['name'])
dfPropertyTypes["ID"]=list(range(len(dfPropertyTypes)))

#print(dfStates)
#print(dfPlaces)
#print(dfPropertyTypes)

auxval=0
df["state_code"]=0
df["place_code"]=0
df["property_type_code"]=0
qryfiltro=""

for index, row in df.iterrows():
  if (math.fmod(index,1000)==0):print("Processing row:", index)
  df.at[index,"state_code"]=0
  
  if not (row.place_name is ''):
    auxval=dfPlaces.query("name=='" + row.place_name + "'").ID
    df.at[index,"place_code"]=auxval
      
  if not (row.property_type is ''):
    auxval=dfPropertyTypes.query("name=='" + row.property_type + "'").ID
    #print("property_type", row.property_type)
    #print("auxval found:", auxval)
    df.at[index,"property_type_code"]=auxval
    #print("row.property_type:",row.property_type)
    
    qryfiltro="place_name=='" + row.place_name + "'"
    qryfiltro+=" and (m2_desde<=" + str(row.surface_total_m2) 
    qryfiltro+=" and m2_hasta>=" + str(row.surface_total_m2) + ")"
    
    df.at[index,"price_usd_per_m2"]=df_m2.query(qryfiltro).Valor_usd
  

#CORRECCION DE PRECIOS y MONEDA
df1 = df[df['price'].isnull()]
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
df.loc[df['price'].isnull(),'price'] = aux['price']
df.loc[df['currency'].isnull(),'currency'] = aux['currency']
df1 = df[df['price'].isnull()]
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
df.loc[df['price'].isnull(),'price'] = aux['price']
df.loc[df['currency'].isnull(),'currency'] = aux['currency']

df.to_csv("Properati_CABA_prueba.csv",encoding="utf-8") 

endTime=datetime.utcnow()

print("Time elapsed:", (endTime - startTime).total_seconds())