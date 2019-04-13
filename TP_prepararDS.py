# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 11:35:40 2019

@author: Adrian
"""
import pandas as pd
from datetime import datetime
import math


df=pd.read_csv("Properati_CABA.csv",encoding = 'utf8')
#Convertir campos categoricos a valores enteros
#dfStates=pd.DataFrame(df["state_name"].unique(),columns=['name'])
#dfStates["ID"]=list(range(len(dfStates)))
dfPlaces=pd.DataFrame(df["place_name"].unique(),columns=['name'])
dfPlaces["ID"]=list(range(len(dfPlaces)))
dfPropertyTypes=pd.DataFrame(df["property_type"].unique(),columns=['name'])
dfPropertyTypes["ID"]=list(range(len(dfPropertyTypes)))

#print(dfStates)
#print(dfPlaces)
#print(dfPropertyTypes)

startTime=datetime.utcnow()

print("Cant. registros del dataset:", len(df))

auxval=0
df["state_code"]=0
df["place_code"]=0
df["property_type_code"]=0

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
  
endTime=datetime.utcnow()

print("Time elapsed:", (endTime - startTime).total_seconds())

print(df.info())
df.to_csv("Properati_CABA_DS.csv") 