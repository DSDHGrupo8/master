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
dfStates=pd.DataFrame(df["state_name"].unique(),columns=['name'])
dfStates["ID"]=list(range(len(dfStates)))
dfPlaces=pd.DataFrame(df["place_name"].unique(),columns=['name'])
dfPlaces["ID"]=list(range(len(dfPlaces)))
dfPropertyTypes=pd.DataFrame(df["property_type"].unique(),columns=['name'])
dfPropertyTypes["ID"]=list(range(len(dfPropertyTypes)))

#print(dfStates.head(5))
#print(dfPlaces.head(5))
#print(dfPropertyTypes.head(5))

startTime=datetime.utcnow()

print("Cant. registros del dataset:", len(df))

auxval=0

for index, row in df.iterrows():
  if (math.fmod(index,1000)==0):print("Processing row:", index)
  auxval=dfStates[dfStates.name==row.state_name].ID
  df.set_value(index,"state_code",auxval)
  auxval=dfPlaces[dfPlaces.name==row.place_name].ID
  df.set_value(index,"place_code",auxval)
  auxval=dfPropertyTypes[dfPropertyTypes.name==row.property_type].ID
  df.set_value(index,"property_type_code",auxval)
  #print("row.property_type_code:",row.property_type_code)

endTime=datetime.utcnow()

print("Time elapsed:", (endTime - startTime).total_seconds())

print(df.info())
df.to_csv("Properati_CABA_DS.csv")