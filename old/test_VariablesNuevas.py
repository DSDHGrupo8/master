# -*- coding: utf-8 -*-
"""
Created on Tue May 14 19:51:18 2019

@author: Adrian
"""

import pandas as pd


df=pd.read_csv("C:\\Users\\Public\\properati.csv",encoding = 'utf8')
print("columnas:", df.columns)

#FIJAR SCOPE EN CABA - Caballito
df = df[df['state_name'] == 'Capital Federal']
df = df[df['place_name'] == 'Caballito']
df.drop('state_name', axis=1, inplace=True)
print("cantidad de registros:", len(df))


vcols=["pileta","balcon","patio","lavadero","cochera","luminoso","terraza","quincho",
 "baulera","parrilla","premium","piscina","ascensor","profesional","alarma",
 "amenities","calefaccion","pozo","gimnasio","aire acondicionado","spa","jacuzzi","cine"]

for x in vcols:
    df["dummy_" + x]=df["description"].str.contains(x).astype(int)
    
print(df.head(5))

df.to_csv("properati_caballito.csv",encoding="utf8")





