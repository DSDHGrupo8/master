# -*- coding: utf-8 -*-
"""
Created on Tue May 14 19:51:18 2019

@author: Adrian
"""

import pandas as pd
import nltk
from nltk.corpus import stopwords


df=pd.read_csv("C:\\Users\\Public\\properati.csv",encoding = 'utf8')
print("columnas:", df.columns)

#FIJAR SCOPE EN CABA - Caballito
df = df[df['state_name'] == 'Capital Federal']
df = df[df['place_name'] == 'Caballito']
df.drop('state_name', axis=1, inplace=True)
print("cantidad de registros:", len(df))

df["pileta"]=0
df["balcon"]=0
df["patio"]=0
df["lavadero"]=0
df["cochera"]=0
df["luminoso"]=0
df["terraza"]=0
df["quincho"]=0
df["baulera"]=0
df["parrilla"]=0
df["premium"]=0
df["piscina"]=0
df["ascensor"]=0
df["apto_credito"]=0
df["profesional"]=0
df["alarma"]=0
df["amenities"]=0
df["calefaccion"]=0
df["aire_acondicionado"]=0
df["pozo"]=0
df["gimnasio"]=0
df["spa"]=0
df["jacuzzi"]=0
df["cine"]=0

#print(str(df["description"]))
L=list(df["description"])
str1=''.join(map(str, L))
#print("str1:" , str1)

word_tokens = nltk.word_tokenize(str1)
stop_words=set(stopwords.words('spanish'))
lineas_filtradas = [w for w in word_tokens if not w in stop_words]
str_lineas_filtradas = " ".join(lineas_filtradas)
#print("str_lineas_filtradas:",str_lineas_filtradas)
tokens_filtrados=nltk.word_tokenize(str_lineas_filtradas)
print("tokens filtrados:", tokens_filtrados)




