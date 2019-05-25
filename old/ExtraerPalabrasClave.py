# -*- coding: utf-8 -*-
"""
Created on Thu May 16 19:18:31 2019

@author: Adrian
"""
import nltk
import pandas as pd
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords


df=pd.read_csv("..\\datasets\\properati_caballito.csv", encoding = 'utf8')
#print(str(df["description"]))
L=list(df["description"])
str1=''.join(map(str, L))
##print("str1:" , str1)
#

word_tokens = nltk.word_tokenize(str1)
stop_words=set(stopwords.words('spanish'))
lineas_filtradas = [w for w in word_tokens if not w in stop_words]
str_lineas_filtradas = " ".join(lineas_filtradas)
#print("str_lineas_filtradas:",str_lineas_filtradas)
tokens_filtrados=nltk.word_tokenize(str_lineas_filtradas)
#print("tokens filtrados:", tokens_filtrados)
df = pd.DataFrame.from_dict(Counter(tokens_filtrados), orient='index').reset_index()
df = df.rename(columns={'index':'palabra', 0:'contador'})
df=df[(df.palabra.str.len()>4) & (df.contador>=10)]
df_tabla_freq=df.sort_values(by=['contador'],ascending=False)
df_tabla_freq.to_csv("tabla_freq_palabras.csv",encoding="utf-8")