# -*- coding: utf-8 -*-
"""
Created on Tue May 14 19:39:06 2019

@author: Adrian
"""
import pandas as pd
from textblob import TextBlob


df=pd.read_csv("C:\\Users\\Public\\properati.csv", encoding = 'utf8')

#EXTRAER SUSTANTIVOS DEL LINK
blob =TextBlob(df['properati_url'])
print(blob.noun_phrases)
