# -*- coding: utf-8 -*-
"""
Created on Thu May 16 19:18:31 2019

@author: Adrian
"""

df=pd.read_csv("properati_caballito.csv", encoding = 'utf8')
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
print("tokens filtrados:", tokens_filtrados)