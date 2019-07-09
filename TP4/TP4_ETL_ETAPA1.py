# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 16:03:47 2019

@author: Adrian
"""
import pandas as pd


df=pd.read_csv("C:\\TEMP\\TP4\\train.csv",nrows=25000,encoding="utf-8")
cols=df.select_dtypes(exclude=['int','int8','int16','int32','int64','float','float16', 'float32','float64']).columns

df.drop("MachineIdentifier",axis=1,inplace=True)


counter=0
for col in cols:
    #print("col:", col,",type:",df[col].dtype)
    try:
        print("Reading col:", col, ", type=", df[col].dtype)
        auxdf=pd.get_dummies(df[col],prefix='dummy_' + col,drop_first=True)
        df=pd.concat([df,auxdf],axis=1)
        df.drop(col,axis=1,inplace=True)
        print("Dummifying col:", col, " ready")
        del auxdf
        counter+=1   
        
        #Esto lo puse para ahorrar memoria y que no crashee el kernel
#        if (counter > 10):
#            break;
    except:
        print("col:",col , " not categorical, skipping")
        
print("Ready.", counter, " columns processed")
df.to_csv("train_cut_dummified.csv",encoding="utf-8")

