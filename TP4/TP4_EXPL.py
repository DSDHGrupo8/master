# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 22:18:50 2019

@author: Adrian
"""

import pandas as pd

for x in range(1,18):
    print("***************ANALIZING CHUNK:" + str(x) + "****************")
    df=pd.read_csv("C:\\Temp\\test\\train_clean_" + str(x) + ".csv")
    size=len(df)
    print("len(df):" , size)
    print(df.isna().sum().sort_values(ascending=False))
    print("************************************************************")
    
    

