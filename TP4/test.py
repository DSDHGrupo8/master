# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 23:54:27 2019

@author: Adrian
"""
import pandas as pd

df=pd.read_csv("train_clean.csv",encoding="utf-8")


#print(df.var().sort_values(ascending=False))
print(df.isnull().sum().sort_values(ascending=False))

