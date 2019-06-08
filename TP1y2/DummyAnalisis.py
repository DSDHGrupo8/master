# -*- coding: utf-8 -*-
"""
Created on Wed May 29 17:54:20 2019

@author: Adrian
"""

import pandas as pd


df_train=pd.read_csv("datasets\\properati_caballito_train.csv",encoding="utf8")
df_test=pd.read_csv("datasets\\properati_caballito_test.csv",encoding="utf8")

dummy_cols = [col for col in df_train if col.startswith('dummy_')]

print(df_train[dummy_cols].mean().sort_values(ascending=False))