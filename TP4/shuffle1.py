# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 17:41:40 2019

@author: Adrian
"""

#import numpy as np
import pandas as pd
df=pd.read_csv("C:\\TEMP\\TP4\\train.csv",nrows=1000000,encoding="utf-8")

df2=df.sample(frac=0.02)

df2.to_csv("train_clean.csv",encoding="utf-8")
