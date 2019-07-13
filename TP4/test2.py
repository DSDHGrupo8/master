# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 11:26:15 2019

@author: Adrian
"""

import pandas as pd
df=pd.read_csv("C:\\TEMP\\TP4\\train.csv",nrows=50000,encoding="utf-8")

df.query("Census_IsSecureBootEnabled in [0,1]",inplace=True)
df.query("Census_OSArchitecture in ['amd64','arm64','x86','x64']",inplace=True)
df.query("Census_TotalPhysicalRAM<=16384",inplace=True)
df.query("Census_ProcessorCoreCount<=8",inplace=True)
df.query("Census_PrimaryDiskTotalCapacity<=1048576",inplace=True)


df.to_csv("train_small.csv",encoding="utf-8")
