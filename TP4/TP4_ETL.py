# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 21:10:01 2019

@author: Adrian
"""

import pandas as pd



def chunk_preprocessing(df_chunk,counter):
    print("processing chunk number:" , counter)
    #print(df_chunk.describe())
    #print(df_chunk.var())
    df_chunk.dropna(thresh=10,inplace=True)
    df_chunk.loc[df_chunk.SmartScreen.isnull(),"SmartScreen"]="Off"
    df_chunk.loc[df_chunk.Firewall.isnull(),"Firewall"]=0
    df_chunk.loc[df_chunk.SMode.isnull(),"SMode"]=0
    df_chunk.to_csv("C:\\Temp\\test\\train_clean_" + str(counter) + ".csv")
    print("written chunk number:", counter)
    
    #return df_chunk

print("pandas version:" , pd.__version__)
    

fields=["MachineIdentifier","ProductName","EngineVersion","AvSigVersion"
        ,"AVProductStatesIdentifier","AVProductsInstalled","AVProductsEnabled","HasTpm",
        "Platform","Processor","OsPlatformSubRelease","IsProtected","SMode",
        "SmartScreen","Firewall","Census_DeviceFamily","Census_OSVersion",
        "Census_OSArchitecture", "Census_OSWUAutoUpdateOptionsName","Census_IsPortableOperatingSystem"
        ,"Census_GenuineStateName","Census_IsSecureBootEnabled","UacLuaenable","HasDetections"]

df_chunk=pd.read_csv("C:\\Temp\\train.csv",chunksize=500000,usecols=fields)

chunk_list = []  # append each chunk df here 

# Each chunk is in df format
counter=1

for chunk in df_chunk:  
    # perform data filtering 
    chunk_preprocessing(chunk,counter)
    counter+=1
    
print("Process finished")

