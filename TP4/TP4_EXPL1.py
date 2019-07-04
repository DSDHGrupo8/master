import pandas as pd
import numpy as np

fields=["ProductName","EngineVersion","AvSigVersion",
    "AVProductStatesIdentifier","AVProductsInstalled","AVProductsEnabled","HasTpm","CountryIdentifier",
    "CityIdentifier","Platform","Processor","OsVer","OsPlatformSubRelease","IsProtected","SMode",
    "SmartScreen","Firewall","Census_DeviceFamily","RtpStateBitfield","IsSxsPassiveMode",
    "Census_OSArchitecture", "Census_OSWUAutoUpdateOptionsName","Census_IsPortableOperatingSystem",
    "Census_GenuineStateName","Census_IsSecureBootEnabled","UacLuaenable","HasDetections"]

df = pd.read_csv('train.csv',usecols=fields,nrows=50000)
#df.dropna(thresh=3)

#df=df[~df.Census_IsSecureBootEnabled=="IS_GENUINE"]
df=df[df.HasDetections<=1]
archs=['amd64','arm64','x86','x64']
df=df[df.Census_OSArchitecture.isin(archs)]

df=df.query("Census_IsSecureBootEnabled!='IS_GENUINE'")


df.to_csv("train_clean.csv")
print(df.describe)

fields=["ProductName","EngineVersion","AvSigVersion",
    "AVProductStatesIdentifier","AVProductsInstalled","AVProductsEnabled","HasTpm","CountryIdentifier",
    "CityIdentifier","Platform","Processor","OsVer","OsPlatformSubRelease","IsProtected","SMode",
    "SmartScreen","Firewall","Census_DeviceFamily","RtpStateBitfield","IsSxsPassiveMode",
    "Census_OSArchitecture", "Census_OSWUAutoUpdateOptionsName","Census_IsPortableOperatingSystem",
    "Census_GenuineStateName","Census_IsSecureBootEnabled","UacLuaenable"]

df = pd.read_csv('test.csv',usecols=fields,nrows=10000)
#df.dropna(thresh=3)
df.to_csv("test_clean.csv")
df = pd.read_csv('test.csv',nrows=10000)
#df.dropna(thresh=3)
df.to_csv("test_clean_2.csv")
print(df.describe)



