import pandas as pd

 

print("pandas:" , pd.__version__)

fields=["ProductName","EngineVersion","AvSigVersion",
    "AVProductStatesIdentifier","AVProductsInstalled","AVProductsEnabled","HasTpm","CountryIdentifier",
    "CityIdentifier","Platform","Processor","OsVer","OsPlatformSubRelease","IsProtected","SMode",
    "SmartScreen","Firewall","Census_DeviceFamily","RtpStateBitfield","IsSxsPassiveMode",
    "Census_OSArchitecture", "Census_OSWUAutoUpdateOptionsName","Census_IsPortableOperatingSystem",
    "Census_GenuineStateName","Census_IsSecureBootEnabled","UacLuaenable","HasDetections"]

df = pd.read_csv('c:\\TEMP\\TP4\\train.csv',usecols=fields,nrows=50000)
print("df.info()=" ,df.info())

#print("df.describe()=" ,df.describe())
print("len(df) before", len(df))

df=df[pd.notnull(df["SmartScreen"])]
minidf=df[df.SmartScreen.str.contains("&#x0")]
print("len(minidf):", len(minidf))
indexNames= df[df['SmartScreen'].str.contains("&#x0")].index

# Delete these row indexes from dataFrame
df.drop(indexNames , inplace=True)

df.query("Census_IsSecureBootEnabled in [0,1]",inplace=True)
df.query("Census_OSArchitecture in ['amd64','arm64','x86','x64']",inplace=True)

print("len(df) after", len(df))

df=df.fillna(0)
df.to_csv("train_clean.csv")