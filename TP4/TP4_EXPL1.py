import pandas as pd

print("pandas:" , pd.__version__)
fields=["ProductName","EngineVersion","AvSigVersion",
    "AVProductStatesIdentifier","AVProductsInstalled","AVProductsEnabled","HasTpm","CountryIdentifier",
    "CityIdentifier","Platform","Processor","OsVer","OsPlatformSubRelease","IsProtected","SMode",
    "SmartScreen","Firewall","Census_DeviceFamily","RtpStateBitfield","IsSxsPassiveMode",
    "Census_OSArchitecture", "Census_OSWUAutoUpdateOptionsName","Census_IsPortableOperatingSystem",
    "Census_GenuineStateName","Census_IsSecureBootEnabled","UacLuaenable","HasDetections"]

df = pd.read_csv('c:\\temp\\train.csv',usecols=fields,nrows=50000)
#df.dropna(thresh=3)
print("df.info()=" ,df.info())
#print("df.describe()=" ,df.describe())
print("len(df) before", len(df))

#df=df[~df.Census_IsSecureBootEnabled=="IS_GENUINE"]
#df=df.loc[df["HasDetections"]<=1]
SmartScreenWrongValues=['&#x01','&#x02']
df["SmartScreen"]=df["SmartScreen"].astype(str)
df.query("SmartScreen not in @SmartScreenWrongValues",inplace=True)

df["HasDetections"] = pd.to_numeric(df["HasDetections"], errors="coerce")
#df.query("HasDetections<=1",inplace=True)

#archs=['amd64','arm64','x86','x64']

#df=df.loc[df.Census_OSArchitecture.isin(archs)]
df["Census_OSArchitecture"]=df["Census_OSArchitecture"].astype(str)
df.query("Census_OSArchitecture in ['amd64','arm64','x86','x64']",inplace=True)

#bits=['0','1']
#df=df.loc[df["Census_IsSecureBootEnabled"].isin(bits)]
df["Census_IsSecureBootEnabled"]=pd.to_numeric(df["Census_IsSecureBootEnabled"], errors="coerce")
#df.query("Census_IsSecureBootEnabled<=1",inplace=True)
#df.dropna(subset=["Census_IsSecureBootEnabled","HasDetections"],axis=1,inplace=True)

print("len(df) after", len(df))
df=df.fillna(0)
df.to_csv("c:\\temp\\train_clean.csv")


#fields=["ProductName","EngineVersion","AvSigVersion",
#    "AVProductStatesIdentifier","AVProductsInstalled","AVProductsEnabled","HasTpm","CountryIdentifier",
#    "CityIdentifier","Platform","Processor","OsVer","OsPlatformSubRelease","IsProtected","SMode",
#    "SmartScreen","Firewall","Census_DeviceFamily","RtpStateBitfield","IsSxsPassiveMode",
#    "Census_OSArchitecture", "Census_OSWUAutoUpdateOptionsName","Census_IsPortableOperatingSystem",
#    "Census_GenuineStateName","Census_IsSecureBootEnabled","UacLuaenable"]
#
#df = pd.read_csv('c:\\temp\\test.csv',usecols=fields,nrows=10000)
##df.dropna(thresh=3)
#df.to_csv("c:\\temp\\test_clean.csv")





