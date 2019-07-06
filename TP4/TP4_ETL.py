import pandas as pd

 

print("pandas:" , pd.__version__)

#fields=["ProductName","EngineVersion","AvSigVersion",
#    "AVProductStatesIdentifier","AVProductsInstalled","AVProductsEnabled","HasTpm","CountryIdentifier",
#    "CityIdentifier","Platform","Processor","OsVer","OsPlatformSubRelease","IsProtected","SMode",
#    "SmartScreen","Firewall","Census_DeviceFamily","RtpStateBitfield","IsSxsPassiveMode",
#    "Census_OSArchitecture", "Census_OSWUAutoUpdateOptionsName","Census_IsPortableOperatingSystem",
#    "Census_GenuineStateName","Census_IsSecureBootEnabled","UacLuaenable","HasDetections"]


fields=["ProductName","EngineVersion","AvSigVersion",
    "AVProductStatesIdentifier","AVProductsInstalled","AVProductsEnabled","HasTpm","CountryIdentifier",
    "CityIdentifier","Platform","Processor","OsVer","OsPlatformSubRelease","IsProtected","SMode",
    "SmartScreen","Firewall","Census_DeviceFamily","RtpStateBitfield","IsSxsPassiveMode",
    "Census_OSArchitecture", "Census_OSWUAutoUpdateOptionsName","Census_IsPortableOperatingSystem",
    "Census_GenuineStateName","Census_IsSecureBootEnabled","UacLuaenable","HasDetections"]

	
dtypes = {
        'MachineIdentifier':                                    'category',
        'ProductName':                                          'category',
        'EngineVersion':                                        'category',
        'AppVersion':                                           'category',
        'AvSigVersion':                                         'category',
        'IsBeta':                                               'int8',
        'RtpStateBitfield':                                     'float16',
        'IsSxsPassiveMode':                                     'int8',
        'DefaultBrowsersIdentifier':                            'float16',
        'AVProductStatesIdentifier':                            'float32',
        'AVProductsInstalled':                                  'float16',
        'AVProductsEnabled':                                    'float16',
        'HasTpm':                                               'int8',
        'CountryIdentifier':                                    'int16',
        'CityIdentifier':                                       'float32',
        'OrganizationIdentifier':                               'float16',
        'GeoNameIdentifier':                                    'float16',
        'LocaleEnglishNameIdentifier':                          'int8',
        'Platform':                                             'category',
        'Processor':                                            'category',
        'OsVer':                                                'category',
        'OsBuild':                                              'int16',
        'OsSuite':                                              'int16',
        'OsPlatformSubRelease':                                 'category',
        'OsBuildLab':                                           'category',
        'SkuEdition':                                           'category',
        'IsProtected':                                          'float16',
        'AutoSampleOptIn':                                      'int8',
        'PuaMode':                                              'category',
        'SMode':                                                'float16',
        'IeVerIdentifier':                                      'float16',
        'SmartScreen':                                          'category',
        'Firewall':                                             'float16',
        'UacLuaenable':                                         'float32',
        'Census_MDC2FormFactor':                                'category',
        'Census_DeviceFamily':                                  'category',
        'Census_OEMNameIdentifier':                             'float16',
        'Census_OEMModelIdentifier':                            'float32',
        'Census_ProcessorCoreCount':                            'float16',
        'Census_ProcessorManufacturerIdentifier':               'float16',
        'Census_ProcessorModelIdentifier':                      'float16',
        'Census_ProcessorClass':                                'category',
        'Census_PrimaryDiskTotalCapacity':                      'float32',
        'Census_PrimaryDiskTypeName':                           'category',
        'Census_SystemVolumeTotalCapacity':                     'float32',
        'Census_HasOpticalDiskDrive':                           'int8',
        'Census_TotalPhysicalRAM':                              'float32',
        'Census_ChassisTypeName':                               'category',
        'Census_InternalPrimaryDiagonalDisplaySizeInInches':    'float16',
        'Census_InternalPrimaryDisplayResolutionHorizontal':    'float16',
        'Census_InternalPrimaryDisplayResolutionVertical':      'float16',
        'Census_PowerPlatformRoleName':                         'category',
        'Census_InternalBatteryType':                           'category',
        'Census_InternalBatteryNumberOfCharges':                'float32',
        'Census_OSVersion':                                     'category',
        'Census_OSArchitecture':                                'category',
        'Census_OSBranch':                                      'category',
        'Census_OSBuildNumber':                                 'int16',
        'Census_OSBuildRevision':                               'int32',
        'Census_OSEdition':                                     'category',
        'Census_OSSkuName':                                     'category',
        'Census_OSInstallTypeName':                             'category',
        'Census_OSInstallLanguageIdentifier':                   'float16',
        'Census_OSUILocaleIdentifier':                          'int16',
        'Census_OSWUAutoUpdateOptionsName':                     'category',
        'Census_IsPortableOperatingSystem':                     'int8',
        'Census_GenuineStateName':                              'category',
        'Census_ActivationChannel':                             'category',
        'Census_IsFlightingInternal':                           'float16',
        'Census_IsFlightsDisabled':                             'float16',
        'Census_FlightRing':                                    'category',
        'Census_ThresholdOptIn':                                'float16',
        'Census_FirmwareManufacturerIdentifier':                'float16',
        'Census_FirmwareVersionIdentifier':                     'float32',
        'Census_IsSecureBootEnabled':                           'int8',
        'Census_IsWIMBootEnabled':                              'float16',
        'Census_IsVirtualDevice':                               'float16',
        'Census_IsTouchEnabled':                                'int8',
        'Census_IsPenCapable':                                  'int8',
        'Census_IsAlwaysOnAlwaysConnectedCapable':              'float16',
        'Wdft_IsGamer':                                         'float16',
        'Wdft_RegionIdentifier':                                'float16',
        'HasDetections':                                        'int8'
        }	
	
df = pd.read_csv('C:\\TEMP\\TP4\\train.csv',usecols=fields,nrows=500000,dtype=dtypes)
print("df.info()=" ,df.info())

#print("df.describe()=" ,df.describe())
print("len(df) before", len(df))

df=df[pd.notnull(df["SmartScreen"])]
minidf=df[df.SmartScreen.str.contains("&#df0")]
print("len(minidf):", len(minidf))
indexNames= df[df['SmartScreen'].str.contains("&#df0")].index

# Delete these row indedfes from dataFrame
df.drop(indexNames , inplace=True)

df.query("Census_IsSecureBootEnabled in [0,1]",inplace=True)
df.query("Census_OSArchitecture in ['amd64','arm64','df86','df64']",inplace=True)

print("len(df) after", len(df))

#df=df.fillna(0)

df.dropna(subset=["CityIdentifier"],inplace=True)

df.loc[df.AVProductStatesIdentifier.isnull(),"AVProductStatesIdentifier"]=0
df.loc[df.AVProductsInstalled.isnull(),"AVProductsInstalled"]=0
df.loc[df.AVProductsEnabled.isnull(),"AVProductsEnabled"]=0
df.loc[df.RtpStateBitfield.isnull(),"RtpStateBitfield"]=0
df.loc[df.UacLuaenable.isnull(),"UacLuaenable"]=0
df.loc[df.IsProtected.isnull(),"IsProtected"]=0
df.loc[df.Firewall.isnull(),"Firewall"]=0
df.loc[df.SMode.isnull(),"SMode"]=0

#Dummificar ProductName
dummies_productName=pd.get_dummies(df['ProductName'],prefix='dummy_ProductName',drop_first=True)
df=pd.concat([df,dummies_productName],axis=1)
   
df.drop("ProductName",axis=1,inplace=True)

#Dummificar OsVer
dummies_OsVer=pd.get_dummies(df['OsVer'],prefix='dummy_OsVer',drop_first=True)
df=pd.concat([df,dummies_OsVer],axis=1)

df.drop("OsVer",axis=1,inplace=True)

#Dummificar EngineVersion
dummies_EngineVersion=pd.get_dummies(df['EngineVersion'],prefix='dummy_EngineVersion',drop_first=True)
df=pd.concat([df,dummies_EngineVersion],axis=1)

df.drop("EngineVersion",axis=1,inplace=True)

##Dummificar AvSigVersion
#dummies_AvSigVersion=pd.get_dummies(df['AvSigVersion'],prefix='dummy_AvSigVersion',drop_first=True)
#df=pd.concat([df,dummies_AvSigVersion],axis=1)

df["AvSigVersion"]=df["AvSigVersion"].astype(str).replace(".","")
df.drop("AvSigVersion",axis=1,inplace=True)

#Dummificar Platform
dummies_Platform=pd.get_dummies(df['Platform'],prefix='dummy_Platform',drop_first=True)
df=pd.concat([df,dummies_Platform],axis=1)

df.drop("Platform",axis=1,inplace=True)

#Dummificar Processor
dummies_Processor=pd.get_dummies(df['Processor'],prefix='dummy_Processor',drop_first=True)
df=pd.concat([df,dummies_Processor],axis=1)

df.drop("Processor",axis=1,inplace=True)

#Dummificar OsPlatformSubRelease
dummies_OsPlatformSubRelease=pd.get_dummies(df['OsPlatformSubRelease'],prefix='dummy_OsPlatformSubRelease',drop_first=True)
df=pd.concat([df,dummies_OsPlatformSubRelease],axis=1)

df.drop("OsPlatformSubRelease",axis=1,inplace=True)

#Dummificar SmartScreen
dummies_SmartScreen=pd.get_dummies(df['SmartScreen'],prefix='dummy_SmartScreen',drop_first=True)
df=pd.concat([df,dummies_SmartScreen],axis=1)

df.drop("SmartScreen",axis=1,inplace=True)
  
#Dummificar Census_DeviceFamily
dummies_Census_DeviceFamily=pd.get_dummies(df['Census_DeviceFamily'],prefix='dummy_Census_DeviceFamily',drop_first=True)
df=pd.concat([df,dummies_Census_DeviceFamily],axis=1)

df.drop("Census_DeviceFamily",axis=1,inplace=True)

#Dummificar Census_OSArchitecture
dummies_Census_OSArchitecture=pd.get_dummies(df['Census_OSArchitecture'],prefix='dummy_Census_OSArchitecture',drop_first=True)
df=pd.concat([df,dummies_Census_OSArchitecture],axis=1)

df.drop("Census_OSArchitecture",axis=1,inplace=True)

#Dummificar Census_OSWUAutoUpdateOptionsName
dummies_Census_OSWUAutoUpdateOptionsName=pd.get_dummies(df['Census_OSWUAutoUpdateOptionsName'],prefix='dummy_Census_OSWUAutoUpdateOptionsName',drop_first=True)
df=pd.concat([df,dummies_Census_OSWUAutoUpdateOptionsName],axis=1)

df.drop("Census_OSWUAutoUpdateOptionsName",axis=1,inplace=True)

#Dummificar Census_GenuineStateName
dummies_Census_GenuineStateName=pd.get_dummies(df['Census_GenuineStateName'],prefix='dummy_Census_GenuineStateName',drop_first=True)
df=pd.concat([df,dummies_Census_GenuineStateName],axis=1)

df.drop("Census_GenuineStateName",axis=1,inplace=True)

df.to_csv("train_clean.csv")
