import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import datetime
import gc 

gc.collect()
print("pandas:" , pd.__version__)

#fields=["ProductName","EngineVersion","AvSigVersion",
#    "AVProductStatesIdentifier","AVProductsInstalled","AVProductsEnabled","HasTpm","CountryIdentifier",
#    "CityIdentifier","Platform","Processor","OsVer","OsPlatformSubRelease","IsProtected","SMode",
#    "SmartScreen","Firewall","Census_DeviceFamily","RtpStateBitfield","IsSxsPassiveMode",
#    "Census_OSArchitecture", "Census_OSWUAutoUpdateOptionsName","Census_IsPortableOperatingSystem",
#    "Census_GenuineStateName","Census_IsSecureBootEnabled","UacLuaenable","HasDetections"]


fields=["ProductName","EngineVersion","AvSigVersion",
    "AVProductStatesIdentifier","AVProductsInstalled","AVProductsEnabled","HasTpm","CountryIdentifier",
    "CityIdentifier","Platform","Processor","OsVer","OsBuild","OsPlatformSubRelease","IsProtected","SMode",
    "SmartScreen","Firewall","Census_DeviceFamily","RtpStateBitfield","IsSxsPassiveMode",
    "Census_HasOpticalDiskDrive","Census_PrimaryDiskTypeName","DefaultBrowsersIdentifier",
    "Census_ProcessorCoreCount","Census_PrimaryDiskTotalCapacity","Census_TotalPhysicalRAM",
    "Census_OSArchitecture", "Census_OSWUAutoUpdateOptionsName","Census_IsPortableOperatingSystem",
    "Census_GenuineStateName","Census_IsSecureBootEnabled","UacLuaenable","HasDetections"]

	
#dtypes = {
#        'MachineIdentifier':                                    'category',
#        'ProductName':                                          'category',
#        'EngineVersion':                                        'category',
#        'AppVersion':                                           'category',
#        'AvSigVersion':                                         'category',
#        'IsBeta':                                               'int8',
#        'RtpStateBitfield':                                     'float16',
#        'IsSxsPassiveMode':                                     'int8',
#        'DefaultBrowsersIdentifier':                            'float16',
#        'AVProductStatesIdentifier':                            'float32',
#        'AVProductsInstalled':                                  'float16',
#        'AVProductsEnabled':                                    'float16',
#        'HasTpm':                                               'int8',
#        'CountryIdentifier':                                    'int16',
#        'CityIdentifier':                                       'float32',
#        'OrganizationIdentifier':                               'float16',
#        'GeoNameIdentifier':                                    'float16',
#        'LocaleEnglishNameIdentifier':                          'int8',
#        'Platform':                                             'category',
#        'Processor':                                            'category',
#        'OsVer':                                                'category',
#        'OsBuild':                                              'int16',
#        'OsSuite':                                              'int16',
#        'OsPlatformSubRelease':                                 'category',
#        'OsBuildLab':                                           'category',
#        'SkuEdition':                                           'category',
#        'IsProtected':                                          'float16',
#        'AutoSampleOptIn':                                      'int8',
#        'PuaMode':                                              'category',
#        'SMode':                                                'float16',
#        'IeVerIdentifier':                                      'float16',
#        'SmartScreen':                                          'category',
#        'Firewall':                                             'float16',
#        'UacLuaenable':                                         'float32',
#        'Census_MDC2FormFactor':                                'category',
#        'Census_DeviceFamily':                                  'category',
#        'Census_OEMNameIdentifier':                             'float16',
#        'Census_OEMModelIdentifier':                            'float32',
#        'Census_ProcessorCoreCount':                            'float16',
#        'Census_ProcessorManufacturerIdentifier':               'float16',
#        'Census_ProcessorModelIdentifier':                      'float16',
#        'Census_ProcessorClass':                                'category',
#        'Census_PrimaryDiskTotalCapacity':                      'float32',
#        'Census_PrimaryDiskTypeName':                           'category',
#        'Census_SystemVolumeTotalCapacity':                     'float32',
#        'Census_HasOpticalDiskDrive':                           'int8',
#        'Census_TotalPhysicalRAM':                              'float32',
#        'Census_ChassisTypeName':                               'category',
#        'Census_InternalPrimaryDiagonalDisplaySizeInInches':    'float16',
#        'Census_InternalPrimaryDisplayResolutionHorizontal':    'float16',
#        'Census_InternalPrimaryDisplayResolutionVertical':      'float16',
#        'Census_PowerPlatformRoleName':                         'category',
#        'Census_InternalBatteryType':                           'category',
#        'Census_InternalBatteryNumberOfCharges':                'float32',
#        'Census_OSVersion':                                     'category',
#        'Census_OSArchitecture':                                'category',
#        'Census_OSBranch':                                      'category',
#        'Census_OSBuildNumber':                                 'int16',
#        'Census_OSBuildRevision':                               'int32',
#        'Census_OSEdition':                                     'category',
#        'Census_OSSkuName':                                     'category',
#        'Census_OSInstallTypeName':                             'category',
#        'Census_OSInstallLanguageIdentifier':                   'float16',
#        'Census_OSUILocaleIdentifier':                          'int16',
#        'Census_OSWUAutoUpdateOptionsName':                     'category',
#        'Census_IsPortableOperatingSystem':                     'int8',
#        'Census_GenuineStateName':                              'category',
#        'Census_ActivationChannel':                             'category',
#        'Census_IsFlightingInternal':                           'float16',
#        'Census_IsFlightsDisabled':                             'float16',
#        'Census_FlightRing':                                    'category',
#        'Census_ThresholdOptIn':                                'float16',
#        'Census_FirmwareManufacturerIdentifier':                'float16',
#        'Census_FirmwareVersionIdentifier':                     'float32',
#        'Census_IsSecureBootEnabled':                           'int8',
#        'Census_IsWIMBootEnabled':                              'float16',
#        'Census_IsVirtualDevice':                               'float16',
#        'Census_IsTouchEnabled':                                'int8',
#        'Census_IsPenCapable':                                  'int8',
#        'Census_IsAlwaysOnAlwaysConnectedCapable':              'float16',
#        'Wdft_IsGamer':                                         'float16',
#        'Wdft_RegionIdentifier':                                'float16',
#        'HasDetections':                                        'int8'
#        }	


dtypes = {
        'MachineIdentifier':                                    'category',
        'ProductName':                                          'category',
        'EngineVersion':                                        'category',
        'AppVersion':                                           'category',
        'AvSigVersion':                                         'category',
        'IsBeta':                                               'int8',
        'RtpStateBitfield':                                     'int8',
        'IsSxsPassiveMode':                                     'int8',
        'DefaultBrowsersIdentifier':                            'float16',
        'AVProductStatesIdentifier':                            'float32',
        'AVProductsInstalled':                                  'float16',
        'AVProductsEnabled':                                    'float16',
        'HasTpm':                                               'int8',
        'CountryIdentifier':                                    'int8',
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
        'IsProtected':                                          'int8',
        'AutoSampleOptIn':                                      'int8',
        'PuaMode':                                              'category',
        'SMode':                                                'int8',
        'IeVerIdentifier':                                      'float16',
        'SmartScreen':                                          'category',
        'Firewall':                                             'int8',
        'UacLuaenable':                                         'int8',
        'Census_MDC2FormFactor':                                'category',
        'Census_DeviceFamily':                                  'category',
        'Census_OEMNameIdentifier':                             'float16',
        'Census_OEMModelIdentifier':                            'float32',
        'Census_ProcessorCoreCount':                            'float16',
        'Census_ProcessorManufacturerIdentifier':               'float16',
        'Census_ProcessorModelIdentifier':                      'float16',
        'Census_ProcessorClass':                                'category',
        'Census_PrimaryDiskTotalCapacity':                      'int64',
        'Census_PrimaryDiskTypeName':                           'category',
        'Census_SystemVolumeTotalCapacity':                     'int32',
        'Census_HasOpticalDiskDrive':                           'int8',
        'Census_TotalPhysicalRAM':                              'int32',
        'Census_ChassisTypeName':                               'category',
        'Census_InternalPrimaryDiagonalDisplaySizeInInches':    'int8',
        'Census_InternalPrimaryDisplayResolutionHorizontal':    'int8',
        'Census_InternalPrimaryDisplayResolutionVertical':      'int8',
        'Census_PowerPlatformRoleName':                         'category',
        'Census_InternalBatteryType':                           'category',
        'Census_InternalBatteryNumberOfCharges':                'int8',
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
        'Census_IsFlightingInternal':                           'int8',
        'Census_IsFlightsDisabled':                             'int8',
        'Census_FlightRing':                                    'category',
        'Census_ThresholdOptIn':                                'float16',
        'Census_FirmwareManufacturerIdentifier':                'float16',
        'Census_FirmwareVersionIdentifier':                     'float32',
        'Census_IsSecureBootEnabled':                           'int8',
        'Census_IsWIMBootEnabled':                              'int8',
        'Census_IsVirtualDevice':                               'int8',
        'Census_IsTouchEnabled':                                'int8',
        'Census_IsPenCapable':                                  'int8',
        'Census_IsAlwaysOnAlwaysConnectedCapable':              'int8',
        'Wdft_IsGamer':                                         'int8',
        'Wdft_RegionIdentifier':                                'int8',
        'HasDetections':                                        'int8'
        }	
gc.collect()

csv_read_start=datetime.datetime.now()
df1=pd.read_csv("C:\\TEMP\\TP4\\train.csv",usecols=fields,nrows=1000000,encoding="utf-8")
csv_read_end=datetime.datetime.now()

print("input CSV read time (secs):", str((csv_read_end-csv_read_start).total_seconds()))
print("ETL start..")

etl_start=datetime.datetime.now()

df=df1.sample(frac=0.1)
#df.to_csv("train_etl_final.csv",encoding="utf-8")
del df1
gc.collect()

print("df.info()=" ,df.info())

#print("df.describe()=" ,df.describe())
print("len(df) before", len(df))

df=df[pd.notnull(df["SmartScreen"])]
minidf=df[df.SmartScreen.str.contains("&#df0")]
print("len(minidf):", len(minidf))
indexNames= df[df['SmartScreen'].str.contains("&#df0")].index

# Delete these row indexes from dataFrame
df.drop(indexNames , inplace=True)

df.query("Census_IsSecureBootEnabled in [0,1]",inplace=True)
df.query("Census_OSArchitecture in ['amd64','arm64','df86','df64']",inplace=True)

print("len(df) after", len(df))

#df=df.fillna(0)

df.dropna(subset=["CityIdentifier","Census_TotalPhysicalRAM","Census_PrimaryDiskTotalCapacity","Census_ProcessorCoreCount"],inplace=True)

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
#df["OsVer"]=df["OsVer"].astype(str).replace(".","")
df.drop("OsVer",axis=1,inplace=True)

#Dummificar OsBuild
dummies_OsBuild=pd.get_dummies(df['OsBuild'],prefix='dummy_OsBuild',drop_first=True)
df=pd.concat([df,dummies_OsBuild],axis=1)
#df["OsVer"]=df["OsVer"].astype(str).replace(".","")
df.drop("OsBuild",axis=1,inplace=True)

#Dummificar EngineVersion
dummies_EngineVersion=pd.get_dummies(df['EngineVersion'],prefix='dummy_EngineVersion',drop_first=True)
df=pd.concat([df,dummies_EngineVersion],axis=1)

df.drop("EngineVersion",axis=1,inplace=True)

#df["EngineVersion"]=df["EngineVersion"].astype(str).replace(".","")

##Dummificar AvSigVersion
dummies_AvSigVersion=pd.get_dummies(df['AvSigVersion'],prefix='dummy_AvSigVersion',drop_first=True)
df=pd.concat([df,dummies_AvSigVersion],axis=1)

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

#Dummificar Census_HasOpticalDiskDrive
dummies_Census_HasOpticalDiskDrive=pd.get_dummies(df['Census_HasOpticalDiskDrive'],prefix='dummy_Census_HasOpticalDiskDrive',drop_first=True)
df=pd.concat([df,dummies_Census_HasOpticalDiskDrive],axis=1)

df.drop("Census_HasOpticalDiskDrive",axis=1,inplace=True)

#Census_PrimaryDiskTypeName

#Dummificar Census_PrimaryDiskTypeName
dummies_Census_PrimaryDiskTypeName=pd.get_dummies(df['Census_PrimaryDiskTypeName'],prefix='dummy_Census_PrimaryDiskTypeName',drop_first=True)
df=pd.concat([df,dummies_Census_PrimaryDiskTypeName],axis=1)

df.drop("Census_PrimaryDiskTypeName",axis=1,inplace=True)

#Dummificar DefaultBrowsersIdentifier
dummies_DefaultBrowsersIdentifier=pd.get_dummies(df['DefaultBrowsersIdentifier'],prefix='dummy_DefaultBrowsersIdentifier',drop_first=True)
df=pd.concat([df,dummies_DefaultBrowsersIdentifier],axis=1)

df.drop("DefaultBrowsersIdentifier",axis=1,inplace=True)

print("df después de dummificar:", df.info())

bestfeatures = SelectKBest(score_func=chi2, k=40)

y=df["HasDetections"]
X=df.drop("HasDetections",axis=1)

etl_end=datetime.datetime.now()

print("ETL processing time (secs):", str((etl_end-etl_start).total_seconds()))

print("Feature selection begin:")

gc.collect()

featureselection_start=datetime.datetime.now()

fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.sort_values(ascending=True,by="Score"))  #print all features
df1=featureScores[featureScores["Score"] < 0.05]
df2=featureScores[featureScores["Score"].isnull()]
print("df1:",df1)
print("df2:",df2)
#print("featureScores:" , featureScores)
df.drop(columns=df1["Specs"],inplace=True)
df.drop(columns=df2["Specs"],inplace=True)
del df1
del df2
gc.collect()

featureselection_end=datetime.datetime.now()
print("Feature selection end:" + str((featureselection_end-featureselection_start).total_seconds()))

output_write_start=datetime.datetime.now()
#df.to_csv("train_etl_final.csv",encoding="utf-8")
df.to_hdf('train_etl_final.h5', "MSMalwareTrainDS", table=True, mode='a')
output_write_end=datetime.datetime.now()

print("output CSV write time (secs):", str((output_write_end-output_write_start).total_seconds()))
del df
gc.collect()
print("READY!")