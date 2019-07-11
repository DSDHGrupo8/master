import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import datetime
 

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
	

csv_read_start=datetime.datetime.now()
#df=pd.read_csv("C:\\TEMP\\TP4\\train.csv",usecols=fields,nrows=250000,encoding="utf-8")
#df=pd.read_csv("train_cut.csv",encoding="utf-8",nrows=5000,dtype=dtypes)
df=pd.read_csv("train.csv",nrows=5000,encoding="utf-8",dtype=dtypes)
csv_read_end=datetime.datetime.now()

etl_start=datetime.datetime.now()
print("df.info()=" ,df.info())

colNames = df.columns

for col in colNames:
    try:
        if (df[col].dtype.name=="category"):
#           if (col.endswith("Version") or col.endswith("Ver")):
#                df[col]=df[col].str.replace(".","").astype(int)
            df[col]=df[col].str.replace(".","").astype(int)
            
            if (col.contains("Census_")):
                df.drop(df[col],inplace=True)
    except:
        print("skipping col:",col)

#df2=df.sample(frac=0.1)
#
#df2.to_csv("train_clean.csv",encoding="utf-8")
#del df
#del df2

#df=pd.read_csv("train_clean.csv",usecols=fields,encoding="utf-8")


#print("df.info()=" ,df.info())

#print("df.describe()=" ,df.describe())
#print("len(df) before", len(df))


################        
##DATA CLEANUP
################
df=df[pd.notnull(df["SmartScreen"])]
minidf=df[df.SmartScreen.str.contains("&#df0")]
print("len(minidf):", len(minidf))
indexNames= df[df['SmartScreen'].str.contains("&#df0")].index

# Delete these row indexes from dataFrame
df.drop(indexNames , inplace=True)

df.query("Census_IsSecureBootEnabled in [0,1]",inplace=True)
df.query("Census_OSArchitecture in ['amd64','arm64','x86','x64']",inplace=True)

print("len(df) after", len(df))

#df=df.fillna(0)
#df.dropna(subset=["CityIdentifier","Census_TotalPhysicalRAM","Census_PrimaryDiskTotalCapacity","Census_ProcessorCoreCount"],inplace=True)

################
#IMPUTACIONES
################

df.loc[df.AVProductStatesIdentifier.isnull(),"AVProductStatesIdentifier"]=0
df.loc[df.AVProductsInstalled.isnull(),"AVProductsInstalled"]=0
df.loc[df.AVProductsEnabled.isnull(),"AVProductsEnabled"]=0
df.loc[df.RtpStateBitfield.isnull(),"RtpStateBitfield"]=0
df.loc[df.UacLuaenable.isnull(),"UacLuaenable"]=0
df.loc[df.IsProtected.isnull(),"IsProtected"]=0
df.loc[df.Firewall.isnull(),"Firewall"]=0
df.loc[df.SMode.isnull(),"SMode"]=0

################################
#DUMMIFICACIONES A MANO
################################

#Dummificar ProductName
dummies_ProductName=pd.get_dummies(df['ProductName'],prefix='dummy_ProductName',drop_first=True)
df=pd.concat([df,dummies_ProductName],axis=1)
#df["OsVer"]=df["OsVer"].astype(str).replace(".","")
df.drop("ProductName",axis=1,inplace=True)

#Dummificar OsBuild
dummies_OsBuild=pd.get_dummies(df['OsBuild'],prefix='dummy_OsBuild',drop_first=True)
df=pd.concat([df,dummies_OsBuild],axis=1)
#df["OsVer"]=df["OsVer"].astype(str).replace(".","")
df.drop("OsBuild",axis=1,inplace=True)

#Dummificar Census_HasOpticalDiskDrive
dummies_Census_HasOpticalDiskDrive=pd.get_dummies(df['Census_HasOpticalDiskDrive'],prefix='dummy_Census_HasOpticalDiskDrive',drop_first=True)
df=pd.concat([df,dummies_Census_HasOpticalDiskDrive],axis=1)

df.drop("Census_HasOpticalDiskDrive",axis=1,inplace=True)

#Dummificar DefaultBrowsersIdentifier
dummies_DefaultBrowsersIdentifier=pd.get_dummies(df['DefaultBrowsersIdentifier'],prefix='dummy_DefaultBrowsersIdentifier',drop_first=True)
df=pd.concat([df,dummies_DefaultBrowsersIdentifier],axis=1)

df.drop("DefaultBrowsersIdentifier",axis=1,inplace=True)
df.drop("MachineIdentifier",axis=1,inplace=True)

bestfeatures = SelectKBest(score_func=chi2, k=40)

y=df["HasDetections"]
X=df.drop("HasDetections",axis=1)

#print("X.cols:",X.columns)
colNames=X.columns

keep_cols=X.select_dtypes(include=['int8','int16','int32','int64', 'float16', 'float32', 'float64']).columns
X=X[keep_cols]

print("Justo antes de correr el feature selector:",X.columns)
    
#########################
#SELECCION DE FEATURES
#########################

#fit = bestfeatures.fit(X,y)
#dfscores = pd.DataFrame(fit.scores_)
#dfcolumns = pd.DataFrame(X.columns)
#featureScores = pd.concat([dfcolumns,dfscores],axis=1)
#featureScores.columns = ['Specs','Score']  #naming the dataframe columns
#print(featureScores.sort_values(ascending=True,by="Score"))  #print all features
#df1=featureScores[featureScores["Score"] < 0.05]
#df2=featureScores[featureScores["Score"].isnull()]
#print("df1:",df1)
#print("df2:",df2)
#print("featureScores:" , featureScores)
#df.drop(columns=df1["Specs"],inplace=True)
#df.drop(columns=df2["Specs"],inplace=True)

etl_end=datetime.datetime.now()

csv_write_start=datetime.datetime.now()
df.to_csv("train_etl_etapa2.csv",encoding="utf-8")
csv_write_end=datetime.datetime.now()


print("input CSV read time (secs):", str((csv_read_end-csv_read_start).total_seconds()))
print("ETL processing time (secs):", str((etl_end-etl_start).total_seconds()))
print("output CSV read time (secs):", str((csv_write_end-csv_write_start).total_seconds()))
