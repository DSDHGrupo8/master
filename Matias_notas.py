!pip install -U -q PyDrive
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistics import mode
from matplotlib import cm as cm

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)
link = 'https://drive.google.com/open?id=1GXhb9LJJshv_gFdiMS6PujIG8SlhaZqv' # The shareable link
fluff, id = link.split('=')
downloaded = drive.CreateFile({'id':id})
downloaded.GetContentFile('properatti.csv')

#Opcional para leer columnas completas
#pd.set_option('display.max_colwidth', -1)
df = pd.read_csv("properatti.csv")
df.info()
df.columns

#Faltan varios datos importantes. Los de ubicación no tanto en principio, pero precio y superficie.
total = 121220
faltantes=df.isnull().sum()
print ('Datos faltantes')
faltantes

print ('Proporción de datos faltantes')
faltantes/total*100


#Top 10 lugares, están muy desgranados
#Medio que en place name no hay criterio
df['place_name'].value_counts().head(10)

#Solo hay operaciones de venta
df['operation'].value_counts()

#Es cualquiera el rango de pisos se puede repararar parcialmente con regex
print ('Piso mínimo es:' + str(df['floor'].min()))
print ('Piso máximo es:' + str(df['floor'].max()))
df['floor'].value_counts().head()

#Rango de expenses.
#No muy confiable
print ('Expensa mínima es:' + str(df['expenses'].min()))
print ('Expensa máxima es:' + str(df['expenses'].max()))
df['expenses'].value_counts().head()

#Es cualquiera el rango de AMBIENTES.
#No muy confiable
print ('Ambiente mínimo es:' + str(df['rooms'].min()))
print ('Ambiente máximo es:' + str(df['rooms'].max()))
df['rooms'].value_counts().head()

#Patrones para ambientes y pisos con los que se pueden rescatar valores para esas columnas antes de tirarlas
#Las descripciones son IMPOSIBLES pero algo se saca, quizás sea mucho trabajo.
#Piso
#pattern= r'\|([\w.\s]*)\|([\w.\s]*)\|([\w.\s]*)'
#regex=re.compile(pattern, flags=re.IGNORECASE)
#m=regex.findall(df2['place_with_parent_names'][1])
#m

#ambiente
#(\d+\sambiente)

#State name parece una columna menos dispersa que place_name
#Las zonas en que está dividido BS AS son muchas pero son categorías de venta usadas por inmobiliarias
df['state_name'].value_counts()

#Sugerencia
#Quedarnos con
#'property_type', 'place_name', 'state_name',
#'geonames_id','lat-lon', 'lat', 'lon',
#'price_aprox_usd', 'surface_total_in_m2','price_usd_per_m2',
#'floor', 'rooms'
#Y pasarlas a español o algo con menos guiones

#Sacar precio promedio por state_name y llenar los faltantes de precio?
#Acá una evaluación sobre una columna con todos los precios equiparados
#Habría que calcular la cotización del dólar en el df, actualizarlo y completar para todos los que estén en pesos
#Completar los nan con promedio de precio por zona (por place sería mejor, pero es un descalabro esa columna)
df[['price_aprox_usd','state_name']].dropna().groupby('state_name').describe()

pd.set_option('display.max_columns', 50)
df[['surface_total_in_m2','surface_covered_in_m2']].fillna(0)
len(df[df.loc[:,'surface_total_in_m2']<df.loc[:,'surface_covered_in_m2']])

df[['surface_total_in_m2','surface_covered_in_m2']].fillna(0)
valores_iguales = df[df.loc[:,'surface_total_in_m2']==df.loc[:,'surface_covered_in_m2']]
print(len(valores_iguales))
print(len(valores_iguales)/len(df.loc[:,'surface_total_in_m2']))

len(df[df.duplicated('description') == True])

type(df['surface_total_in_m2'])
