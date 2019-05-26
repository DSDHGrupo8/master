# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistics import mode 
from matplotlib import cm as cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn import svm
import seaborn as sns

df = pd.read_csv('datasets\\properati_caballito.csv')

dummycols = [col for col in df if col.startswith('dummy_')]
distcols=[col for col in df if col.startswith('dist')]
cols=['lat', 'lon', 'surface_total_in_m2','price_aprox_usd','expenses'] + dummycols + distcols


#df.drop(cols, axis=1, inplace=True)
#print(df.columns)
#print(df.shape)

#Distribución de precio aproximado
sns.distplot(df.loc[:,'price_aprox_usd'], kde=False);
plt.axvline(0, color="k", linestyle="--");

#Distribución de superficie
sns.distplot(df.loc[:,'surface_total_in_m2'], kde=False);
plt.axvline(0, color="k", linestyle="--");

#unificar "house" y "ph"
df['property_type'] = df.loc[:,'property_type'].replace('PH', 'house')


#Chequear valores categóricos por columnas
print(df.loc[:,'property_type'].unique())

#Resumir nombres de barrios
#Ver Cantidades por tipo
print(df.loc[:,'property_type'].value_counts().to_frame())

#Grilla de pares de precio y superficie total
print("Grilla de pares de precio y superficie total")
g = sns.PairGrid(df, vars=['distSubte','distParque','expenses','price_aprox_usd', 'surface_total_in_m2'],hue='property_type')
g.map(plt.scatter, alpha=0.8)
g.add_legend()
plt.savefig('pairgrid')
plt.show();

fig = plt.figure(figsize=(10,10))
ax = Axes3D(fig) # Method 1

x=df["lat"]
y=df["lon"]
z=df["price_aprox_usd"]
ax.scatter(x, y, z, c=x, marker='*')
ax.set_xlabel('Latitud')
ax.set_ylabel('Longitud')
ax.set_zlabel('Precio u$s')
plt.savefig('3dscatter_lat_lon_precio')

plt.show()

#Matriz de correlación
print("Matriz de correlación")
k = 12
cols = df.corr().nlargest(k,'price_aprox_usd')['price_aprox_usd'].index
cm = df[cols].corr()
plt.figure(figsize=(10,6))
sns.heatmap(cm, annot=True, cmap = 'OrRd');
plt.savefig('matrix_corr')
plt.show()