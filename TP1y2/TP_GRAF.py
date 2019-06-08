# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistics import mode 
from matplotlib import cm as cm
from sklearn import svm
import seaborn as sns

df = pd.read_csv('datasets\\Properati_CABA_DS.csv')
cols = ['Unnamed: 0']
df.drop(cols, axis=1, inplace=True)
print(df.columns)
print(df.shape)

#unificar las variantes de "Palermo"
df['place_name'] = df.loc[:,'place_name'].replace(['Palermo Soho','Palermo Hollywood','Palermo Chico','Palermo Viejo'], 'Palermo')
#unificar "house" y "ph"
df['property_type'] = df.loc[:,'property_type'].replace('PH', 'house')
#Nombres comprensibles para graficar
df.columns = ['tipo', 'barrio', 'lat-lon', 'precio','sup_total', 'sup_cubierta', 'precio_usd__m2','precio_m2', 'expensas', 'description', 'title']
#Eliminar Outlier gigante
df.drop([5684], inplace = True)
df.sort_values(['precio'], ascending = False).head(3)

#Chequear valores categóricos por columnas
print(df.loc[:,'barrio'].unique())
print(df.loc[:,'tipo'].unique())

#Resumir nombres de barrios
df['barrio'] = df['barrio'].replace(['Mataderos', 'Belgrano', 'Palermo', 'Flores', 'Boedo', 'Las Cañitas', 'Puerto Madero', 'Balvanera', 'Caballito', 'Nuñez', 'Almagro', 'Capital Federal', 'Floresta', 'Barracas', 'Recoleta', 'Congreso', 'Villa Crespo', 'Chacarita', 'Constitución', 'Colegiales', 'Villa Urquiza', 'Barrio Norte', 'Saavedra', 'Paternal', 'Agronomía', 'Villa Pueyrredón', 'Coghlan', 'Parque Centenario', 'San Telmo', 'Monserrat', 'Boca', 'Parque Avellaneda', 'San Cristobal', 'Abasto', 'Versalles', 'Villa del Parque', 'Monte Castro', 'Retiro', 'Parque Chas', 'Villa Devoto', 'Centro / Microcentro', 'Liniers', 'Tribunales', 'Once', 'San Nicolás', 'Parque Chacabuco', 'Velez Sarsfield', 'Catalinas', 'Pompeya', 'Villa Lugano', 'Parque Patricios', 'Villa Luro' ,'Villa General Mitre', 'Villa Ortuzar', 'Villa Santa Rita', 'Villa Soldati', 'Villa Real', 'Villa Riachuelo'], ['Mataderos', 'Liniers', 'Belgrano', 'Palermo', 'Flores', 'Boedo', 'Cañitas',
 'P_Madero', 'Balvanera', 'Caballito', 'Nuñez', 'S_Telmo', 'Almagro',
 'CABA', 'Colegiales', 'Floresta', 'B_Norte', 'Barracas',
 'Recoleta', 'Congreso', 'V_Crespo', 'Chacarita', 'Constitución',
 'V_Urquiza', 'Saavedra', 'Monserrat', 'Pompeya', 'P_Chas', 'Paternal',
 'Agronomía', 'V_Pueyrredón', 'Coghlan', 'P_Centenario', 'V_Luro',
 'V_Devoto', 'Boca', 'P_Avellaneda', 'S_Cristobal',
 'Velez', 'Abasto', 'Versalles', 'V_dParque', 'M_Castro',
 'Retiro', 'P_Patricios', 'S_Nicolás', 'V_S_Rita',
 'Centro', 'Once', 'Tribunales', 'P_Chacabuco', 'Catalinas',
 'V_Gral_Mitre', 'V_Lugano', 'V_Ortuzar', 'V_Soldati',
 'V_Real', 'V_Riachuelo'])

#Ver Cantidades por tipo
print(df.loc[:,'tipo'].value_counts().to_frame())
#Ver cantidades por barrio
print(df.loc[:,'barrio'].value_counts().to_frame().head(10))

#distribución de propiedades por barrio según precio y superficie total
#sns.lmplot(x="sup_total", y="precio", data = df, hue = 'barrio', height=10, fit_reg=False);
sns.lmplot(x="sup_total", y="precio", data = df, hue = 'barrio', fit_reg=False);
plt.savefig('distribución de propiedades por barrio según precio y superficie total')

#Medias de precio y superficie por barrio ordenadas de mayor (rojo) a menor (azul)
df_means = df[['precio','sup_total','barrio']].groupby(['barrio']).mean().sort_values('precio', ascending = False)
df_means.reset_index(inplace = True)
#sns.lmplot(x="sup_total", y="precio", data = df_means, hue = 'barrio', palette = 'RdYlGn', height=10, fit_reg=False)
sns.lmplot(x="sup_total", y="precio", data = df_means, hue = 'barrio', palette = 'RdYlGn', fit_reg=False)
plt.savefig('Medias de precio y superficie por barrio ordenadas de mayor (rojo) a menor (verde)')

#Grilla de pares de precio y superficie total
g = sns.PairGrid(df, vars=['precio', 'sup_total'],hue='tipo')
g.map(plt.scatter, alpha=0.8)
g.add_legend();

#Precio medio por barrio
precioMedioBarrio = df[['precio','barrio']].groupby(['barrio']).mean().sort_values('precio', ascending = False)
precioMedioBarrio.plot.bar(figsize=(30, 10))
plt.grid(True)
plt.axis('tight')
plt.xlabel('Barrio')
plt.ylabel('Precio Aprox. USD')
plt.title('Media de Precio por Barrio');
plt.savefig('Media de precio por barrio')

#Matriz de correlación
k = 12
cols = df.corr().nlargest(k,'precio')['precio'].index
cm = df[cols].corr()
plt.figure(figsize=(10,6))
sns.heatmap(cm, annot=True, cmap = 'OrRd');
plt.savefig('Matriz de correlación con valores')

# Boxplot agrupado de todos los barrios
plt.figure(figsize=(50, 10))
sns.boxplot(y="precio", x="barrio", hue="tipo", data=df, palette="colorblind",width=0.5);
plt.legend(loc='upper left')
#sns.plt.show()
plt.savefig('Outliers por barrio y tipo de propiedad')

#boxplot por barrio
dfpalermo = df.loc[df['barrio'] == 'Palermo']
plt.figure(figsize=(10, 10))
sns.boxplot(y="precio", x="barrio", hue="tipo", data=dfpalermo, palette="colorblind",width=0.5);
plt.legend(loc='upper left')
#sns.plt.show()
plt.savefig('Outliers Palermo por tipo de propiedad')

#Tipo por barrio
df2 = df[['precio','tipo','barrio']].groupby(['barrio', 'tipo']).count().sort_values(['precio','tipo', 'barrio'], ascending = False)
df2.reset_index(inplace = True)
df2.columns = ['Barrio', 'Tipo', 'Cantidad']
plt.figure(figsize=(50, 10))
sns.barplot(x="Barrio", y="Cantidad", hue="Tipo",data=df2, palette="colorblind");
plt.legend(loc='upper left')
plt.savefig('Cantidades de tipo por barrio')

#Top 5 Casas
plt.figure(figsize=(30, 10))
sns.barplot(x="Barrio", y="Cantidad", hue="Tipo",data=df2.loc[df2['Tipo'] == 'house'].head(5), palette="YlOrRd");
plt.legend(loc='upper left')
plt.savefig('Top 5 casas')

#Top 5 Locales
plt.figure(figsize=(30, 10))
sns.barplot(x="Barrio", y="Cantidad", hue="Tipo",data=df2.loc[df2['Tipo'] == 'store'].head(5), palette="YlGn");
plt.legend(loc='upper left')
plt.savefig('Top 5 locales')

#Top 5 Deptos
plt.figure(figsize=(30, 10))
sns.barplot(x="Barrio", y="Cantidad", hue="Tipo",data=df2.loc[df2['Tipo'] == 'apartment'].head(5), palette="PuBu");
plt.legend(loc='upper left')
plt.savefig('Top 5 Deptos')