import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistics import mode 
from matplotlib import cm as cm
#from sklearn import svm
import seaborn as sns

pd.set_option('display.expand_frame_repr', False)
pd.options.display.float_format = '{:.2f}'.format

class tp1_VIZ:    
  
    def conteo_por_grupos(self):
        totalregs=len(self.df)
        self.df['COUNTER'] =1       #initially, set that counter to 1.
        #g=self.df.groupby(['state_name','place_name'])['COUNTER'].size();
        g=self.df.groupby(['state_name','place_name'])['COUNTER'].size()/totalregs * 100;
        g=g.sort_values(ascending=False)
        print(g)
  
    def correlation_matrix(self):
        self.df = pd.read_csv('Properati_CABA_DS_fixed.csv')
        #plt.rcParams.update({'font.size': 12})
        fig = plt.figure()
        fig.set_figheight(10)
        fig.set_figwidth(50)
        ax1 = fig.add_subplot(111)
        cmap = cm.get_cmap('jet', 20)
        cax = ax1.imshow(self.df.corr(), interpolation="nearest", cmap=cmap)
        ax1.grid(True)
        plt.title('Correlación entre variables')
        labels=self.df.columns
        ax1.set_xticklabels(labels,fontsize=12)
        ax1.set_yticklabels(labels,fontsize=12)
        # Add colorbar, make sure to specify tick locations to match desired ticklabels
        fig.colorbar(cax, ticks=[0,0.25,0.50,0.75,1])
        plt.show()
    
    def mostrarHistogramas(self):
        
        precioSerie=self.df['price_aprox_usd']
        
        print("******************PRECIO**************************************")
        print("Promedio de Precio:" + str(np.round(np.mean(precioSerie),2)))
        print("Mediana del Precio:" + str(np.round(np.median(precioSerie),2)))
        print("Moda:" + str(mode(precioSerie)))
        
        plt.hist(precioSerie,bins=50)
        plt.xlabel("Precio en USD - propiedades")
        plt.ylabel("Frecuencia")
        plt.show()
        
        supSerie=self.df['surface_total_in_m2']
        
        print("******************SUPERFICIE**************************************")
        print("Promedio de superficie:" + str(np.round(np.mean(supSerie),2)))
        print("Mediana de superficie:" + str(np.round(np.median(supSerie),2)))
        print("Moda:" + str(mode(supSerie)))
        
        
        print("Cantidad de registros con price mayor > 0 sin nulos:" ,str(len(precioSerie)))
        
        plt.hist(supSerie,bins=50)
        plt.xlabel("Superficie m2 - propiedades")
        plt.ylabel("Frecuencia")
        plt.show()


    def correrAnalisis(self):
        
        
        print("Conteo total de registros sin nulos en price y surface_total:" , str(len(self.df3)))
        
        #print(precioSerie)
        
        
        print("**********************DF2*******************************")
        print(self.df2.describe())
        print("*****************************************************")
        
        
        print("**********************DF3*******************************")
        print(self.df3.describe())
        print("*****************************************************")

x=tp1_VIZ()
#x.correlation_matrix()
#x.conteo_por_grupos()

df2 = pd.read_csv('Properati_CABA_DS_fixed.csv')
cols = ['Unnamed: 0', 'Unnamed: 0.1']
df2.drop(cols, axis=1, inplace=True)
df2.loc[:,'property_type'].value_counts().to_frame()
df2.columns
#Unificar PH y House
df2.loc[:,'property_type'].value_counts().to_frame()
#Unificar las variantes de Palermo
df2['place_name'] = df_test=df2.loc[:,'place_name'].replace(['Palermo Soho','Palermo Hollywood','Palermo Chico','Palermo Viejo'], 'Palermo')
df2['property_type'] = df_test=df2.loc[:,'property_type'].replace('PH', 'house')
#Valores categóricos por columnas
#print(df2.loc[:,'state_name'].unique())
#print(df2.loc[:,'place_name'].unique())
#print(df2.loc[:,'property_type'].unique())
#Duplicados de description  
df2.duplicated('description').value_counts().to_frame()

#Duplicados de lat, lon y description 
df2.duplicated(subset=['lat','lon','description']).value_counts().to_frame()

#¿Qué pasaba en el original?
df_original = pd.read_csv('C:\\Users\\Public\\properati.csv')
df_original.duplicated(subset=['lat','lon','description']).value_counts().to_frame()

#Crear un DF nuevo sin duplicados
df3 = df2.drop_duplicates('description')
df3.count()
#Datos antes y después
df2.iloc[:,:9].describe()
df3.iloc[:,:9].describe()

#sns.lmplot(x="surface_total_in_m2", y="price_aprox_usd", data = df3, hue = 'place_name', height=10, fit_reg=False);
sns.lmplot(x="surface_total_in_m2", y="price_aprox_usd", data = df3, hue = 'place_name',fit_reg=False);
plt.savefig('grafico')
#Distribución de precio aproximado
sns.distplot(df3.loc[:,'price_aprox_usd'], kde=False);
plt.axvline(0, color="k", linestyle="--");

sns.distplot(df3.loc[:,'surface_total_in_m2'], kde=False);
plt.axvline(0, color="k", linestyle="--");

#Grilla de pares de precio y superficie totalg = sns.PairGrid(df3, vars=['price_aprox_usd', 'surface_total_in_m2'],hue='property_type')
g = sns.PairGrid(df3, vars=['price_aprox_usd', 'surface_total_in_m2'],hue='property_type')
g.map(plt.scatter, alpha=0.8)
g.add_legend();

#AGRUPACION
#Barrio y tipo de propiedad con describe
df2[['price_aprox_usd','surface_total_in_m2','property_type','place_name']].groupby(['place_name', 'property_type']).describe().head()
precioMedioBarrio = df2[['price_aprox_usd','place_name']].groupby(['place_name']).mean().sort_values('price_aprox_usd', ascending = False)
precioMedioBarrio.plot.bar(figsize=(30, 10))
plt.grid(True)
plt.axis('tight')
plt.xlabel('Barrio')
plt.ylabel('Precio Aprox. USD')
plt.title('Media de Precio por Barrio');
plt.savefig('Media de precio por barrio')

precioMedioTipoBarrio = df2[['property_type','place_name','price_aprox_usd']].groupby(['place_name','property_type']).mean().sort_values('place_name', ascending = False)
plt.figure(figsize=(10, 50))
precioMedioTipoBarrio.plot.bar(figsize=(30, 10))
plt.grid(True)
plt.axis('tight')
plt.xlabel('Barrio')
plt.ylabel('Precio Aprox. USD')
plt.title('Media de Precio por Barrio');
plt.savefig('Media de precio por barrio')



