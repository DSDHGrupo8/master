import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistics import mode 
from matplotlib import cm as cm
from sklearn import svm
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
                    
        #plt.rcParams.update({'font.size': 12})
        fig = plt.figure()
        fig.set_figheight(10)
        fig.set_figwidth(40)
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

    def __init__(self):
        self.df=pd.read_csv("properatti.csv", encoding = 'utf8')
        self.df_m2=pd.read_csv("precioxm2_pais.csv", encoding = 'utf8')
        print("DataSet registros:", len(self.df))
        print("DataSet Lookup Precio x m2:", len(self.df_m2))
        
        #FIJAR SCOPE EN CABA
        self.df=self.df.loc[self.df['state_name'] == 'Capital Federal']
            
        #CORRECCION DE PRECIOS y MONEDA
        df1 = self.df[self.df['price'].isnull()]
        aux = df1['title'].str.extract(r'(U?u?\$[SDsd]?)\s?(\d+)\.?(\d*)\.?(\d*)')
        aux=aux.dropna()
        aux[0]=aux[0].str.replace('^\$$', 'ARS', regex=True)
        aux[0]=aux[0].str.replace('^[^A].*$', 'USD', regex=True)
        aux['currency']=aux[0]
        aux['price']=aux[1]+aux[2]+aux[3]
        aux['price']=aux['price'].astype('float64')
        aux=aux[['currency','price']]
        self.df.loc[self.df['price'].isnull(),'price'] = aux['price']
        self.df.loc[self.df['currency'].isnull(),'currency'] = aux['currency']
        df1 = self.df[self.df['price'].isnull()]
        aux = df1['description'].str.extract(r'(U?u?\$[SDsd]?)\s?(\d+)\.?(\d*)\.?(\d*)')
        aux=aux.dropna()
        aux[0]=aux[0].str.replace('^\$$', 'ARS', regex=True)
        aux[0]=aux[0].str.replace('^[^A].*$', 'USD', regex=True)
        aux['currency']=aux[0]
        aux['price']=aux[1]+aux[2]+aux[3]
        aux['price']=aux['price'].astype('float64')
        aux=aux[['currency','price']]
        self.df.loc[self.df['price'].isnull(),'price'] = aux['price']
        self.df.loc[self.df['currency'].isnull(),'currency'] = aux['currency']
        
        #CORRECCION DE M2
             
        #JOINEAR CON TABLA LOOKUP
        #values = {'surface_total_in_m2': 0}
        #self.df=self.df.fillna(value=values)
        #aux_caba=self.df.merge(df_m2, how='left', on='place_name')
        #values = {'m2_Desde': 0, 'm2_Hasta': 999999}
        #aux_caba=aux_caba.fillna(value=values)
        #aux_caba=aux_caba[(aux_caba['surface_total_in_m2']>=aux_caba['m2_Desde']) & (aux_caba['surface_total_in_m2']<=aux_caba['m2_Hasta'])]
        #aux_caba=aux_caba.drop(['state_name_y', 'm2_Desde', 'm2_Hasta'], axis=1)
        
        
        #DROPEAMOS VARIABLES NO INTERESANTES
        #dropeo columnas que no son de interés
        cols=['country_name', 'currency','price', 'price_aprox_local_currency','operation','lat','lon','properati_url', 'expenses','place_with_parent_names','image_thumbnail','floor','rooms','geonames_id','price_usd_per_m2']
        self.df.drop(cols, axis=1, inplace=True)
        self.df.dropna(subset=['surface_total_in_m2','price_aprox_usd'],inplace=True)

        self.df2=self.df.dropna()
        #print(self.df.columns)
        
        #ARREGLAR DATOS CORREGIBLES
        #Arreglar precio x m2 en dólares
        self.df['price_aprox_usd']=np.round(self.df['price_aprox_usd'],0).fillna(0).astype(np.int64)
        self.df['surface_total_in_m2']=np.round(self.df['surface_total_in_m2'],0).fillna(0).astype(np.int64)
        self.df['precio_m2_usd']=np.round(self.df['price_aprox_usd'] / self.df['surface_total_in_m2'],0)
        
        #auxdf=self.df.query('precio_m2_usd>10000').filter(items=['precio_m2_usd','price_aprox_usd','surface_total_in_m2'])
        #print(auxdf)
        #print(self.df['precio_m2_usd'])
        
        #Arreglar latitud y longitud por medio de la columna lat/lon
        latlongdf=self.df['lat-lon'].str.split(",",expand=True)
        self.df['lat']=latlongdf.loc[:,0]
        self.df['lon']=latlongdf.loc[:,1]
        self.df.drop('lat-lon',axis=1,inplace=True)
        self.df.drop('price_per_m2',axis=1,inplace=True)
        
        
        qryFiltro='(price_aprox_usd >= 10000 and price_aprox_usd <= 1000000) '
        qryFiltro+='and (surface_total_in_m2 >= 20 and surface_total_in_m2 <= 1000) '
        qryFiltro+='and (surface_total_in_m2 >= surface_covered_in_m2)'
        
        #print("Query de filtrado:" , qryFiltro)
        self.df=self.df.query(qryFiltro)
        
        self.df.to_csv("Properati_fixed.csv",encoding='utf-8')
                
        print("All done!")
        

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

df2 = pd.read_csv('Properati_fixed.csv')
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
print(df2.loc[:,'state_name'].unique())
print(df2.loc[:,'place_name'].unique())
print(df2.loc[:,'property_type'].unique())
#Duplicados de description  
df2.duplicated('description').value_counts().to_frame()

#Duplicados de lat, lon y description 
df2.duplicated(subset=['lat','lon','description']).value_counts().to_frame()

#¿Qué pasaba en el original?
df_original = pd.read_csv('properatti.csv')
df_original.duplicated(subset=['lat','lon','description']).value_counts().to_frame()

#Crear un DF nuevo sin duplicados
df3 = df2.drop_duplicates('description')
df3.count()
#Datos antes y después
df2.iloc[:,:9].describe()
df3.iloc[:,:9].describe()

sns.lmplot(x="surface_total_in_m2", y="price_aprox_usd", data = df3, hue = 'place_name', height=10, fit_reg=False);
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



