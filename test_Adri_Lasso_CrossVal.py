import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score
import math
#X, y = make_regression(noise=4, random_state=0)

df=pd.read_csv("dataset.csv",encoding="utf8")
print("cant. registros antes de limpiar basura:", len(df))
#LIMPIAR BASURA
df=df[pd.to_numeric(df['dummy_property_type__store'], errors='coerce').notnull()]
df=df[pd.to_numeric(df['dummy_property_type__apartment'], errors='coerce').notnull()]
df=df[pd.to_numeric(df['dummy_property_type__house'], errors='coerce').notnull()]
df=df[pd.to_numeric(df['lat'], errors='coerce').notnull()]
df=df[pd.to_numeric(df['lon'], errors='coerce').notnull()]
df=df[pd.to_numeric(df['distSubte'], errors='coerce').notnull()]

regs_train=round((len(df)/100)*80,0)
df_train=df.loc[:regs_train,:]
df_test=df.loc[regs_train:len(df),:]

promedio_exp=round(df['expenses'].mean(),2)
print("promedio expensas:", promedio_exp)
df['expenses']=df['expenses'].fillna(promedio_exp)

print("cant. registros después de limpieza:", len(df))
dummy_cols = [col for col in df if col.startswith('dummy_')]
#print("dummy columns:" , dummy_cols)
#cols=dummy_cols + ['surface_total_in_m2','price_aprox_usd']
#cols=dummy_cols + ['surface_total_in_m2','price_usd_per_m2']
#cols=dummy_cols
cols=dummy_cols + ['lat','lon','surface_total_in_m2','distSubte','price_aprox_usd','expenses']
#cols=dummy_cols + ['lat','lon','surface_total_in_m2','distSubte']
X_train=df_train[cols]
y_train=df_train["precio_m2_usd"]
X_test=df_test[cols]
y_test=df_test["precio_m2_usd"]
#print("X:",X)
#print("y:",y)

scaler = StandardScaler()
scaler.fit_transform(df[cols])

lasso = Lasso()
print("cross_val_score para Lasso común:",round(max(cross_val_score(lasso, X_train, y_train, cv=5)),2))
lasso.fit(X_train,y_train)
print("Lasso común r2:",round(lasso.score(X_train,y_train),2))
reg = LassoCV(cv=5,n_alphas=100, random_state=0).fit(X_train, y_train)
print("LassoCV r2:" ,round(reg.score(X_train, y_train),2)) 
#print("predict:", reg.predict(X[:1,]))