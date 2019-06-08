import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

#X, y = make_regression(noise=4, random_state=0)

df_train=pd.read_csv("datasets\\properati_caballito_train.csv",encoding="utf8")
df_test=pd.read_csv("datasets\\properati_caballito_test.csv",encoding="utf8")
#print("cant. registros antes de limpiar basura:", len(df))

scaler = Normalizer()
df_train[cols]=scaler.fit_transform(df_train[cols])

regs_train=round((len(df_train)/100)*80,0)
df_train=df.loc[:regs_train,:]
df_test=df.loc[regs_train:len(df),:]

print("cant. registros después de limpieza:", len(df))
dummy_cols = [col for col in df if col.startswith('dummy_')]
#print("dummy columns:" , dummy_cols)
#cols=dummy_cols + ['surface_total_in_m2','price_aprox_usd']
#cols=dummy_cols + ['surface_total_in_m2','price_usd_per_m2']
#cols=dummy_cols
distance_cols = [col for col in df if col.startswith('dist')]
cols=dummy_cols + distance_cols + ['surface_total_in_m2','expenses']
#cols=dummy_cols + distance_cols + ['lat','lon','surface_total_in_m2','expenses','price_aprox_usd']
#cols=dummy_cols + ['lat','lon','surface_total_in_m2','distSubte']

X_train=df_train[cols]
y_train=df_train["precio_m2_usd"]
X_test=df_test[cols]
y_test=df_test["precio_m2_usd"]
#print("X:",X)
#print("y:",y)

lasso = Lasso()
print("cross_val_score para Lasso común:",round(max(cross_val_score(lasso, X_train, y_train, cv=5)),2))
lasso.fit(X_train,y_train)
y_test=lasso.predict(X_test)
print("Lasso común r2:",round(lasso.score(X_train,y_train),2))
kf = KFold(n_splits=5, shuffle=True, random_state=12)
reg = LassoCV(n_alphas=100, random_state=0,cv=kf, normalize=False).fit(X_train, y_train)
y_test=reg.predict(X_test)
print("LassoCV r2:" ,round(reg.score(X_train, y_train),2)) 

x=X_train["surface_total_in_m2"]
y=y_train
plt.title("Trained")
plt.scatter(x, y, s=20)
# sort the values of x before line plot
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(x,y_train_predicted), key=sort_axis)
x, y_poly_pred = zip(*sorted_zip)
plt.plot(x, y_poly_pred, color='m')
plt.show()

x=X_test["surface_total_in_m2"]
y=y_test
plt.title("Predicted")
plt.scatter(x, y, s=20)
# sort the values of x before line plot
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(x,y_test_predict), key=sort_axis)
x, y_poly_pred = zip(*sorted_zip)
plt.plot(x, y_poly_pred, color='m')
plt.show()


#print("predict:", reg.predict(X[:1,]))

#polynomial_features= PolynomialFeatures(degree=2)
#x_poly = polynomial_features.fit_transform(X_train)
#
#model = LinearRegression()
#model.fit(x_poly, y_train)
#y_poly_pred = model.predict(x_poly)
#
#rmse = np.sqrt(mean_squared_error(y,y_poly_pred))
#r2 = r2_score(y,y_poly_pred)
#print("Poly rmse:",rmse)
#print("Poly r2:",r2)
