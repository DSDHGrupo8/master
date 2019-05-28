import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
import operator

#df=pd.read_csv("datasets\\properati_caballito.csv",encoding="utf8")
#print("cant. registros antes de limpiar basura:", len(df))

#regs_train=round((len(df)/100)*80,0)
#df_train=df.loc[:regs_train,:]
#df_test=df.loc[regs_train:len(df),:]
df_train=pd.read_csv("datasets\\properati_caballito_train.csv",encoding="utf8")
df_test=pd.read_csv("datasets\\properati_caballito_test.csv",encoding="utf8")

#FILTRAR OUTLIERS
qryFiltro="(price_aprox_usd >= 50000 and price_aprox_usd <= 250000)"
qryFiltro+=" and (surface_total_in_m2 >= 30 and surface_total_in_m2 <= 100)"
#qryFiltro+=" and (surface_total_in_m2 >= surface_covered_in_m2)"
qryFiltro+=" and (precio_m2_usd <= 3750 and precio_m2_usd >= 2000)"
qryFiltro+=" and (price_usd_per_m2 <= 3750 and price_usd_per_m2 >= 2000)"

df_train=df_train.query(qryFiltro)
df_test=df_test.query(qryFiltro)


#dummy_cols = [col for col in df_train if col.startswith('dummy_')]
dummy_cols=["dummy_property_type__apartment","dummy_property_type__house"]
#dummy_cols=[]
distance_cols = [col for col in df_train if col.startswith('dist')]
cols=dummy_cols + distance_cols + ['surface_total_in_m2','expenses']
#
scaler = Normalizer()
df_train[cols]=scaler.fit_transform(df_train[cols])
df_test[cols]=scaler.fit_transform(df_test[cols])

#x=df[cols]
#y=df["price_usd_per_m2"]
X_train=df_train[cols]
y_train=df_train["price_usd_per_m2"]
X_test=df_test[cols]
y_test=df_test["price_usd_per_m2"]
#print("X:",X)
#print("y:",y)

degree=2
poly_features = PolynomialFeatures(degree=degree)

# transforms the existing features to higher degree features.
X_train_poly = poly_features.fit_transform(X_train)

# fit the transformed features to Linear Regression
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

# predicting on training data-set
y_train_predicted = poly_model.predict(X_train_poly)

# predicting on test data-set
X_test_poly=poly_features.transform(X_test)
y_test_predict = poly_model.predict(X_test_poly)

# evaluating the model on training dataset
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_predicted))
r2_train = r2_score(y_train, y_train_predicted)


print("y_test.max:", max(y_test))
print("y_test_predict.max:", max(y_test_predict))

print("y_test.avg:", np.average(y_test))
print("y_test_predict.avg:", np.average(y_test_predict))

print("y_test.std:", np.std(y_test))
print("y_test_predict.std:", np.std(y_test_predict))


# evaluating the model on test dataset
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_predict))
r2_test = r2_score(y_test, y_test_predict)

print("The model performance for the training set")
print("-------------------------------------------")
print("RMSE of training set is {}".format(rmse_train))
print("R2 score of training set is {}".format(r2_train))

print("\n")

print("The model performance for the test set")
print("-------------------------------------------")
print("RMSE of test set is {}".format(rmse_test))
print("R2 score of test set is {}".format(r2_test))

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

print("y_test.min:",min(y_test))
