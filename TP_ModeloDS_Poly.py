import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

df=pd.read_csv("datasets\\properati_caballito.csv",encoding="utf8")
print("cant. registros antes de limpiar basura:", len(df))

regs_train=round((len(df)/100)*80,0)
df_train=df.loc[:regs_train,:]
df_test=df.loc[regs_train:len(df),:]

dummy_cols = [col for col in df if col.startswith('dummy_')]
distance_cols = [col for col in df if col.startswith('dist')]
cols=dummy_cols + distance_cols + ['lat','lon','surface_total_in_m2','expenses']

scaler = StandardScaler()
scaler.fit_transform(df[cols])

x=df[cols]
y=df["precio_m2_usd"]
X_train=df_train[cols]
y_train=df_train["precio_m2_usd"]
X_test=df_test[cols]
y_test=df_test["precio_m2_usd"]
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
X_test_poly=poly_features.fit_transform(X_test)
y_test_predict = poly_model.predict(X_test_poly)

# evaluating the model on training dataset
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_predicted))
r2_train = r2_score(y_train, y_train_predicted)


print("y_test", y_test)
print("y_test_predict", y_test_predict)

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
