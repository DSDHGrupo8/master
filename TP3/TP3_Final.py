import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
#import matplotlib.pyplot as plt
#from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict,cross_val_score
from sklearn import metrics

df = pd.read_csv('train.csv')

X = df.drop(['price_range'],axis=1)
y = df['price_range']

# Separación entre train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

#estandarizar
scaler = Normalizer().fit(X_train)
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)

#convertir a df
X_train_std_df = pd.DataFrame(X_train_std, index=X_train.index, columns=X_train.columns)
X_test_std_df = pd.DataFrame(X_test_std, index=X_test.index, columns=X_test.columns) 

modelo_NB = MultinomialNB()
#print("X_train_std:", X_train_std)
modelo_NB.fit(X_train_std, y_train)
predicciones=cross_val_predict(modelo_NB, X_train_std, y_train, cv=6)

scores = cross_val_score(modelo_NB, X_train_std, y_train, cv=6)
print("Mejor puntaje NB:",round(max(scores),4))

#prediccion = modelo_NB.predict(X_test_std)

# Primero calculamos el accuracy general del modelo
#print("accuracy NB:" , accuracy_score(y_test, prediccion))

#estandarizar
scaler2 = StandardScaler().fit(X_train)
X_train_std = scaler2.transform(X_train)
X_test_std = scaler2.transform(X_test)

#convertir a df
X_train_std_df = pd.DataFrame(X_train_std, index=X_train.index, columns=X_train.columns)
X_test_std_df = pd.DataFrame(X_test_std, index=X_test.index, columns=X_test.columns) 

mat = confusion_matrix(y_train, predicciones)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
#xticklabels=['negativo','positivo'], yticklabels=['negativo','positivo']
plt.xlabel('Etiquetas verdaderas')
plt.ylabel('Etiquetas predichas')

#modelos
svm_lin = SVC(kernel='sigmoid',degree=3,C=10, gamma='auto')
svm_rbf = SVC(kernel='rbf',C=10, gamma='auto')
svm_rbf.fit(X_train_std_df, y_train)
svm_lin.fit(X_train_std_df, y_train)

knn = KNeighborsClassifier()

best_k = int(round(math.sqrt(len(X)),0))
print("K selected for KNN:" , best_k)
knn = KNeighborsClassifier(n_neighbors=best_k)

knn.fit(X_train_std_df, y_train)
predicciones=cross_val_predict(knn, X_train_std, y_train, cv=6)

#score= knn.score(X_test_std_df, y_test)
scores = cross_val_score(knn, X_train_std, y_train, cv=6)
print("Mejor puntaje KNN (CV=6):",round(max(scores),4))


#predicciones
#prediccion_svm_lin = svm_lin.predict(X_test_std_df)
#prediccion_svm_rbf = svm_rbf.predict(X_test_std_df)

#accuracy
#print(accuracy_score(y_test, prediccion_svm_lin))
#print(accuracy_score(y_test, prediccion_svm_rbf))

print("SVM-sigmoid score:", svm_lin.score(X_test_std_df, y_test))
print("SVM-rbf score:", svm_rbf.score(X_test_std_df, y_test))
