from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
from scikitplot.metrics import plot_roc
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import scipy.stats as st
from catboost import CatBoostClassifier

df=pd.read_pickle("train_clean.pkl")

X = df.drop(columns=['HasDetections'])
y = df['HasDetections']

X_train, X_test, y_train, y_test = train_test_split(X,y)


#LGBM
model_lgb = lgb.LGBMClassifier()

params = {'num_leaves': st.randint(64,512),
         "n_estimators": st.randint(80,500),
         'min_data_in_leaf': st.randint(30,300),
         'max_depth': st.randint(6,10),
         'learning_rate': st.uniform(0.05, 0.1),
         'objective': ['binary'],
         'metric': ['auc'],
         'num_threads': [18],
         "boosting": ["gbdt"],
         "feature_fraction": [0.8],
         "bagging_freq": [5],
         "bagging_fraction": [0.8],
         "bagging_seed": [11],
         "lambda_l1": st.randint(0, 1),
         "lambda_l2": st.randint(0, 1),
         "random_state": [42]}

lgbm = RandomizedSearchCV(model_lgb, params, n_iter = 25, verbose=2, n_jobs=-1, cv=5)
lgbm.fit(X_train, y_train)

opt_lgbm = lgbm.best_estimator_
y_predicted_lgbm = lgbm.predict_proba(X_test)

lgb.plot_importance(lgbm.best_estimator_, max_num_features=5);
lgbm.best_estimator_.feature_importances_


#CATBOOST
model_cat = CatBoostClassifier()
model_cat.fit(X_train, y_train)
y_predicted_cat = model_cat.predict_proba(X_test)


lgbm_auc = roc_auc_score(y_test,y_predicted_lgbm[:,1])
print("El valor del AUC de LGBM es: ", lgbm_auc)
plot_roc(y_test,y_predicted_lgbm, plot_micro = False, plot_macro= False, title="ROC LGBM");

cat_auc = roc_auc_score(y_test,y_predicted_cat[:,1])
print("El valor del AUC de CatBoost es: ", cat_auc)
plot_roc(y_test,y_predicted_cat, plot_micro = False, plot_macro= False, title="ROC CATBOOST");