# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 23:54:27 2019

@author: Adrian
"""
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

print("pandas:" , pd.__version__)

df=pd.read_csv("train_clean2.csv",encoding="utf-8")


#print(df.var().sort_values(ascending=False))
#print(df.isnull().sum().sort_values(ascending=False))


#apply SelectKBest class to extract top x best features
bestfeatures = SelectKBest(score_func=chi2, k=40)

y=df["HasDetections"]
X=df.drop("HasDetections",axis=1)

fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.sort_values(ascending=True,by="Score"))  #print all features
df1=featureScores[featureScores["Score"] < 0.5]
df2=featureScores[featureScores["Score"].isnull()]
print("df1:",df1)
print("df2:",df2)
#print("featureScores:" , featureScores)
df.drop(columns=df1["Specs"],inplace=True)
df.drop(columns=df2["Specs"],inplace=True)
df.to_csv("train_clean2.csv",encoding="utf-8")

