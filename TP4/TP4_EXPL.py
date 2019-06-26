import pandas as pd
import numpy as np

df = pd.read_csv('train.csv',nrows=40000)
#df.dropna(thresh=3)
df.to_csv("train_clean.csv")
print(df.describe)

df = pd.read_csv('test.csv',nrows=10000)
#df.dropna(thresh=3)
df.to_csv("test_clean.csv")
print(df.describe)




