# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 13:47:38 2019

@author: Adrian
"""

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10,10))
fig2 = plt.figure(figsize=(10,10))

df = pd.read_csv('datasets/train.csv')
df.head()

df.hist("ram")
df.hist("int_memory")
df.hist("clock_speed")
df.hist("price_range")
df.hist("battery_power")
df.hist("n_cores")
df.hist("blue")

ax = Axes3D(fig) # Method 1
x=df["ram"]
y=df["int_memory"]
z=df["price_range"]
ax.scatter(x, y,z, c=x, marker='.')
ax.set_xlabel('RAM')
ax.set_ylabel('Almacenamiento interno')
ax.set_zlabel('Rango de precio')
plt.savefig('3dscatter_TP3_1')
plt.show()

