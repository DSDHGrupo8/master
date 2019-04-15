# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 12:20:43 2019

@author: Adrian
"""

import pandas as pd

df=pd.read_csv("Properati.csv",encoding = 'utf8')
df2=df.query("state_name=='Capital Federal'")
cols=['country_name', 'currency', 'price_aprox_local_currency','operation','lat','lon','properati_url','description','title','place_with_parent_names','image_thumbnail','floor','rooms','geonames_id','price_usd_per_m2']
df2.drop(cols, axis=1, inplace=True)
df2.dropna(subset=['price', 'surface_total_in_m2'],inplace=True)
df2.to_csv("Properati_CABA.csv",encoding = 'utf8')
