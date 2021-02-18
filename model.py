# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from datetime import date

"""Read csv"""
dataset = pd.read_csv('data2.csv')
dataset.head()

"""Select columns from the dataframe needed for the model"""
new_ds = dataset[['Individual_Id', 'DOB', 'Gender','Pregnant', 'Annual_Income','Household_members','Citizenship', 'Homeless', 'Eligible']]
new_ds.head()
new_ds.dtypes

"""Calculate age"""
today = date.today()

new_ds['current_date']=today
new_ds['age'] = ((pd.to_datetime(new_ds.current_date) - pd.to_datetime(new_ds.DOB)).dt.days)/365
new_ds[['current_date','DOB','age']].head()

"""One hot encoding"""
new_ds['male_flag']= pd.get_dummies(new_ds["Gender"],prefix='gender',drop_first=True)
new_ds['pregnant_flag'] = pd.get_dummies(new_ds["Pregnant"],prefix='pregnant',drop_first=True)
new_ds[['can_flag','mex_flag','oth_flag','us_flag']]=pd.get_dummies(new_ds["Citizenship"],prefix='citizen')
new_ds['homeless_flag'] =pd.get_dummies(new_ds["Homeless"],prefix='homeless',drop_first=True)
new_ds['eligible_flag']= pd.get_dummies(new_ds["Eligible"],prefix='eligible',drop_first=True)

new_ds.head()
new_ds.info()

X = new_ds[['Annual_Income','Household_members','age','male_flag','pregnant_flag','us_flag','homeless_flag']]
#X = new_ds[['Annual_Income','Household_members','age','male_flag','pregnant_flag','can_flag','mex_flag','oth_flag','us_flag','homeless_flag']]
#X = dataset.iloc[:, :4]

#y = dataset.iloc[:, -1]
y= new_ds['eligible_flag']

from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression()

regressor.fit(X,y)

pickle.dump(regressor, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))
#Z = model.predict_proba([[1, 1, 1, 1]])
Z = model.predict_proba([[20000, 2, 60, 1, 0,1,1]])[0,0]
print(Z)

