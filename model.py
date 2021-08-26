# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 19:17:03 2021

@author: Ramesh VG
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 14:35:15 2021

@author: Ramesh VG
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv('startups_modified.csv')

#dataset['TV'].fillna(dataset['TV'].mean(), inplace=True)
#dataset['Radio'].fillna(dataset['Radio'].mean(), inplace=True)
#pd.get_dummies(dataset.state, drop_first=True).head()
#fstate = pd.get_dummies(dataset['state'], drop_first=True)
#fstate.head()
#dataset= dataset.drop(['State'], axis = 1, inplace = True)

X = dataset.iloc[:, :3]

#Converting words to integer values
"""def convert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
    return word_dict[word]"""

# X['experience'] = X['experience'].apply(lambda x : convert_to_int(x))

y = dataset.iloc[:, -1]

#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X, y)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[6, 16, 9]]))