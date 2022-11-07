# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 15:36:04 2022

@author: Gopinath
"""

from keras.models import Sequential
import pandas as pd
from keras.layers import Dense
import numpy
seed = 7
numpy.random.seed(seed)
# load pima indians dataset
# split into input (X) and output (Y) variables
gas = pd.read_csv("gas_turbines.csv")
X = gas.iloc[:,0:10]
Y = gas.iloc[:,10]
X
gas.info()
gas.describe
gas.shape
gas.T

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('darkgrid')
sns.pairplot(gas)
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt
#correlation matrix
corrmat = gas.corr()
f, ax = plt.subplots(figsize=(20, 10))
sns.heatmap(corrmat, vmax=.8, square=True, annot=True);

# correlation with TEY

data2 = X.copy()

correlations = data2.corrwith(gas["TEY"])
correlations = correlations[correlations!=1]
positive_correlations = correlations[correlations >0].sort_values(ascending = False)
negative_correlations =correlations[correlations<0].sort_values(ascending = False)

correlations.plot.bar(figsize = (18, 10),fontsize = 15,color = 'b',rot = 90, grid = True)
plt.title('Correlation with Turbine energy yield \n',horizontalalignment="center", fontstyle = "normal",fontsize = "22", fontfamily = "sans-serif")


def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)
gas.info()



X_norm = norm_func(X)
model = Sequential()
model.add(Dense(12, input_dim=10, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_norm, Y, epochs=150, batch_size=10)

accuracy = model.evaluate(X, Y)
print('Accuracy:  ' %(accuracy*100))








