# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np #any math 
import matplotlib.pyplot as plt
import pandas as pd

# variable explorer
dataset = pd.read_csv('C:\c\Rawls\online_courses\Machine Learning A-Z Template Folder\Part 1 - Data Preprocessing\Data.csv')
# creating a matrix with iloc for all the independent variables
X = dataset.iloc[:, :-1].values #first : means all rows, :-1 means all columns except the last one
# dependent variables 
y = dataset.iloc[:, 3].values
# in R we dont need to distinguish between dependent variables vactor and matrix of features 

#Taking care of missing data
from sklearn.preprocessing import Imputer #imputer class takes care of missing data
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)       #ctrl+i for details
imputer = imputer.fit(X[:,1:3]) #taking indexes 1 & 2, 3 is the upper bound
X[:, 1:3] = imputer.transform(X[:, 1:3]) #missing data is replaced by the means 

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder #for dummy variables
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:,0]) #getting encoded values of texts 
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
#purchased is dependent, so we dont need OneHotEncoder, because the machine knows 
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#splitting the dataset into Training set and Test set
from sklearn.cross_validation import train_test_split
#machine learning model will understand the correlation between independent & dependent variables 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) 

#Feature Scaling 
from sklearn.preprocessing import StandardScaler #stabdardization or normalization of data
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) #we dont need to fit this because sc_X is already fitted to the trainibg set
# can you scale dummy variables? it depends 
# do we need to scale dependent variables? not here, but for regression we will need to scale it 

















