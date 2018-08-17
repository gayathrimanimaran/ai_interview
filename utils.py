# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 12:27:45 2018

@author: Onkar
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder



def preprocess(train_data, train_labels, test_data, test_labels, label_encoder):
    
    #getting the required features
    train_features = train_data.loc[:,['country','category', 'main_category' , 'success_probability']].values
    test_features = test_data.loc[:,['country','category', 'main_category' , 'success_probability']].values
    
    #encoding each categorical column one at a time
    for i in range(len(train_features[0,:])):
        label_encode = LabelEncoder()
        label_encode.fit(train_features[:,i])
        train_features[:,i] = label_encode.transform(train_features[:,i])
        test_features[:,i] = label_encode.transform(test_features[:,i])
    
    #Addinf the deadline year column
    train_features = np.concatenate((train_features, np.reshape(train_data['deadline_year'].values, (-1,1))), axis = 1)
    test_features = np.concatenate((test_features, np.reshape(test_data['deadline_year'].values, (-1,1))), axis = 1)
    
    #converting all the encoded values to one hot vectors
    onehotencoder = OneHotEncoder(categorical_features = 'all')
    train_features = onehotencoder.fit_transform(train_features).toarray()
    test_features = onehotencoder.transform(test_features).toarray()
    
    #Adding the numerical column to one hot vectors
    train_features = np.concatenate((train_features, np.reshape(train_data['backers'].values, (-1,1))), axis = 1)
    test_features = np.concatenate((test_features, np.reshape(test_data['backers'].values, (-1,1))), axis = 1)
    
    #Encoding the target variable
    y_train = label_encoder.transform(train_labels)
    y_test = label_encoder.transform(test_labels)
    
    return train_features, y_train, test_features, y_test
    
    
    
def success_(goal, pledged):
    '''
    returns True if pledged amount is almost equal to the goal amount
    else returns False
    
    params:
        goal: goal amount
        pledged: pledged amount
    '''
    success = goal - pledged
    success_prob = (success < 5)
    return success_prob

def get_deadline_year(date_str):
    date = pd.datetime.strptime(date_str, '%Y-%m-%d')
    year = date.year
    return year
