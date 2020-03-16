# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 22:29:08 2020

@author: admin
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 14:02:52 2020

@author: diggee

script created and compiled in spyder
"""

#%% importing libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import string
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
import time

#%% reading data

def get_data():
    full_train_data = pd.read_csv('train.csv', index_col = 'id')
    full_test_data = pd.read_csv('test.csv', index_col = 'id')
    return full_train_data, full_test_data

#%% preprocessing data

def clean_data(full_train_data, full_test_data):
    # dropping keyword and location columns
    cols_to_drop = ['keyword','location']
    full_train_data.drop(cols_to_drop, axis = 1, inplace = True)
    full_test_data.drop(cols_to_drop, axis = 1, inplace = True)

    # there are no null values in either the train set or the test set.
    print('Total null values in the train set - '+str(full_train_data.isnull().sum().sum()))
    print('Total null values in the test set - '+str(full_test_data.isnull().sum().sum()))

    full_train_data['text'] = full_train_data['text'].apply(lambda x: x.lower()) # convert everything to lower case
    full_test_data['text'] = full_test_data['text'].apply(lambda x: x.lower()) # convert everything to lower case
    
    def tweet_clean(tweet):
        for i in range(len(full_train_data['text'])):
            # remove website links
            tweet = re.sub('www.|https://|http://|.com|t.co/','',tweet)    
            # remove all punctuation 
            tweet = ''.join([j for j in tweet if j not in string.punctuation])    
            # remove all digits
            tweet = ''.join([j for j in tweet if j not in string.digits])    
            # remove stopwords
            tweet = ' '.join([j for j in tweet.split() if j not in stopwords.words('english')])    
            # remove non ASCII characters
            tweet = ''.join([j for j in tweet if ord(j) < 128])
        return tweet
    
    full_train_data['text'] = full_train_data['text'].apply(tweet_clean)
    full_test_data['text'] = full_test_data['text'].apply(tweet_clean)
    full_train_data['text'] = full_train_data['text'].apply(lambda x: x.lstrip())   # remove all leading spaces
    full_train_data['text'] = full_train_data['text'].apply(lambda x: x.rstrip())   # remove all trailing spaces
    full_test_data['text'] = full_test_data['text'].apply(lambda x: x.lstrip())   # remove all leading spaces
    full_test_data['text'] = full_test_data['text'].apply(lambda x: x.rstrip())   # remove all trailing spaces

    CV = CountVectorizer()
    X = CV.fit_transform(full_train_data['text'])
    y = full_train_data['target']
    X_valid = CV.transform(full_test_data['text'])
    return X, y, X_valid

#%% data scaling
    
def scaled_data(X, X_valid):
    from sklearn.preprocessing import StandardScaler
    scaler_X = StandardScaler()
    X = scaler_X.fit_transform(X)
    X_valid = scaler_X.transform(X_valid)
    return X, X_valid, scaler_X  

#%% regressor functions
    
def regressor_fn_optimised(X, y, X_valid, choice):      
    from bayes_opt import BayesianOptimization
    
    if choice == 1:    
        from sklearn.linear_model import RidgeClassifier        
        def regressor_fn(alpha):            
            regressor = RidgeClassifier(alpha = alpha)        
            cval = cross_val_score(regressor, X, y, scoring = 'balanced_accuracy', cv = 5)
            return cval.mean()
        pbounds = {'alpha': (0, 0)}
        
    elif choice == 2:    
        from sklearn.neighbors import KNeighborsClassifier        
        def regressor_fn(n_neighbors):     
            n_neighbors = int(n_neighbors)
            regressor = KNeighborsClassifier(n_neighbors = n_neighbors)        
            cval = cross_val_score(regressor, X, y, scoring = 'balanced_accuracy', cv = 5)
            return cval.mean()
        pbounds = {'n_neighbors': (2,10)}
        
    elif choice == 3:    
        from sklearn.ensemble import RandomForestClassifier        
        def regressor_fn(n_estimators, max_depth):     
            max_depth, n_estimators = int(max_depth), int(n_estimators)
            regressor = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth)        
            cval = cross_val_score(regressor, X, y, scoring = 'balanced_accuracy', cv = 5)
            return cval.mean()
        pbounds = {'n_estimators': (10, 500), 'max_depth': (2,20)}
        
    elif choice == 4: 
        X, X_valid, scaler_X = scaled_data(X, X_valid)
        from sklearn.svm import SVC        
        def regressor_fn(C, gamma):            
            regressor = SVC(C = C, kernel = 'rbf', gamma = gamma)        
            cval = cross_val_score(regressor, X, y, scoring = 'balanced_accuracy', cv = 5)
            return cval.mean()
        pbounds = {'C': (0.1, 100), 'gamma': (0.001, 100)}
        
    elif choice == 5:
        from lightgbm.sklearn import LGBMClassifier
        def regressor_fn(learning_rate, max_depth, n_estimators):            
            max_depth, n_estimators = int(max_depth), int(n_estimators)
            regressor = LGBMClassifier(learning_rate = learning_rate, max_depth = max_depth, n_estimators = n_estimators)        
            cval = cross_val_score(regressor, X, y, scoring = 'balanced_accuracy', cv = 5)
            return cval.mean()
        pbounds = {'learning_rate': (0.01, 1), 'max_depth': (2,40), 'n_estimators': (10, 500)}        
        
    else:
        from xgboost import XGBClassifier
        def regressor_fn(learning_rate, max_depth, n_estimators):            
            max_depth, n_estimators = int(max_depth), int(n_estimators)
            regressor = XGBClassifier(learning_rate = learning_rate, max_depth = max_depth, n_estimators = n_estimators)        
            cval = cross_val_score(regressor, X, y, scoring = 'balanced_accuracy', cv = 3)
            return cval.mean()
        pbounds = {'learning_rate': (0.01, 1), 'max_depth': (2,50), 'n_estimators': (10, 500)}
    
    optimizer = BayesianOptimization(regressor_fn, pbounds, verbose = 2)
    optimizer.probe(params = {'alpha':0}, lazy = True)
    optimizer.maximize(init_points = 5, n_iter = 30)    
    # change next line in accordance with choice of regressor made
    # y_valid_pred = RandomForestClassifier(max_depth = int(optimizer.max['params']['max_depth']), n_estimators = int(optimizer.max['params']['max_depth'])).fit(X, y).predict(X_valid)
    y_valid_pred = RidgeClassifier().fit(X, y).predict(X_valid)
    
    return y_valid_pred, optimizer.max

#%% main

if __name__ == '__main__':
    full_train_data, full_test_data = get_data()
    t1 = time.time()
    X, y, X_valid = clean_data(full_train_data, full_test_data)
    t2 = time.time()
    print(t2-t1)
    # y_valid_pred, optimal_params = regressor_fn_optimised(X, y, X_valid, choice = 1)
    # df = pd.DataFrame({'Id':full_test_data.index, 'Target':y_valid_pred})
    # df.to_csv('prediction.csv', index = False)
