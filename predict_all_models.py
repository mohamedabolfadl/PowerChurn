# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 23:20:05 2017
Script to write the result of the trained model
@author: Moh2
"""

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.metrics import f1_score, brier_score_loss, accuracy_score

test_pred_flag = True
  
# Importing the dataset
dataset = pd.read_csv('ml_case_training_data_cleaned.csv')
#Import the predict dataset
dataset_predict = pd.read_csv('ml_case_test_data_cleaned.csv')

# Indices of optimal features from most significant to least
slct_inds = [0,12,3,15,21,23,4,26,6,20,2,7,25,16,13,5,19,29,27,10,28,17,1,11,30,24,31,8,18,22,9,34,38,36,33,40,39,35,14,37]
N_feat = len(slct_inds)-10 # Select top N_feat features

slct_inds_pred = [1,13,4,16,22,24,5,27,7,21,3,8,26,17,14,6,20,30,28,11,29,18,2,12,31,25,32,9,19,23,10,34,38,36,33,40,41,39,35,15,37]
X = dataset.iloc[:, slct_inds].values
X = X[:,0:N_feat]
X_predict = dataset_predict.iloc[:, slct_inds_pred].values
X_predict = X_predict[:,0:N_feat]
y = dataset.iloc[:, 32].values


# Filling NaNs with their mean
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X)
X = imputer.transform(X) 
imputer = imputer.fit(X_predict)
X_predict = imputer.transform(X_predict) 


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_predict = sc.transform(X_predict)




# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 200, criterion = 'entropy', random_state=0)
classifier.fit(X_train, y_train)

# Optimal threshold 0.263
threshold = 0.263
# Test set 
y_conf = classifier.predict_proba(X_test)
y_conf_RF_test = y_conf[:,1]   
y_pred_RF_test = (y_conf_RF_test>threshold).astype(int)

# Print out stats about test    
print('F1 score on test set = '+str(f1_score(y_test, y_pred_RF_test, average='binary')))
print('Accuracy on test set = '+str(accuracy_score(y_test, y_pred_RF_test)))
print('Brier score on test set = '+str(brier_score_loss(y_test, y_pred_RF_test)))


# Task set
y_conf = classifier.predict_proba(X_predict)
y_conf_RF = y_conf[:,1]   
y_pred_RF = (y_conf_RF>threshold).astype(int) # Optimal threshold obtained from f1 score and sales maximization approaches
 





#Filling in results
dataset_output = pd.read_csv('ml_case_test_output_template.csv')
dataset_output.columns = ['','id','Churn_prediction','Churn_probability']
dataset_output["Churn_probability"] = np.array(y_conf_RF)
dataset_output["Churn_prediction"] = np.array(y_pred_RF)

dataset_output = dataset_output.sort_values(['Churn_probability'], ascending=False)

dataset_output.to_csv('ml_case_test_output_result.csv', index=False)  






























































