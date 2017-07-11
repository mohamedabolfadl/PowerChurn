# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 23:20:05 2017

@author: Moh2
"""

# Importing the libraries
import numpy as np
import pandas as pd

test_pred_flag = True
  
# Importing the dataset
dataset = pd.read_csv('ml_case_training_data_cleaned.csv')
#Import the predict dataset
dataset_predict = pd.read_csv('ml_case_test_data_cleaned.csv')

# Number of features
N_feat = 30 # Select top N_feat features
# Indices of optimal features from most significant to least
slct_inds = [0,12,3,15,21,23,4,26,6,20,2,7,25,16,13,5,19,29,27,10,28,17,1,11,30,24,31,8,18,22,9,34,38,36,33,40,41,39,35,14,37]
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
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy')
classifier.fit(X_train, y_train)
if test_pred_flag:
    y_conf = classifier.predict_proba(X_predict)
    y_pred_RF = classifier.predict(X_predict)
else:
    y_conf = classifier.predict_proba(X_test)
    y_pred_RF = classifier.predict(X_test)
y_conf_RF = y_conf[:,1]    
    

# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
if test_pred_flag:
    y_conf = classifier.predict_proba(X_predict)
    y_pred_DT = classifier.predict(X_predict)
else:
    y_conf = classifier.predict_proba(X_test)
    y_pred_DT = classifier.predict(X_test)
y_conf_DT = y_conf[:,1]    

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
if test_pred_flag:
    y_conf = classifier.predict_proba(X_predict)
    y_pred_LR = classifier.predict(X_predict)
else:
    y_conf = classifier.predict_proba(X_test)
    y_pred_LR = classifier.predict(X_test)
y_conf_LR = y_conf[:,1]    


# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
if test_pred_flag:
    y_conf = classifier.predict_proba(X_predict)
    y_pred_NB = classifier.predict(X_predict)
else:
    y_conf = classifier.predict_proba(X_test)
    y_pred_NB = classifier.predict(X_test)
y_conf_NB = y_conf[:,1]    


# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)
if test_pred_flag:
    y_conf = classifier.predict_proba(X_predict)
    y_pred_KN = classifier.predict(X_predict)
else:
    y_conf = classifier.predict_proba(X_test)
    y_pred_KN = classifier.predict(X_test)
    
y_conf_KN = y_conf[:,1]    



# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0, probability = True)
classifier.fit(X_train, y_train)
if test_pred_flag:
    y_conf = classifier.predict_proba(X_predict)
    y_pred_SV = classifier.predict(X_predict)
else:
    y_conf = classifier.predict_proba(X_test)
    y_pred_SV = classifier.predict(X_test)
    
y_conf_SV = y_conf[:,1]    

results_conf = pd.DataFrame({'SVM':np.array(y_conf_SV),'RF':np.array(y_conf_RF),'KN':np.array(y_conf_KN),'LR':np.array(y_conf_LR),'DT':np.array(y_conf_DT)})
results_decision = pd.DataFrame({'SVM':np.array(y_pred_SV),'RF':np.array(y_pred_RF),'KN':np.array(y_pred_KN),'LR':np.array(y_pred_LR),'DT':np.array(y_pred_DT)})




# Fitting Neural Network SVM to the Training set
#import keras
#from keras.models import Sequential
#from keras.layers import Dense
#classifier = Sequential()
#classifier.add(Dense(units = int(np.floor(2*(N_feat-1)/3)), kernel_initializer = 'uniform', activation = 'relu', input_dim = N_feat))
#classifier.add(Dense(units = int(np.floor(2*(N_feat-1)/3)), kernel_initializer = 'uniform', activation = 'relu'))
#classifier.add(Dense(units = int(np.floor((N_feat-1)/3)), kernel_initializer = 'uniform', activation = 'relu'))
#classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
#classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#classifier.fit(X_train, y_train, batch_size = 5, epochs = 100)
#y_conf_NN = classifier.predict(X_test)
#y_pred_NN = (y_conf_NN > 0.5)


#Filling in results
dataset_output = pd.read_csv('ml_case_test_output_template.csv')
dataset_output.columns = ['','id','Churn_prediction','Churn_probability']
dataset_output["Churn_probability"] = np.array(y_conf_RF)
dataset_output["Churn_prediction"] = np.array(y_pred_RF)



dataset_output.to_csv('ml_case_test_output_result.csv', index=False)  




































