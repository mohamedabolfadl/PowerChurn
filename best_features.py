# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 09:03:51 2017
Function to investigate the correlation between features and output
Generates 'Feature ranking.csv' which ranks features in terms of relevance to churn
@author: Moh2
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('ml_case_training_data_cleaned.csv')
dataset_all_pos = dataset
minVs = dataset_all_pos.min(axis=0)
colns = list(dataset)
for i in range(len(colns)):
    if minVs[i]<0:
        dataset_all_pos[colns[i]] = dataset_all_pos[colns[i]] - minVs[i]

X = dataset.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,33,34,35,36,37,38,39,40,41]].values
y = dataset.iloc[:, 32].values

X_p = dataset_all_pos.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,33,34,35,36,37,38,39,40,41]].values
y_p = dataset_all_pos.iloc[:, 32].values


# Filling NaNs with their mean
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:32])
X[:, 1:32] = imputer.transform(X[:, 1:32]) 

imputer = imputer.fit(X_p[:, 1:32])
X_p[:, 1:32] = imputer.transform(X_p[:, 1:32]) 


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_norm = sc.fit_transform(X)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
#y_conf_1 = classifier.decision_function(X_test)
y_conf_2 = classifier.predict_proba(X_test)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)




# Recursive Feature selection
from sklearn.feature_selection import RFE
rfe = RFE(classifier, 1)
fit = rfe.fit(X_norm, y)
ranks1 = fit.ranking_
sort_ind = sorted(range(len(ranks1)), key=lambda k: ranks1[k])
feat_sort = []
score_sort=[]
for i in range(len(sort_ind)):
    feat_sort.append(colns[sort_ind[i]])

df = pd.DataFrame(feat_sort, columns=["Feature_Rank"])
df.to_csv('Feature ranking.csv', index=False)


# Iterate with cross validation to pick up the best features
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
rfecv = RFECV(estimator=classifier, step=1, cv=StratifiedKFold(2),
             scoring='accuracy')
rfecv.fit(X_train, y_train)
print("Optimal number of features : %d" % rfecv.n_features_)
print(rfecv.support_ )
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (accuracy)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()
plt.grid()
classifier_opt = rfecv.estimator_
classifier_opt.fit(X_train, y_train)
y_conf_2 = classifier_opt.predict_proba(X_test)
# Predicting the Test set results
y_pred = classifier_opt.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# Proves to be a weak indicator of feature power
if False:
    # K Best based on chi2 or f_classif
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2, f_classif
    test = SelectKBest(score_func=chi2, k=4)
    fit = test.fit(X_p, y)
    np.set_printoptions(precision=3)
    print(fit.scores_)
    features = fit.transform(X_norm)
    print(features[0:5,:])
    scores = fit.scores_
    sort_ind = sorted(range(len(scores)), key=lambda k: scores[k])
    sort_ind.reverse()
    feat_sort = []
    score_sort=[]
    colns.remove('churn')
    for i in range(len(sort_ind)):
        feat_sort.append(colns[sort_ind[i]])
        score_sort.append(scores[sort_ind[i]])
    #Bar chart
    N = 5
    fig, ax = plt.subplots()
    ind = np.arange(len(feat_sort[0:N]))
    width = 0.2 
    rects1 = ax.bar(ind, (score_sort[0:N])/(sum(score_sort)), width, color='r')
    ax.set_ylabel('Score [%]')
    ax.set_title('Best Features')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(( feat_sort[0], feat_sort[1], feat_sort[2], feat_sort[3], feat_sort[4]))
    ax.grid()
    plt.savefig('KBest.png')
    plt.savefig('Kbest.pdf')
    


# Available only in Random forest and decision tree
if False:
    # Feature importance for tree classifiers
    print classifier.feature_importances_
  


