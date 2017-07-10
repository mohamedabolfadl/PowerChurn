
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc

rf_col='navy'
dt_col='m'
lr_col='mediumpurple'
nb_col='darkorchid'
knn_col='plum'
svm_col='blue'
p = 0.11 # Fraction of volume who will churn
d = 0.2  # Discount

def plotROC(y_test, y_score_l,label,col):
    
    y_score = np.array(y_score_l[:,1])
#    y_score = np.array(y_score_l)

    y_test = np.array(y_test)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    
    lw = 1
    # , color=col
    plt.plot(fpr, tpr,
             lw=lw, label=label + ' (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    

plt.figure()

# Importing the dataset
dataset = pd.read_csv('ml_case_training_data_cleaned.csv')
X = dataset.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,33,34,35,36,37,38,39,40,41]].values
y = dataset.iloc[:, 32].values


# Filling NaNs with their mean
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:32])
X[:, 1:32] = imputer.transform(X[:, 1:32]) 

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
#from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
#y_conf_1 = classifier.decision_function(X_test)
y_conf_2 = classifier.predict_proba(X_test)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print('Random Forest accuracy = ' + str(100.0*accuracy_score(y_test, y_pred)))
# ------- ROC plot


X_norm = sc.inverse_transform(X)

plotROC(y_test, y_conf_2,"Random Forest ",rf_col)

    
    

# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
#y_conf_1 = classifier.decision_function(X_test)
y_conf_2 = classifier.predict_proba(X_test)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)
print('Decision tree accuracy = ' + str(100.0*accuracy_score(y_test, y_pred)))
plotROC(y_test, y_conf_2,"Decision tree",dt_col)




# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
#y_conf_1 = classifier.decision_function(X_test)
y_conf_2 = classifier.predict_proba(X_test)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)
print('Logistic regression accuracy = ' + str(100.0*accuracy_score(y_test, y_pred)))
plotROC(y_test, y_conf_2,"Logistic Regression ",lr_col)


# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
#y_conf_1 = classifier.decision_function(X_test)
y_conf_2 = classifier.predict_proba(X_test)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)
print('Naive Bayes accuracy = ' + str(100.0*accuracy_score(y_test, y_pred)))
plotROC(y_test, y_conf_2,"Naive Bayes",nb_col)


# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)
y_conf_2 = classifier.predict_proba(X_test)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)
print('K-nn accuracy = ' + str(100.0*accuracy_score(y_test, y_pred)))
plotROC(y_test, y_conf_2,"K-nn",knn_col)






# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0, probability = True)
classifier.fit(X_train, y_train)
# Getting confideence per user
y_conf_2 = classifier.predict_proba(X_test)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)
print('SVM accuracy = ' + str(100.0*accuracy_score(y_test, y_pred)))
plotROC(y_test, y_conf_2,"SVM",svm_col)


x_arr = np.linspace(0.0, 1.0, num=10)
y_thresh = ((1-p)/p)*(d/(1-d))*x_arr
plt.plot(x_arr,y_thresh,lw=1,color='r', linestyle='--',label='Margin '+'(p:'+str(100*p)+'%,d:'+str(100*d)+'%)')



plt.grid()

plt.show()
















