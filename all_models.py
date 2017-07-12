



# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import f1_score, precision_recall_curve, brier_score_loss

rf_col='g'
dt_col='b'
lr_col='k'
nb_col='darkorchid'
knn_col='plum'
svm_col='orange'
nn_col = 'r'
p = 0.11 # Fraction of volume who will churn
d = 0.2  # Discount
p_a_m = 5 # relationship between dicount provided and probability of accepting it  
threshold_vec = np.linspace(0.01,0.99,70)



def getFscores(y_prob, y_true):
    
    f_score_vec = []
    for i in range(len(threshold_vec)):
        thresh=threshold_vec[i]
        y_pred = (y_prob > thresh)
        f_score_vec.append(f1_score(y_true, y_pred, average='binary'))
    return f_score_vec        
        
        
def acceptProb(d,acceptance_kernel):
    
    if acceptance_kernel == 'lin':    
        return p_a_m*d
    else:
        return -(1/ (0.2*0.2))*(d-0.2)**2+1

def plotRET(y_test, y_score_l,lbl,col,p,d,plotFlag, kern):
    
 #   y_score = np.array(y_score_l[:,1])
    y_score = np.array(y_score_l)

    y_test = np.array(y_test)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    x_arr = np.linspace(0.0, 1.0, num=10)
    y_thresh = ((1-p)/p)*(d/(1-d))*x_arr
    lw = 1
   
    
    S_improvement = []
    
    for i in range(0, len(fpr)):
        f = fpr[i]
        t = tpr[i]
        S_improvement.append(100.0*(t*p*(1-d)*min(1,acceptProb(d,kern)) + (1-p)*(1-f) + (1-p)*f*(1-d) - (1-p))/(1-p))
        
    if plotFlag:
        #plt.figure()
        plt.plot(thresholds,S_improvement,label=lbl)
        plt.plot(np.array(thresholds),np.zeros(len(thresholds)),color='r', linestyle='--')
        #plt.plot(np.array(thresholds),np.ones(10))
        plt.xlim([0.0, 1.0])
        plt.xlabel('Threshold')
        plt.ylabel('Improvement in S [%]')
        plt.legend(loc="lower right")
        #plt.title('Effectiveness of model')
        plt.grid()
    print("--------------------------")
    print("Discount = "+str(100.0*d)+" %")
    print("Max improvement = "+str(max(S_improvement)))
    return max(S_improvement)




def plotROC(y_test, y_score_l,label,col):
    
 #   y_score = np.array(y_score_l[:,1])
    y_score = np.array(y_score_l)

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
    


# Indices of optimal features from most significant to least
slct_inds = [0,12,3,15,21,23,4,26,6,20,2,7,25,16,13,5,19,29,27,10,28,17,1,11,30,24,31,8,18,22,9,34,38,36,33,40,39,35,14,37]
N_feat = len(slct_inds)-10 # Select top N_feat features
#N_feat = 10 # Select top N_feat features

# Importing the dataset
dataset = pd.read_csv('ml_case_training_data_cleaned.csv')
#X = dataset.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,33,34,35,36,37,38,39,40,41]].values
X = dataset.iloc[:, slct_inds].values
X = X[:,0:N_feat]
y = dataset.iloc[:, 32].values


# Filling NaNs with their mean
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X)
X = imputer.transform(X) 

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

plt.figure()



# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 200, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
y_conf_2 = classifier.predict_proba(X_test)
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print('Random Forest accuracy = ' + str(100.0*accuracy_score(y_test, y_pred)))
print('Random Forest Brier score = ' + str(brier_score_loss(y_test, y_pred)  ))
plotROC(y_test, y_conf_2[:,1],"Random Forest ",rf_col)
y_conf_RF = y_conf_2[:,1]
F_score_RF = getFscores(y_conf_RF, y_test)    
precision_RF, recall_RF, thresholds_RF = precision_recall_curve(y_test, y_conf_RF)
  

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
print('Decision tree Brier score = ' + str(brier_score_loss(y_test, y_pred)  ))
plotROC(y_test, y_conf_2[:,1],"Decision tree",dt_col)
y_conf_DT = y_conf_2[:,1]
F_score_DT = getFscores(y_conf_DT, y_test)    
precision_DT, recall_DT, thresholds_DT = precision_recall_curve(y_test, y_conf_DT)




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
print('Logistic regression Brier score = ' + str(brier_score_loss(y_test, y_pred)  ))
plotROC(y_test, y_conf_2[:,1],"Logistic Regression ",lr_col)
y_conf_LR = y_conf_2[:,1]
F_score_LR = getFscores(y_conf_LR, y_test)    
precision_LR, recall_LR, thresholds_LR = precision_recall_curve(y_test, y_conf_LR)


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
print('naive Bayes Brier score = ' + str(brier_score_loss(y_test, y_pred)  ))
plotROC(y_test, y_conf_2[:,1],"Naive Bayes",nb_col)
y_conf_NB = y_conf_2[:,1]
F_score_NB = getFscores(y_conf_NB, y_test)    
precision_NB, recall_NB, thresholds_NB = precision_recall_curve(y_test, y_conf_NB)


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
print('K-nn Brier score = ' + str(brier_score_loss(y_test, y_pred)  ))
plotROC(y_test, y_conf_2[:,1],"K-nn",knn_col)
y_conf_KN = y_conf_2[:,1]
F_score_KNN = getFscores(y_conf_KN, y_test)    
precision_KNN, recall_KNN, thresholds_KNN = precision_recall_curve(y_test, y_conf_KN)






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
print('SVM Brier score = ' + str(brier_score_loss(y_test, y_pred)  ))
plotROC(y_test, y_conf_2[:,1],"SVM",svm_col)
y_conf_SV = y_conf_2[:,1]
F_score_SVM = getFscores(y_conf_SV, y_test)    
precision_SVM, recall_SVM, thresholds_SVM = precision_recall_curve(y_test, y_conf_SV)


# Fitting Neural Network SVM to the Training set
import keras
from keras.models import Sequential
from keras.layers import Dense
classifier = Sequential()
classifier.add(Dense(units = int(np.floor(3*(N_feat-1)/4)), kernel_initializer = 'uniform', activation = 'relu', input_dim = N_feat))
classifier.add(Dense(units = int(np.floor(3*(N_feat-1)/4)), kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = int(np.floor(2*(N_feat-1)/4)), kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = int(np.floor(1*(N_feat-1)/4)), kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 5, epochs = 10)
y_conf_2 = classifier.predict(X_test)
y_pred = (y_conf_2 > 0.5)
print('NN accuracy = ' + str(100.0*accuracy_score(y_test, y_pred)))
print('NN Brier score = ' + str(brier_score_loss(y_test, y_pred)  ))
plotROC(y_test, y_conf_2,"NN",nn_col)
y_conf_NN = y_conf_2
F_score_NN = getFscores(y_conf_NN, y_test)    
precision_NN, recall_NN, thresholds_NN = precision_recall_curve(y_test, y_conf_NN)

#Accuracy
#NN accuracy = 90.5878674171 N_feat = len(slct_inds)-10


#classifier.add(Dense(units = int(np.floor(2*(N_feat-1)/3)), kernel_initializer = 'uniform', activation = 'relu', input_dim = N_feat))
#classifier.add(Dense(units = int(np.floor(2*(N_feat-1)/3)), kernel_initializer = 'uniform', activation = 'relu'))
#classifier.add(Dense(units = int(np.floor((N_feat-1)/3)), kernel_initializer = 'uniform', activation = 'relu'))







x_arr = np.linspace(0.0, 1.0, num=10)
y_thresh = ((1-p)/p)*(d/(1-d))*x_arr
plt.plot(x_arr,y_thresh,lw=1,color='r', linestyle='--',label='Margin '+'(p:'+str(100*p)+'%,d:'+str(100*d)+'%)')



plt.grid()
plt.legend(loc="lower right")

plt.show()




plt.figure()
plt.plot(recall_RF,precision_RF,label='RF')
plt.plot(recall_NN,precision_NN,label='NN')
plt.plot(recall_SVM,precision_SVM,label='SVM')
plt.plot(recall_LR,precision_LR,label='LR')
plt.plot(recall_DT,precision_DT,label='DT')
plt.plot(recall_NB,precision_NB,label='NB')
plt.plot(recall_KNN,precision_KNN,label='KNN')
plt.legend(loc="upper right")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid()
plt.show()
plt.savefig("Precision_recall.png")
plt.savefig("Precision_recall.pdf")

plt.figure()
plt.plot(threshold_vec,F_score_RF,label='RF')
plt.plot(threshold_vec,F_score_DT,label='DT')
plt.plot(threshold_vec,F_score_NB,label='NB')
plt.plot(threshold_vec,F_score_SVM,label='SVM')
plt.plot(threshold_vec,F_score_NN,label='NN')
plt.plot(threshold_vec,F_score_LR,label='LR')
plt.plot(threshold_vec,F_score_KNN,label='KNN')
plt.legend(loc="upper right")
plt.xlabel('Threshold')
plt.ylabel('F 1 score')
plt.grid()
plt.show()
plt.savefig("F1score.png")
plt.savefig("F1score.pdf")


plt.figure()
pval = p
dval = d
plotRET(y_test, y_conf_DT,lbl='DT',col=dt_col,p=pval,d=dval,plotFlag=True, kern="Quad")
plotRET(y_test, y_conf_RF,lbl='RF',col=rf_col,p=pval,d=dval,plotFlag=True, kern="Quad")
plotRET(y_test, y_conf_SV,lbl='SV',col=svm_col,p=pval,d=dval,plotFlag=True, kern="Quad")
plotRET(y_test, y_conf_NN,lbl='NN',col=nn_col,p=pval,d=dval,plotFlag=True, kern="Quad")
plotRET(y_test, y_conf_NB,lbl='NB',col=nb_col,p=pval,d=dval,plotFlag=True, kern="Quad")
plotRET(y_test, y_conf_LR,lbl='LR',col=lr_col,p=pval,d=dval,plotFlag=True, kern="Quad")
plt.grid()
plt.ylim([-2,2.5])
plt.show()

















