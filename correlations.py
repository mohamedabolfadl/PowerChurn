
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 11:34:32 2017
Script to find the correlations among features and cross correlation between features and churn
Writes Correlations.csv and Correlations_churn.csv

@author: m00760171
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('ml_case_training_data_cleaned.csv')
X = dataset.iloc[:, [25]].values
y = dataset.iloc[:, 0].values


# Filling NaNs with their mean
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X)
X = imputer.transform(X) 
imputer = imputer.fit(y)
y = imputer.transform(y) 


# Scatter plot of consumed and subscribed powers
plt.scatter(X, y/10000,s =0.2)
plt.show()
plt.grid()
plt.xlabel("Subscribed")
plt.ylabel("Consumed")
plt.ylim([0,700])
plt.xlim([0,200])

X = dataset.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]].values
y = dataset.iloc[:, [32]].values

# Filling NaNs with their mean
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X)
X = imputer.transform(X) 


# Correlations of all features
X_a = np.array(X)
dims = X_a.shape
plt.figure()
corrs = np.zeros((dims[1], dims[1]))
for i in range(dims[1]):
    for j in range(dims[1]):
        R = np.corrcoef(X_a[:,i],X_a[:,j])
        cc = R[0][1]
        corrs[i][j]=cc
        if cc<0:
            plt.scatter(i,j,s=150.0*abs(cc),alpha=0.5, c = [1.0*abs(cc),0,0.0])
        else:
            plt.scatter(i,j,s=150.0*abs(cc),alpha=0.5, c = [0.0,0,1.0*abs(cc)])
           
plt.grid()
colns = list(dataset)
cols = colns[0:32]
df_all_corrs = pd.DataFrame(corrs, columns=cols, index = cols)
df_all_corrs.to_csv('Correlations.csv', index=False)

# Image of correlations
plt.figure()
plt.imshow(corrs)

# Correlations of features
y_r = np.array(y)
corrs = np.zeros(( dims[1]))
for i in range(dims[1]):
        R = np.corrcoef(X_a[:,i],y_r[:,0])
        cc = R[0][1]
        corrs[i]=cc    

colns = list(dataset)
cols = colns[0:32]
df_churn = pd.DataFrame(corrs, columns=["Churn"], index = cols)
df_churn.to_csv('Correlations_churn.csv', index=True)


# Plot top 4 correlated features to churn
fig, ax = plt.subplots()
ind = np.arange(5)
width = 0.2 
rects1 = ax.bar(1-width/2, 0.0809, width, color='b')
rects2 = ax.bar(2-width/2, 0.0736, width, color='r')
rects3 = ax.bar(3-width/2, 0.0519, width, color='r')
rects4 = ax.bar(4-width/2, 0.0503, width, color='r')
ax.set_ylabel('|Correlation|')
ax.set_title('Churn correlations')
#plt.xlim([0,5])
#ax.set_xticks(ind + width / 2)
ax.set_xticklabels(( '','margin_gross_pow_ele', '','date_activ','', 'cons_12m','', 'date_modif'))
ax.grid()




# Churn channel_sales correlations
df_ch_0 = dataset[dataset.channel_sales_0 == 1]
df_ch_1 = dataset[dataset.channel_sales_1 == 1]
df_ch_2 = dataset[dataset.channel_sales_2 == 1]
df_ch_3 = dataset[dataset.channel_sales_3 == 1]
N_ch_0 = len(df_ch_0)
N_ch_1 = len(df_ch_1)
N_ch_2 = len(df_ch_2)
N_ch_3 = len(df_ch_3)
N_tot_users = N_ch_0 + N_ch_1 + N_ch_2 + N_ch_3
Ch_0_churn = len(df_ch_0[df_ch_0.churn==1])
Ch_1_churn = len(df_ch_1[df_ch_1.churn==1])
Ch_2_churn = len(df_ch_2[df_ch_2.churn==1])
Ch_3_churn = len(df_ch_3[df_ch_3.churn==1])
N_tot_churn = Ch_0_churn+Ch_1_churn+Ch_2_churn+Ch_3_churn
fig, ax = plt.subplots()
ind = np.arange(5)
width = 0.2 
rects1 = ax.bar(1-1*width, 100.0*Ch_0_churn/N_ch_0, width, color='r',label='Churns per channel')
rects1 = ax.bar(2-1*width, 100.0*Ch_1_churn/N_ch_1, width, color='r')
rects1 = ax.bar(3-1*width, 100.0*Ch_2_churn/N_ch_2, width, color='r')
rects1 = ax.bar(4-1*width, 100.0*Ch_3_churn/N_ch_3, width, color='r')
rects1 = ax.bar(1+0*width, 100.0*Ch_0_churn/N_tot_churn, width, color='g',label='Churns per total')
rects1 = ax.bar(2+0*width, 100.0*Ch_1_churn/N_tot_churn, width, color='g')
rects1 = ax.bar(3+0*width, 100.0*Ch_2_churn/N_tot_churn, width, color='g')
rects1 = ax.bar(4+0*width, 100.0*Ch_3_churn/N_tot_churn, width, color='g')
rects1 = ax.bar(1+1.0*width, 100.0*N_ch_0/N_tot_users, width, color='b',label='Sales per total')
rects1 = ax.bar(2+1.0*width, 100.0*N_ch_1/N_tot_users, width, color='b')
rects1 = ax.bar(3+1.0*width, 100.0*N_ch_2/N_tot_users, width, color='b')
rects1 = ax.bar(4+1.0*width, 100.0*N_ch_3/N_tot_users, width, color='b')
ax.legend(loc="upper right")
ax.set_ylabel('[%]')
ax.set_title('Sales channel analysis')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(( '','Chan A', 'Chan B', 'Chan C', 'Chan D'))
ax.grid()


# Churn origin correlations
df_ch_0 = dataset[dataset.origin_up_kamkkxfxxuwbdslkwifmmcsiusiuosws == 1]
df_ch_1 = dataset[dataset.origin_up_ldkssxwpmemidmecebumciepifcamkci == 1]
df_ch_2 = dataset[dataset.origin_up_lxidpiddsbxsbosboudacockeimpuepw == 1]
N_ch_0 = len(df_ch_0)
N_ch_1 = len(df_ch_1)
N_ch_2 = len(df_ch_2)
N_tot_users = N_ch_0 + N_ch_1 + N_ch_2 
Ch_0_churn = len(df_ch_0[df_ch_0.churn==1])
Ch_1_churn = len(df_ch_1[df_ch_1.churn==1])
Ch_2_churn = len(df_ch_2[df_ch_2.churn==1])
N_tot_churn = Ch_0_churn+Ch_1_churn+Ch_2_churn
fig, ax = plt.subplots()
ind = np.arange(4)
width = 0.2 
rects1 = ax.bar(1-0.5*width, 100.0*Ch_0_churn/N_ch_0, width, color='r',label='Churns per campaign')
rects1 = ax.bar(2-0.5*width, 100.0*Ch_1_churn/N_ch_1, width, color='r')
rects1 = ax.bar(3-0.5*width, 100.0*Ch_2_churn/N_ch_2, width, color='r')
rects1 = ax.bar(1+0.5*width, 100.0*Ch_0_churn/N_tot_churn, width, color='g',label='Churns per total')
rects1 = ax.bar(2+0.5*width, 100.0*Ch_1_churn/N_tot_churn, width, color='g')
rects1 = ax.bar(3+0.5*width, 100.0*Ch_2_churn/N_tot_churn, width, color='g')
rects1 = ax.bar(1+1.5*width, 100.0*N_ch_0/N_tot_users, width, color='b',label='Sales per campaign')
rects1 = ax.bar(2+1.5*width, 100.0*N_ch_1/N_tot_users, width, color='b')
rects1 = ax.bar(3+1.5*width, 100.0*N_ch_2/N_tot_users, width, color='b')
ax.legend(loc="upper left")
ax.set_ylabel('[%]')
ax.set_title('Campaign analysis')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(( '','Campaign A', 'Campaign B', 'Campaign C'))
ax.grid()
plt.xlim([0.5,3.5])


















