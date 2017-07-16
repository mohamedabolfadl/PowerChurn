# -*- coding: utf-8 -*-
"""
Created on Sun Jul 09 23:06:34 2017
Script to preprocess the features and prepare them for learning
@author: Moh2
"""


# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from datetime import datetime
import re

current_date = '2016-01-01'

# Categories
customer_IDS = []
company_IDS = []
channel_IDS = []
origin_IDS = []
gas_IDS = []


plotFlag = False


# Function to check if a strings is a date 
def is_date(st):
    if type(st) is str:
        match=re.match(r'\d{4}-\d{2}-\d{2}',st)
        if match:
            return True
        else:
            return False
    else:
        return False

# Function to find the days between two dates
def days_between(d1, d2):
    d1 = datetime.strptime(d1, "%Y-%m-%d")
    d2 = datetime.strptime(d2, "%Y-%m-%d")
    return abs((d2 - d1).days)
    
                
# Function to change dates from strings to number of days before/after 2016-01-01
def transformDates(ds):
     for index, row in ds.iterrows():
            date_activ = row['date_activ']
            date_end = row['date_end']
            date_first_activ = row['date_first_activ']
            date_modif_prod = row['date_modif_prod']
            date_renewal = row['date_renewal']
            
            if is_date(date_activ): 
                ds.set_value(index, 'date_activ', days_between(date_activ, current_date))
            if is_date(date_end): 
                ds.set_value(index, 'date_end', days_between(date_end, current_date))            
            if is_date(date_first_activ): 
                ds.set_value(index, 'date_first_activ', days_between(date_first_activ, current_date))
            if is_date(date_modif_prod): 
                ds.set_value(index, 'date_modif_prod', days_between(date_modif_prod, current_date))            
            if is_date(date_renewal): 
                ds.set_value(index, 'date_renewal', days_between(date_renewal, current_date))   
                
                
# Function to compute the number of unique customers/company/channel/origin
def collectCompanyIDS(ds):
     for index, row in ds.iterrows():
            customer = row['id']
            company = row['activity_new']
            channel = row['channel_sales']
            origin = row['origin_up']
            gas = row['has_gas']
            
            if not customer in customer_IDS:
                customer_IDS.append(customer)
                
            if not channel in channel_IDS:
                channel_IDS.append(channel)               
                
            if not company in company_IDS:
                company_IDS.append(company)                
                
            if not origin in origin_IDS:
                origin_IDS.append(origin)
            
            if not gas in gas_IDS:
                gas_IDS.append(gas)    

# Fill churns per category
def churnsPerCategory(dx,dy, N_companies, N_channels, N_origins):

    # Number of churns per category
    company_IDS_churns = np.zeros(N_companies)
    channel_IDS_churns = np.zeros(N_channels)
    origin_IDS_churns = np.zeros(N_origins)
    gas_IDS_churns = np.zeros(2)
    
    # Initialize total per category
    company_IDS_churns_totals = np.zeros(N_companies)
    channel_IDS_churns_totals = np.zeros(N_channels)
    origin_IDS_churns_totals = np.zeros(N_origins)
    gas_IDS_churns_totals = np.zeros(2)


    for index, row in dx.iterrows():
        company = row['activity_new']
        channel = row['channel_sales']
        origin = row['origin_up']
        gas = row['has_gas']
            
        # Fill in total per category
        company_IDS_churns_totals[company_IDS.index(company)] =company_IDS_churns_totals[company_IDS.index(company)] +1
        origin_IDS_churns_totals[origin_IDS.index(origin)] =origin_IDS_churns_totals[origin_IDS.index(origin)] +1
        channel_IDS_churns_totals[channel_IDS.index(channel)] =channel_IDS_churns_totals[channel_IDS.index(channel)] +1
        gas_IDS_churns_totals[gas_IDS.index(gas)] =gas_IDS_churns_totals[gas_IDS.index(gas)] +1        
        
        
        # Fill in the churns
        churned = dy.at[index,'churn']  
        if churned == 1:

            company_IDS_churns[company_IDS.index(company)] =company_IDS_churns[company_IDS.index(company)] +1
            origin_IDS_churns[origin_IDS.index(origin)] =origin_IDS_churns[origin_IDS.index(origin)] +1
            channel_IDS_churns[channel_IDS.index(channel)] =channel_IDS_churns[channel_IDS.index(channel)] +1
            gas_IDS_churns[gas_IDS.index(gas)] =gas_IDS_churns[gas_IDS.index(gas)] +1
    
    
    return [company_IDS_churns, origin_IDS_churns, channel_IDS_churns, gas_IDS_churns, company_IDS_churns_totals, origin_IDS_churns_totals, channel_IDS_churns_totals, gas_IDS_churns_totals]
    
    
# Average the price over the whole year         
def fillPrices(dataset,dataset_prices,legend):
    for index, row in dataset.iterrows():
        curr_id_code = row['id_code']
        ind = legend.tolist().index(curr_id_code)
        en = min(ind+12,len(dataset_prices))
        av_vec = dataset_prices[['price_p1_var', 'price_p2_var', 'price_p3_var','price_p1_fix', 'price_p2_fix', 'price_p3_fix']].iloc[ind:en].mean(axis=0)
        dataset.set_value(index, 'price_p1_var',av_vec[0] )
        dataset.set_value(index, 'price_p2_var',av_vec[1] )
        dataset.set_value(index, 'price_p3_var',av_vec[2] )
        dataset.set_value(index, 'price_p1_fix',av_vec[3] )
        dataset.set_value(index, 'price_p2_fix',av_vec[4] )
        dataset.set_value(index, 'price_p3_fix',av_vec[5] )
        
        
# This section takes some time. Run it ONLY if 'dataset_with_prices.csv' has not been previously generated
# Reading out the feature file and price file and adding the price features to the feature file        
if False:         
    # Importing the dataset
    dataset_prices = pd.read_csv('ml_case_training_hist_data.csv')
    dataset = pd.read_csv('ml_case_training_data.csv')
    #Creating price information columns
    dataset['price_p1_var'] = pd.Series(np.nan, index=dataset.index)
    dataset['price_p2_var'] = pd.Series(np.nan, index=dataset.index)
    dataset['price_p3_var'] = pd.Series(np.nan, index=dataset.index)
    dataset['price_p1_fix'] = pd.Series(np.nan, index=dataset.index)
    dataset['price_p2_fix'] = pd.Series(np.nan, index=dataset.index)
    dataset['price_p3_fix'] = pd.Series(np.nan, index=dataset.index)
    # Extracting customer hash id
    y = dataset_prices.iloc[:, 0].values
    y_o = dataset.iloc[:, 0].values
    # Convert hash to integer
    from sklearn.preprocessing import LabelEncoder
    labelencoder_y = LabelEncoder()
    y = labelencoder_y.fit_transform(y)
    y_o = labelencoder_y.fit_transform(y_o)
    # Insert id_code column to have easier to handle user id
    dataset_prices['id_code'] = pd.Series(y, index=dataset_prices.index)
    dataset['id_code'] = pd.Series(y_o, index=dataset.index)
    # Getting a legend of hash to integer for easy access of ids
    legend = dataset_prices.iloc[:, 8].values
    # Fill the dataset with average the prices per period
    fillPrices(dataset,dataset_prices,legend) # 4:30 mins
    dataset.to_csv('dataset_with_prices.csv', index=False)


dataset = pd.read_csv('dataset_with_prices.csv')
dataset_result = pd.read_csv('ml_case_training_output.csv')
churns_rate_2015 = 100.0*float(len(dataset_result[dataset_result['churn']==1]))/ float(len(dataset_result[dataset_result['churn']==1])+len(dataset_result[dataset_result['churn']==0]))

# Collecting categorial data
collectCompanyIDS(dataset)

# Counting churn rate per category
churns_per_category_all = churnsPerCategory(dataset,dataset_result,len(company_IDS),len(channel_IDS),len(origin_IDS))

# Getting totals per feature
N_companies = sum(churns_per_category_all[4])
N_origins = sum(churns_per_category_all[5])
N_channels = sum(churns_per_category_all[6])
N_gas = sum(churns_per_category_all[7])


if plotFlag:
    
    #Getting Number of users per category
    company_hist_percentage = 100.0*churns_per_category_all[4]/N_companies
    origin_hist_percentage = 100.0*churns_per_category_all[5]/N_origins
    channel_hist_percentage = 100.0*churns_per_category_all[6]/N_channels
    gas_hist_percentage = 100.0*churns_per_category_all[7]/N_gas
    
    # Getting churn percentage per category
    company_churn_percentage = 100.0*churns_per_category_all[0]/churns_per_category_all[4]
    origin_churn_percentage = 100.0*churns_per_category_all[1]/churns_per_category_all[5]
    channel_churn_percentage = 100.0*churns_per_category_all[2]/churns_per_category_all[6]
    gas_churn_percentage = 100.0*churns_per_category_all[3]/churns_per_category_all[7]
    
    
    # Bar chart of companies
    fig, ax = plt.subplots()
    ind = np.arange(len(company_IDS))
    width = 0.35 
    rects1 = ax.bar(ind, company_hist_percentage, width, color='g')
    ax.set_ylabel('Amount [%]')
    ax.set_title('Users per company category')
    ax.set_xticks(ind + width / 2)
    #ax.set_xticklabels(('S1', 'S2', 'NaN', 'S4', 'S5','S6', 'S7', 'S8'))
    plt.savefig('Companies_hist.png')
    plt.savefig('Companies_hist.pdf')
    
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, company_churn_percentage, width, color='g')
    ax.set_ylabel('Amount [%]')
    ax.set_title('Churn per company category')
    ax.set_xticks(ind + width / 2)
    plt.savefig('Companies_churn.png')
    plt.savefig('Companies_churn.pdf')
    
    #ax.set_xticklabels(('S1', 'S2', 'NaN', 'S4', 'S5','S6', 'S7', 'S8'))
    
    # Bar chart of origin
    fig, ax = plt.subplots()
    ind = np.arange(len(origin_IDS))
    width = 0.35 
    rects1 = ax.bar(ind, origin_hist_percentage, width, color='r')
    ax.set_ylabel('Amount [%]')
    ax.set_title('Users per origin')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(('O1', 'O2', 'O3', 'NaN', 'O5','S6'))
    plt.savefig('Origin_hist.png')
    plt.savefig('Origin_hist.pdf')
    
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, origin_churn_percentage, width, color='r')
    ax.set_ylabel('Amount [%]')
    ax.set_title('Churn per origin')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(('O1', 'O2', 'O3', 'NaN', 'O5','S6'))
    plt.savefig('Origin_churn.png')
    plt.savefig('Origin_churn.pdf')
    
    
    # Bar chart of channels
    fig, ax = plt.subplots()
    ind = np.arange(len(channel_IDS))
    width = 0.35 
    rects1 = ax.bar(ind, channel_hist_percentage, width, color='r')
    ax.set_ylabel('Amount [%]')
    ax.set_title('Users per sales channel')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(('S1', 'S2', 'NaN', 'S4', 'S5','S6', 'S7', 'S8'))
    plt.savefig('Channel_hist.png')
    plt.savefig('Channel_hist.pdf')
    
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, channel_churn_percentage, width, color='r')
    ax.set_ylabel('Amount [%]')
    ax.set_title('Churn per sales channel')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(('S1', 'S2', 'NaN', 'S4', 'S5','S6', 'S7', 'S8'))
    plt.savefig('Channel_churn.png')
    plt.savefig('Channel_churn.pdf')
    
    
    
    # Bar chart of gas
    fig, ax = plt.subplots()
    ind = np.arange(len(gas_IDS))
    width = 0.35 
    rects1 = ax.bar(ind, gas_hist_percentage, width, color='r')
    ax.set_ylabel('Amount [%]')
    ax.set_title('Users per gas subscription')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(('False', 'True'))
    plt.savefig('Gas_hist.png')
    plt.savefig('Gas_hist.pdf')
    
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, gas_churn_percentage, width, color='r')
    ax.set_ylabel('Amount [%]')
    ax.set_title('Churn per gas subscription')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(('False', 'True'))
    plt.savefig('Gas_churn.png')
    plt.savefig('Gas_churn.pdf')



# Merge 
dataset_full = pd.concat([dataset, dataset_result.churn], axis=1)

# Cleaning the data frame from observations which might enlarge the number of features
#Cleaning channel
dataset_nosales = dataset_full[dataset.channel_sales != 'epumfxlbckeskwekxbiuasklxalciiuu']
dataset_nosales = dataset_nosales[dataset_nosales.channel_sales != 'sddiedcslfslkckwlfkdpoeeailfpeds' ]
dataset_nosales = dataset_nosales[ dataset_nosales.channel_sales != 'fixdbufsefwooaasfcxdxadsiekoceaa']
#Cleaning origin
dataset_cleaned = dataset_nosales.dropna(subset=['origin_up']) #Removing NaN
dataset_cleaned = dataset_cleaned[dataset_cleaned.origin_up != 'usapbepcfoloekilkwsdiboslwaxobdp' ]
dataset_cleaned = dataset_cleaned[dataset_cleaned.origin_up != 'ewxeelcelemmiwuafmddpobolfuxioce']

# Transforming dates to time in days
transformDates(dataset_cleaned)
    
# Dropping campaign_disc_ele since it is only NaN, also drop activity_new since it is made of many categories
dataset_cleaned.drop('campaign_disc_ele', axis=1, inplace=True)
dataset_cleaned.drop('activity_new', axis=1, inplace=True)
    
#--------- Predict NaN of sales channel ------------------

#Filling channel_sales with 0 for easy selection    
dataset_cleaned['channel_sales'].fillna(0, inplace=True)

#Selecting training and test set to fill NaN 
dataset_channel_sales_training = dataset_cleaned[dataset_cleaned.channel_sales != 0]
dataset_channel_sales_predict = dataset_cleaned[dataset_cleaned.channel_sales == 0]

# Dropping useless columns
dataset_channel_sales_training.drop('id', axis=1, inplace=True)
dataset_channel_sales_training.drop('id_code', axis=1, inplace=True)
dataset_channel_sales_training_gas = pd.get_dummies(dataset_channel_sales_training, columns=['has_gas'])    
dataset_channel_sales_training_gas_origin = pd.get_dummies(dataset_channel_sales_training_gas, columns=['origin_up'])    

dataset_channel_sales_predict.drop('id', axis=1, inplace=True)
dataset_channel_sales_predict.drop('id_code', axis=1, inplace=True)
dataset_channel_sales_predict_gas = pd.get_dummies(dataset_channel_sales_predict, columns=['has_gas'])    
dataset_channel_sales_predict_gas_origin = pd.get_dummies(dataset_channel_sales_predict_gas, columns=['origin_up']) 




# Average out missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
X_r = dataset_channel_sales_training_gas_origin.iloc[:, 1:].values
# Deleting churn column
X = np.delete(X_r,32,1)

y = dataset_channel_sales_training_gas_origin.iloc[:, 0].values

X_predict_r = dataset_channel_sales_predict_gas_origin.iloc[:, 1:].values
X_predict = np.delete(X_predict_r,32,1)




imputer = imputer.fit(X[:, 1:33])
X[:, 1:33] = imputer.transform(X[:, 1:33])

imputer = imputer.fit(X_predict[:, 1:33])
X_predict[:, 1:33] = imputer.transform(X_predict[:, 1:33])




from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_predict = sc.transform(X_predict)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

classifier1 = KNeighborsClassifier(n_neighbors = 4, metric = 'minkowski', p = 1)
classifier1.fit(X_train, y_train)
y_pred1 = classifier1.predict(X_train)
cm1 = confusion_matrix(y_train, y_pred1)
print 'Model 1 : '+str(accuracy_score(y_train, y_pred1))+' model = ('+str(4)+','+str(1)+')'
 
classifier2 = KNeighborsClassifier(n_neighbors = 2, metric = 'minkowski', p = 1)
classifier2.fit(X_train, y_train)
y_pred2 = classifier2.predict(X_train)
cm2 = confusion_matrix(y_train, y_pred2)
print 'Model 2 : '+str(accuracy_score(y_train, y_pred2))+' model = ('+str(2)+','+str(1)+')'

classifier3 = KNeighborsClassifier(n_neighbors = 3, metric = 'minkowski', p = 1)
classifier3.fit(X_train, y_train)
y_pred3 = classifier3.predict(X_train)
cm3 = confusion_matrix(y_train, y_pred3)
print 'Model 3 : '+str(accuracy_score(y_train, y_pred3))+' model = ('+str(3)+','+str(1)+')'

classifier4 = KNeighborsClassifier(n_neighbors = 4, metric = 'minkowski', p = 2)
classifier4.fit(X_train, y_train)
y_pred4 = classifier4.predict(X_train)
cm4 = confusion_matrix(y_train, y_pred4)
print 'Model 4 : '+str(accuracy_score(y_train, y_pred4))+' model = ('+str(4)+','+str(2)+')'

classifier5 = KNeighborsClassifier(n_neighbors = 2, metric = 'minkowski', p = 2)
classifier5.fit(X_train, y_train)
y_pred5 = classifier5.predict(X_train)
cm5 = confusion_matrix(y_train, y_pred5)
print 'Model 5 : '+str(accuracy_score(y_train, y_pred5))+' model = ('+str(2)+','+str(2)+')'
  
classifier6 = KNeighborsClassifier(n_neighbors = 3, metric = 'minkowski', p = 2)
classifier6.fit(X_train, y_train)
y_pred6 = classifier6.predict(X_train)
cm6 = confusion_matrix(y_train, y_pred6)
print 'Model 6 : '+str(accuracy_score(y_train, y_pred6))+' model = ('+str(3)+','+str(2)+')'
    
# Predicted channel sales using best classifier 
y_predict = classifier2.predict(X_predict)
        
        
# Remove old channel sales with 0 value which is NaN        
dataset_channel_sales_predict_gas_origin.drop('channel_sales', axis=1, inplace=True)
# Remove old hashes to be replaced with their integer labels
dataset_channel_sales_training_gas_origin.drop('channel_sales', axis=1, inplace=True)
    
# Merge predicted sales_channel
dataset_channel_sales_predict_gas_origin['channel_sales'] = pd.Series(y_predict, index=dataset_channel_sales_predict_gas_origin.index)
dataset_channel_sales_training_gas_origin['channel_sales'] = pd.Series(y, index=dataset_channel_sales_training_gas_origin.index)

# Creat dummy of the channel_sales
#dataset_channel_sales_predict_gas_origin = pd.get_dummies(dataset_channel_sales_predict_gas_origin, columns=['channel_sales'])    
#dataset_channel_sales_training_gas_origin = pd.get_dummies(dataset_channel_sales_training_gas_origin, columns=['channel_sales'])    

# Complete dataset    
dataset_merged = dataset_channel_sales_training_gas_origin.append(dataset_channel_sales_predict_gas_origin)
dataset_complete = pd.get_dummies(dataset_merged, columns=['channel_sales'])     

dataset_complete.to_csv('ml_case_training_data_cleaned.csv', index=False)    
    
    
    
    
