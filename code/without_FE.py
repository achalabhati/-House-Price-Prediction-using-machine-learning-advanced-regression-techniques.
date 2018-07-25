#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 15:05:55 2017

@author: achala
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
from sklearn.linear_model import Ridge,Lasso
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

from sklearn.neural_network import MLPRegressor

from sklearn.kernel_ridge import KernelRidge
from sklearn import svm

from sklearn.linear_model import SGDRegressor

from sklearn import linear_model

from sklearn.cross_decomposition import PLSRegression

from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split




raw_train = pd.read_csv('train.csv')
raw_test = pd.read_csv('test.csv')

raw_train = raw_train.replace(to_replace='NA',value=np.nan)

raw_test = raw_test.replace(to_replace='NA',value=np.nan)

raw_train = raw_train.apply(lambda x:x.fillna(x.value_counts().index[0]))


raw_test = raw_test.apply(lambda x:x.fillna(x.value_counts().index[0]))

#print (raw_train)

cat_list = []
for i in raw_train.columns:
    if raw_train[i].dtype == object:
        raw_train[i] = raw_train[i].astype('category')
        cat_list += [i]


#print (cat_list)

number = LabelEncoder()
for i in cat_list:
    raw_train[i]=number.fit_transform(raw_train[i].astype('str'))
    
cat_list = []
for i in raw_test.columns:
    if raw_test[i].dtype == object:
        raw_test[i] = raw_test[i].astype('category')
        cat_list += [i]




#print (cat_list)

number = LabelEncoder()
for i in cat_list:
    raw_test[i]=number.fit_transform(raw_test[i].astype('str'))
    
corelation_mat  = raw_train.corr()
sns.heatmap(corelation_mat, vmax=.8, square=True);
X_train= raw_train.loc[:,'MSSubClass':'SaleCondition'];

X_test=raw_test.loc[:,'MSSubClass':'SaleCondition'];
                  
raw_train['SalePrice'].describe()


Y_train=raw_train.SalePrice


############################################ Ridge ######################################
## 0.9316
X1 = np.array(X_train)
Y1 = np.array(Y_train)
X_test1 = np.array(X_test)
reg = Ridge(alpha = 0.1)
reg.fit(X1,Y1);
op=reg.predict(X_test1);

              
yo1 = pd.DataFrame(columns=["Id","SalePrice"])
yo1["Id"]=raw_test.Id.values
yo1["SalePrice"]=op

yo1.to_csv('predictions_ridge.csv',index=False)
print (op);

########################################### Ridge #######################################
########################################## Ridge cross validation ############################
######## 0.55
def model_generation(X_train,y_train,model):   
    kf=KFold(X_train.shape[0], n_folds=8)
    rf = Ridge(alpha = 0.1)
    if(model=="Lasso"):
        rf = Lasso(alpha = 0.1)
    if(model=="kernel"):
        rf = KernelRidge(alpha=1, kernel='linear', gamma=None, degree=3, coef0=1, kernel_params=None)
    if(model=="Neural"):
        rf = MLPRegressor(hidden_layer_sizes=(100, ), activation='relu',learning_rate='constant', learning_rate_init=0.01, max_iter=1000)
    if(model=="svm"):
        rf = svm.SVR()
    if(model=="partial"):
        rf = PLSRegression()
    if(model=="Bay"):
        rf = linear_model.BayesianRidge()
    err=list()
    min_rf = rf
    min_e=100
    for a,b in kf:
        print ("hrrrrr")
        X_trn, X_tst = X_train[a] , X_train[b]
        y_trn, y_tst = y_train[a], y_train[b]
        rf.fit(X_trn,y_trn)
        y_pred=rf.predict(X_tst)
        err = mean_squared_error(y_tst, y_pred)
        if(err < min_e):
            min_e = err
            min_rf = rf
    return min_rf


X_train, X_test, y_train, y_test = train_test_split(X1, Y1, test_size=0.33, random_state=42) #dividing trainig data into train and test
rf=model_generation(X_train , y_train,"Ridge")  
op=rf.predict(X_test1)
         
yo1 = pd.DataFrame(columns=["Id","SalePrice"])
yo1["Id"]=raw_test.Id.values
yo1["SalePrice"]=op

yo1.to_csv('predictions_Ridge_cross.csv',index=False)
print ("k-fold_validation Ridge" ,op);

#########################################Ridge cross validation #############################
########################################### Lasso #######################################
#0.94
reg = Lasso(alpha = 0.1)
reg.fit(X1,Y1);
op=reg.predict(X_test1);

              
yo1 = pd.DataFrame(columns=["Id","SalePrice"])
yo1["Id"]=raw_test.Id.values
yo1["SalePrice"]=op

yo1.to_csv('predictions_Lasso.csv',index=False)
print (op);
########################################### Lasso #######################################
########################################### Lasso Cross Validation ######################
#0.57
rf=model_generation(X_train , y_train,"Lasso")  
op=rf.predict(X_test1)
         
yo1 = pd.DataFrame(columns=["Id","SalePrice"])
yo1["Id"]=raw_test.Id.values
yo1["SalePrice"]=op

yo1.to_csv('predictions_Lasso_cross.csv',index=False)
print ("k-fold_validation Lasso" ,op);

########################################## Lasso Cross Validation #######################
########################################## Neural Network ###############################
#0.218
reg = MLPRegressor(hidden_layer_sizes=(100, ), activation='relu',learning_rate='constant', learning_rate_init=0.01, max_iter=1000)
reg.fit(X1, Y1)
op=reg.predict(X_test1);

              
yo1 = pd.DataFrame(columns=["Id","SalePrice"])
yo1["Id"]=raw_test.Id.values
yo1["SalePrice"]=op

yo1.to_csv('predictions_nn.csv',index=False)
print (op);
########################################## Neural Network ###############################
########################################## Neural Network Cross Validation ##############
#0.233
rf=model_generation(X_train , y_train,"Neural")  
op=rf.predict(X_test1)
         
yo1 = pd.DataFrame(columns=["Id","SalePrice"])
yo1["Id"]=raw_test.Id.values
yo1["SalePrice"]=op

yo1.to_csv('predictions_nn_cross.csv',index=False)
print ("k-fold_validation Neural" ,op);



########################################## Neural Network Cross Validation ##############
######################################### Kernelized ridge regression ###################

reg = KernelRidge(alpha=1, kernel='linear', gamma=None, degree=3, coef0=1, kernel_params=None)
reg.fit(X1, Y1)
op=reg.predict(X_test1);

              
yo1 = pd.DataFrame(columns=["Id","SalePrice"])
yo1["Id"]=raw_test.Id.values
yo1["SalePrice"]=op

yo1.to_csv('predictions_Kernelized.csv',index=False)
print (op);
#########################################  Kernelized ridge regression #####################
######################################### Cross Validation Kernelized ridge #######################
rf=model_generation(X_train , y_train,"kernel")  
op=rf.predict(X_test1)
         
yo1 = pd.DataFrame(columns=["Id","SalePrice"])
yo1["Id"]=raw_test.Id.values
yo1["SalePrice"]=op

yo1.to_csv('predictions_kernel_cross.csv',index=False)
print ("k-fold_validation kernel" ,op);


######################################### Cross Validation Kernelized ridge #######################

######################################### SVM ##############################################

reg = svm.SVR() # rbf, linear 
#print (Y1)
reg.fit(X1, Y1) 
op=reg.predict(X_test1);

              
yo1 = pd.DataFrame(columns=["Id","SalePrice"])
yo1["Id"]=raw_test.Id.values
yo1["SalePrice"]=op

yo1.to_csv('predictions_svm.csv',index=False)
print (op);

######################################## SVM ################################################
######################################## SVM Cross #############################################
rf=model_generation(X_train , y_train,"svm")  
op=rf.predict(X_test1)
         
yo1 = pd.DataFrame(columns=["Id","SalePrice"])
yo1["Id"]=raw_test.Id.values
yo1["SalePrice"]=op

yo1.to_csv('predictions_svm_cross.csv',index=False)
print ("k-fold_validation svm" ,op);


######################################## SVM Cross #############################################
######################################## Partial Least Square ###############################

reg = PLSRegression()
reg.fit(X1, Y1) 
op=reg.predict(X_test1);

              
yo1 = pd.DataFrame(columns=["Id","SalePrice"])
yo1["Id"]=raw_test.Id.values
yo1["SalePrice"]=op

yo1.to_csv('predictions7_pls.csv',index=False)
print (op);

######################################## Partial Least Square ###############################
######################################### Partial Cross valid ###############################
rf=model_generation(X_train , y_train,"partial")  
op=rf.predict(X_test1)
         
yo1 = pd.DataFrame(columns=["Id","SalePrice"])
yo1["Id"]=raw_test.Id.values
yo1["SalePrice"]=op

yo1.to_csv('predictions_partial_cross.csv',index=False)
print ("k-fold_validation Partial" ,op);

######################################### Partial Cross Valid ################################
######################################## stocastic gradient decent ##########################
# for more than 10000 sample

reg = SGDRegressor(alpha=0.6,n_iter=1000);
reg.fit(X1, Y1) 
op=reg.predict(X_test1);

              
yo1 = pd.DataFrame(columns=["Id","SalePrice"])
yo1["Id"]=raw_test.Id.values
yo1["SalePrice"]=op

yo1.to_csv('predictions_SGD.csv',index=False)
print('SGSSSSSSSSSSSSSSSSSSSSS')
print (op);
print (X1.shape)

####################################### stocastic gradient decent ###########################

###################################### Baysian regression ##################################
reg = linear_model.BayesianRidge()
reg.fit(X1, Y1) 
op=reg.predict(X_test1);

              
yo1 = pd.DataFrame(columns=["Id","SalePrice"])
yo1["Id"]=raw_test.Id.values
yo1["SalePrice"]=op

yo1.to_csv('predictions_bayesian.csv',index=False)
print (op);

###################################### Baysian regression ##################################
###################################### Baysian Cross #######################################
rf=model_generation(X_train , y_train,"Bay")  
op=rf.predict(X_test1)
         
yo1 = pd.DataFrame(columns=["Id","SalePrice"])
yo1["Id"]=raw_test.Id.values
yo1["SalePrice"]=op

yo1.to_csv('predictions_Bay_cross.csv',index=False)
print ("k-fold_validation Baysian" ,op);

###################################### Baysian Cross ########################################