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

#########################BOOSTING STARTS##########################################
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score
from scipy.stats import skew
raw_train_boost = pd.read_csv('train.csv')
raw_test_boost = pd.read_csv('test.csv')
all_data = pd.concat((raw_train_boost.loc[:,'MSSubClass':'SaleCondition'],raw_test_boost.loc[:,'MSSubClass':'SaleCondition']))

#log transform the target:
raw_train_boost["SalePrice"] = np.log1p(raw_train_boost["SalePrice"])

#log transform skewed numeric features:
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

log_feats = raw_train_boost[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
log_feats = log_feats[log_feats > 0.75]
log_feats = log_feats.index

all_data[log_feats] = np.log1p(all_data[log_feats])

all_data = pd.get_dummies(all_data)

#filling NA's with the mean of the column:
all_data = all_data.fillna(all_data.mean())

#creating matrices for sklearn:
X_train_b = all_data[:raw_train_boost.shape[0]]
X_test_b = all_data[raw_train_boost.shape[0]:]
y_b = raw_train_boost.SalePrice
def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train_b, y_b, scoring="neg_mean_squared_error", cv = 10))
    return(rmse)
model_ridge=RidgeCV().fit(X_train_b,y_b);
rmse_cv(model_ridge).mean();
import xgboost as xgb
dtrain = xgb.DMatrix(X_train_b, label = y_b)
dtest = xgb.DMatrix(X_test_b)
params = {"max_depth":3, "eta":0.1}
model = xgb.cv(params, dtrain,  num_boost_round=600, early_stopping_rounds=100)
model_xgb = xgb.XGBRegressor(n_estimators=500, max_depth=3, learning_rate=0.01)
model_xgb.fit(X_train_b, y_b)
xgb_preds = np.expm1(model_xgb.predict(X_test_b))
lasso_preds = np.expm1(model_ridge.predict(X_test_b))
predictions = pd.DataFrame({"xgb":xgb_preds, "lasso":lasso_preds})
preds = 0.8*lasso_preds + 0.2*xgb_preds
solution = pd.DataFrame({"id":raw_test_boost.Id, "SalePrice":preds})
solution.to_csv("ridge_sol.csv", index = False)
###########################BOOSTING ENDS##################################################################

raw_train = pd.read_csv('processed_train.csv')
raw_test = pd.read_csv('processed_test.csv')
#print(raw_test.head);
raw_train = raw_train.replace(to_replace='-inf',value=0)
raw_train = raw_train.replace(to_replace='inf',value=0)
#remeber to remove it , here nan is Infinity
raw_train = raw_train.apply(lambda x:x.fillna(x.mean()))
raw_test = raw_test.replace(to_replace='-inf',value=0)
raw_test = raw_test.replace(to_replace='inf',value=0)
raw_test = raw_test.apply(lambda x:x.fillna(x.mean()))


#print (raw_train)
'''
cat_list = []
for i in raw_train.columns:
    if raw_train[i].dtype == object:
        raw_train[i] = raw_train[i].astype('category')
        cat_list += [i]


#print (cat_list)

number = LabelEncoder()
for i in cat_list:
    raw_train[i]=number.fit_transform(raw_train[i].astype('str'))
'''   
cat_list = []
for i in raw_test.columns:
    if raw_test[i].dtype == object:
        raw_test[i] = raw_test[i].astype('category')
        cat_list += [i]




#print (cat_list)

number = LabelEncoder()
for i in cat_list:
    raw_test[i]=number.fit_transform(raw_test[i].astype('str'))
    

X_train= raw_train.loc[:,'OverallQual':'YearRemodAdd'];

X_test= raw_test.loc[:,'OverallQual':'YearRemodAdd'];
                  
raw_train['SalePrice'].describe()


Y_train=raw_train.SalePrice


############################################ Ridge ######################################
## 0.9316
X1 = np.array(X_train)
Y1 = np.array(Y_train)
X_test1 = np.array(X_test)
reg = Ridge(alpha = 225)
reg.fit(X1,Y1);
op=reg.predict(X_test1);

X_train, X_test, y_train, y_test = train_test_split(X1, Y1, test_size=0.33, random_state=42) #dividing trainig data into train and test            
yo1 = pd.DataFrame(columns=["Id","SalePrice"])
yo1["Id"]=raw_test.Id.values
yo1["SalePrice"]=op

yo1.to_csv('predictions.csv',index=False)
print (op);

########################################### Ridge #######################################
########################################## Ridge cross validation ############################
######## 0.55
def model_generation(X_train,y_train,model):   
    kf=KFold(X_train.shape[0], n_folds=8)
    rf = Ridge(alpha = 0.1)
    if(model=="Lasso"):
        rf = Lasso(alpha = 0.1)
    if(model=="Neural"):
        rf = MLPRegressor(hidden_layer_sizes=(100, ), activation='relu',learning_rate='constant', learning_rate_init=0.01, max_iter=1000)

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

yo1.to_csv('predictions8.csv',index=False)
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

yo1.to_csv('predictions1.csv',index=False)
print (op);
########################################### Lasso #######################################
########################################### Lasso Cross Validation ######################
#0.57
rf=model_generation(X_train , y_train,"Lasso")  
op=rf.predict(X_test1)
         
yo1 = pd.DataFrame(columns=["Id","SalePrice"])
yo1["Id"]=raw_test.Id.values
yo1["SalePrice"]=op

yo1.to_csv('predictions9.csv',index=False)
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

yo1.to_csv('predictions2.csv',index=False)
print (op);
########################################## Neural Network ###############################
########################################## Neural Network Cross Validation ##############
#0.233
rf=model_generation(X_train , y_train,"Neural")  
op=rf.predict(X_test1)
         
yo1 = pd.DataFrame(columns=["Id","SalePrice"])
yo1["Id"]=raw_test.Id.values
yo1["SalePrice"]=op

yo1.to_csv('predictions10.csv',index=False)
print ("k-fold_validation Neural" ,op);



########################################## Neural Network Cross Validation ##############
######################################### Kernelized ridge regression ###################

reg = KernelRidge(alpha=1, kernel='linear', gamma=None, degree=3, coef0=1, kernel_params=None)
reg.fit(X1, Y1)
op=reg.predict(X_test1);

              
yo1 = pd.DataFrame(columns=["Id","SalePrice"])
yo1["Id"]=raw_test.Id.values
yo1["SalePrice"]=op

yo1.to_csv('predictions3.csv',index=False)
print (op);
#########################################  Kernelized ridge regression #####################



######################################### SVM ##############################################
'''
reg = svm.SVR(kernel='linear', C=1e3) # rbf, linear 
#print (Y1)
reg.fit(X1, Y1) 
op=reg.predict(X_test1);

              
yo1 = pd.DataFrame(columns=["Id","SalePrice"])
yo1["Id"]=raw_test.Id.values
yo1["SalePrice"]=op

yo1.to_csv('predictions4.csv',index=False)
print (op);
'''
######################################## SVM ################################################

######################################## Partial Least Square ###############################

reg = PLSRegression()
reg.fit(X1, Y1) 
op=reg.predict(X_test1);

              
yo1 = pd.DataFrame(columns=["Id","SalePrice"])
yo1["Id"]=raw_test.Id.values
yo1["SalePrice"]=op

yo1.to_csv('predictions7.csv',index=False)
print (op);

######################################## Partial Least Square ###############################

######################################## stocastic gradient decent ##########################
# for more than 10000 sample
'''
reg = SGDRegressor(alpha=0.001,n_iter=1000);
reg.fit(X1, Y1) 
op=reg.predict(X_test1);

              
yo1 = pd.DataFrame(columns=["Id","SalePrice"])
yo1["Id"]=raw_test.Id.values
yo1["SalePrice"]=op

yo1.to_csv('predictions5.csv',index=False)
print (op);
print (X1.shape)
'''
####################################### stocastic gradient decent ###########################

###################################### Baysian regression ##################################
reg = linear_model.BayesianRidge()
reg.fit(X1, Y1) 
op=reg.predict(X_test1);

              
yo1 = pd.DataFrame(columns=["Id","SalePrice"])
yo1["Id"]=raw_test.Id.values
yo1["SalePrice"]=op

yo1.to_csv('predictions6.csv',index=False)
print (op);

###################################### Baysian regression ##################################


























