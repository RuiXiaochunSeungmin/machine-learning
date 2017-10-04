import pandas as pd
import numpy as np
import scipy.io
import scipy as sp
from scipy.stats import multivariate_normal
import math

# Import data
mat = scipy.io.loadmat('hw1data.mat')
df = pd.DataFrame(mat['X'], mat['Y'])

# Setting std threshold 
f_std_threshold = [1, 2, 5,10,20, 40, 60, 80, 100, 110]

# Helpper method normalize
def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        mean_value = df[feature_name].mean()
        std_value = df[feature_name].std()
        result[feature_name] = (df[feature_name] - mean_value) / std_value
    return result

# For every std threshold, filter columns, build model and predict
for s in f_std_threshold:
    df_clean = df.loc[:, (df.std(axis=0) > s)]
    #Normalize every columns
    df_clean = normalize(df_clean)
    print("Std threshold is %(std)d, feature num is %(shape1)d"%{"std":s,"shape1":df_clean.shape[1]})
    #split train and test data
    train, test = train_test_split(df_clean, test_size=0.2, random_state=42)
    #calculate mean vector
    u = train.groupby(train.index).mean()
    um = u.as_matrix()
    #calculate covarience matrix and build model
    train_sub ={}
    for i in range(10):
        train_sub[i] = train[train.index == i].transpose()
    train_sub_cov = {}
    model = {}
    boolean = {}
    for i in range(10):
        train_sub_cov[i] = np.cov(train_sub[i])
        n =len(train_sub_cov[i])
        train_sub_cov[i] = train_sub_cov[i].reshape((n,n))+0.3*np.identity(n)
        model[i] = multivariate_normal(mean=um[i], cov=train_sub_cov[i])
    #predict based on train data
    m_train = train.as_matrix()
    predict = [None] * 8000
    for i in range(8000):
        max_proj = 0
        for j in range(10):
            #print("i is %(i)d and j is %(j)d" % {"i": i, "j":j})
            tmp = m_train[i]
            prob = model[j].pdf(tmp)
            #print(prob)
            if prob > max_proj:
                max_proj = prob
                predict[i] = j
    train_sum = 0
    train_r = train.reset_index()
    for i in range(8000):
        if predict[i] == int(train_r.iloc[i]['index']):
            train_sum += 1
    print("Std threshold is %(std)d and accuracy on train data is %(train_sum)f%%" % {"std":s,"train_sum": train_sum/8000*100})
    
    test_predict = [None] * 2000
    m_test = test.as_matrix()
    test_r = test.reset_index()
    for i in range(2000):
        max_proj_t = 0
        for j in range(10):
            #print("i is %(i)d and j is %(j)d" % {"i": i, "j":j})
            tmp_t = m_test[i]
            prob_t = model[j].pdf(tmp_t)
            #print(prob_t)
            if prob_t > max_proj_t:
                max_proj_t = prob_t
                test_predict[i] = j
    test_sum = 0
    for i in range(2000):
        if predict[i] == int(test_r.iloc[i]['index']):
            test_sum += 1
    print("Std threshold is %(std)d and total correct number on test data is %(test_sum)f%%" % {"std":s,"test_sum": test_sum/2000*100})
