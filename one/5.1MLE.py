import pandas as pd
import numpy as np
import scipy.io
import scipy as sp
from sklearn.model_selection import train_test_split
import math
from scipy.stats import multivariate_normal
# import data
mat = scipy.io.loadmat('hw1data.mat')
df = pd.DataFrame(mat['X'], mat['Y'])
df_clean = df.loc[:, (df.sum(axis=0) > 0)]
df.shape
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
    train_sub_cov[i] = train_sub_cov[i]/len(train_sub_cov[i])
    train_sub_cov[i] = train_sub_cov[i][200:600,200:600]
    boolean[i] = train_sub_cov[i] != 0
    train_sub_cov[i] = train_sub_cov[i][np.nonzero(train_sub_cov[i])]
    n =int(math.sqrt(len(train_sub_cov[i])))
    train_sub_cov[i] = train_sub_cov[i].reshape((n,n))+np.identity(n)
    model[i] = multivariate_normal(mean=um[i][boolean[i][0,:]], cov=train_sub_cov[i])
    print(len(um[i][boolean[i][0,:]]))
    print(len(train_sub_cov[i]))
#predict based on train data
m_train = train.as_matrix()
predict = []
for i in range(8000):
    max_proj = 0
    for j in range(10):
        tmp = m_train[i]
        tmp = tmp[200:600]
        tmpj = tmp[boolean[j][0,:]]
        prob = model[j].pdf(tmpj)
        print(prob)
        if prob > max_proj:
            max_proj = prob
            predict[i] = j