from scipy import io
from scipy.stats import mode
import numpy as np
from matplotlib import pyplot as plt

Matrix = io.loadmat('C:\Users\lhren\Desktop\Fall 2017/4771\hw\hw1data.mat')
X_Y = np.concatenate((Matrix['X'],Matrix['Y']),axis=1)

np.random.shuffle(X_Y)
##train_test split
train_data = X_Y[0:7000,:]
test_data = X_Y[7000:10000,:]
train_X = train_data[:,0:784]
train_Y = train_data[:,784]
test_X = test_data[:,0:784]
test_Y = test_data[:,784]
X = train_data
n = len(X)
k = int(np.sqrt(n))
d = len(X[0]) - 1
f_list = np.linspace(100,750,14)
#function to minimize uncertainty in a node
def entropy(y):
    m = len(y)
    count = np.zeros(10)
    H = 0
    for i in range(10):
        count[i] = len(y[y==i])
        p = 1.0*count[i]/m
        if(p==0):
            H+=0
        else:
            H += -p*np.log(p)
    return H

def split_node(X,y,d=784):
    H_old = entropy(y)
    opt_thres = 0
    H = H_old
    opt_f = 0
    #choose from a list of popular features and thresholds based on observations
    for feature in f_list:
        for thres in [0,4,8,16,32,64,128,160,200]:     
            x_L = X[X[:,feature]<=thres]
            #print x_L
            x_R = X[X[:,feature]>thres]
            #print x_R
            y_L = x_L[:,d]
            y_R = x_R[:,d]
           
            H_new = entropy(y_L)*1.0*len(x_L)/n + entropy(y_R)*1.0*len(x_R)/n
            #test entropy decrease to find best feature and threshold
            if H_new<H:
                H = H_new
                opt_thres = thres
                opt_f = feature
    x_L = X[X[:,opt_f]<=opt_thres]
    x_R = X[X[:,opt_f]>opt_thres]
    y_L = x_L[:,d]
    y_R = x_R[:,d]
    #This is equivalent to stopping a node from further splitting due to insufficient data points
    if len(y_L)==0 or len(y_R)==0:
        x_L = X
        x_R = X
        y_L = y
        y_R = y
    return x_L,x_R,y_L,y_R,opt_f,opt_thres
#test against 2^depth hyperparameter
K_list = [8,16,32,64,128,256,512,1024,2048]
depth = []
train_E=[]
test_E=[]
for K in K_list:
    #print K
    X = train_data
    n = len(X)
    k = int(np.sqrt(n))
    d = len(X[0]) - 1
    f_list = np.linspace(0,750,16)
    max_depth = int(np.log2(K))
    depth.append(max_depth)
    num_pop = K-1
    DT_node = []
    end_class = []
    y = X[:,d]
    #initialize
    DT_cur = [[X,y]]
    
    for i in range(num_pop):
        X,y = DT_cur.pop(0)
        #optimal split
        x_L,x_R,y_L,y_R,opt_f,opt_thres = split_node(X,y,d)
        #print opt_f, opt_thres
        DT_node.append([opt_f,opt_thres])
        DT_cur.append([x_L,y_L])
        DT_cur.append([x_R,y_R])
    for node in DT_cur:
        #label each leaf node at the end of model
        class_label = int(mode(node[1])[0][0])
        end_class.append(class_label)
    
    error_train = 0
    for x in train_data:
        i = 0
        for j in range(max_depth):
            split = DT_node[i]
            f = split[0]
            t = split[1]
            if x[f]<=t:
                i = 2*i + 1
            else:
                i = 2*i + 2
        #locate data point in a leaf node
        class_pred = end_class[i-num_pop]
        if class_pred!=x[d]:
            error_train+=1
    train_E.append(1.0*error_train/len(train_data))  
    error_test=0
    for x in test_data:
        i = 0
        for j in range(max_depth):
            split = DT_node[i]
            f = split[0]
            t = split[1]
            if x[f]<=t:
                i = 2*i + 1
            else:
                i = 2*i + 2
        class_pred = end_class[i-num_pop]
        if class_pred!=x[d]:
            error_test+=1
    test_E.append(1.0*error_test/len(test_data))  
#compare results
plt.plot(depth,train_E,'b',label='train')
plt.plot(depth,test_E,'r',label='test')
plt.legend()
plt.show()
