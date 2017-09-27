from scipy import io
from scipy.stats import mode
from heapq import nsmallest
import numpy as np
from matplotlib import pyplot as plt

Matrix = io.loadmat('C:\Users\lhren\Desktop\Fall 2017/4771\hw\hw1data.mat')
X_Y = np.concatenate((Matrix['X'],Matrix['Y']),axis=1)
np.random.shuffle(X_Y)
train_data = X_Y[0:7000,:]
test_data = X_Y[7000:10000,:]
train_X = train_data[:,0:784]
train_Y = train_data[:,784]
test_X = test_data[:,0:784]
test_Y = test_data[:,784]
X = train_data
##construct kd tree

n = len(X)
k = int(np.sqrt(n))
d = len(X[0]) - 1
def split_node(X,y,feature,d=784):
 
    thres = np.median(X[:,feature])     
            
    x_L = X[X[:,feature]<=thres]
    x_R = X[X[:,feature]>thres]
    y_L = x_L[:,d]
    y_R = x_R[:,d]
    if len(y_L)==0 or len(y_R)==0:
        x_L = X
        x_R = X
        y_L = y
        y_R = y
    return x_L,x_R,y_L,y_R,thres
K = 1024
max_depth = int(np.log2(K))
num_pop = K-1
DT_node = []
end_class = []
y = X[:,d]
DT_cur = [[X,y]]
    
for i in range(num_pop):
    X,y = DT_cur.pop(0)
    feature = np.random.randint(783)
    x_L,x_R,y_L,y_R,thres = split_node(X,y,feature,d)
        #print opt_f, opt_thres
    DT_node.append([feature,thres])
    DT_cur.append([x_L,y_L])
    DT_cur.append([x_R,y_R])
count_correct = 0
count = 0
for x in test_data:
    count+=1
    print count
    dist = []
    i = 0
    for k in range(max_depth):
            split = DT_node[i]
            f = split[0]
            t = split[1]
            if x[f]<=t:
                i = 2*i + 1
            else:
                i = 2*i + 2
    cell_x = DT_cur[i-num_pop][0]
    cell_y = DT_cur[i-num_pop][1]
    m = len(cell_y)
    k = int(np.sqrt(m))
    for t in range(m):
        x_i = cell_x[t]
        
        distance = sum((x_i[:d]-x[:d])**2)
        dist.append([np.sqrt(distance),int(x_i[d])])
    
    k_neighbor = np.array(nsmallest(k,dist))[:,1]
    most_frequent = int(mode(k_neighbor)[0][0])
    if(x[d]==int(most_frequent)):
        count_correct+=1
print('Error rate: '+str(1.0*count_correct/len(test_data)))
