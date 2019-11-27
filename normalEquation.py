import pickle
import numpy

import pickle
import numpy as np
import matplotlib.pyplot as plt 

f=open("data_copy.pkl",'rb')
data=pickle.load(f)
data = np.hstack((np.ones((data.shape[0], 1)), data))

size_train=int(data.shape[0]*0.7)
size_test=int(data.shape[0]-size_train)
x_train,x_test,y_train,y_test=data[:size_train,0:2],data[:size_test,0:2],data[:size_train,2:3],data[:size_test,2:3]

X=np.asmatrix(x_train)
Y=np.asmatrix(y_train)

Q=np.linalg.inv(np.dot(X.transpose(),X))

theta=np.dot(Q,np.dot(X.transpose(),Y))

hypothesis=np.dot(x_train,theta)
J = np.dot((hypothesis - y_train).transpose(), (hypothesis - y_train)) 
J /= 2
J=np.squeeze(np.asarray(J))
print("Training Error:"+str(J))

hypothesis=np.dot(x_test,theta)
J = np.dot((hypothesis - y_test).transpose(), (hypothesis - y_test))
size=x_train.shape
J=J/size[0]
J=np.squeeze(np.asarray(J))
print("Rmse:"+str(J))

hypothesis=np.dot(x_test,theta)
J = np.dot((hypothesis - y_test).transpose(), (hypothesis - y_test))
SSE=np.sum(J)
mean_test=np.mean(y_test)
y_test=y_test-mean_test
y_test=np.square(y_test)
TSE=np.sum(y_test)
print("R2:"+str(1-SSE/TSE))