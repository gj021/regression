import pickle
import numpy as np
import matplotlib.pyplot as plt 

f=open("data.pkl",'rb')
data=pickle.load(f)
data = np.hstack((np.ones((data.shape[0], 1)), data))

class stochasticGradient: 
    def __init__(self,data): 
        size_train=int(data.shape[0]*0.7)
        size_test=int(data.shape[0]*0.3)
        self.x_train,self.x_test,self.y_train,self.y_test=data[:size_train,0:3],data[:size_test,0:3],data[:size_train,3:4],data[:size_test,3:4]
    
    def intialize(self,w,learning_rate,stopping_crieteria):
        self.w=theta 
        self.learning_rate=learning_rate
        self.stopping_crieteria=stopping_crieteria
    
    def cost(self):
        hypothesis=np.dot(self.x_train,self.w)
        J = np.dot((hypothesis - self.y_train).transpose(), (hypothesis - self.y_train)) 
        J /= 2
        J=np.squeeze(np.asarray(J))
        return J

    def gradient(self):
        size=self.x_train.shape
        error_list=[]
        for i in range(100000): 
            hypothesis=np.dot(self.x_train[i],self.w)
            hypothesis=hypothesis-self.y_train[i]
            X=np.asmatrix(self.x_train[i])
            grad=np.dot(hypothesis,X)
            c=self.cost()
            error_list.append(c)
            print(c,i)
            self.w=self.w-self.learning_rate*grad.transpose()
        return error_list

p=stochasticGradient(data) 
theta = np.zeros((p.x_train.shape[1], 1))
theta[0]=5
theta[1]=15
theta[2]=25
p.intialize(theta,0.000001,0.00001)
error_list=p.gradient()

plt.plot(error_list) 
plt.xlabel("Number of iterations") 
plt.ylabel("Cost") 
plt.show()