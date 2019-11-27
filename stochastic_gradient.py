import pickle
import numpy as np
import matplotlib.pyplot as plt 

f=open("data_copy.pkl",'rb')
data=pickle.load(f)
data = (data - np.min(data,axis=0))/np.ptp(data,axis=0)
data = np.hstack((np.ones((data.shape[0], 1)), data))


class stochasticGradient: 
    def __init__(self,data): 
        size_train=int(data.shape[0]*0.7)
        size_test=int(data.shape[0]*0.3)
        self.x_train,self.x_test,self.y_train,self.y_test=data[:size_train,0:3],data[:size_test,0:3],data[:size_train,3:4],data[:size_test,3:4]
    
    def intialize(self,w,learning_rate,stopping_crieteria):
        """
            Intialization of w,learning rate,stopping crieteria
        """
        self.w=theta 
        self.learning_rate=learning_rate
        self.stopping_crieteria=stopping_crieteria
    
    def cost(self):
        """
            To calculate Cost of the model
        """
        hypothesis=np.dot(self.x_train,self.w)
        J = np.dot((hypothesis - self.y_train).transpose(), (hypothesis - self.y_train)) 
        J /= 2
        J=np.squeeze(np.asarray(J))
        return J

    def gradient(self):
        """
            Stochastic gradient applied to the model
            Take sequential value everytime
        """
        error_list=[]
        for i in range(10000): 
            hypothesis=np.dot(self.x_train[i],self.w)
            hypothesis=hypothesis-self.y_train[i]
            X=np.asmatrix(self.x_train[i])
            grad=np.dot(hypothesis,X)
            c=self.cost()
            self.w=self.w-self.learning_rate*grad.transpose()
            if(i!=0 and abs(c-error_list[-1])<self.stopping_crieteria):
                break;
            error_list.append(c)
            print(c,i)
        return error_list

    def measure(self):
        """
            To calculate RMSE value
        """
        hypothesis=np.dot(self.x_test,self.w)
        J = np.dot((hypothesis - self.y_test).transpose(), (hypothesis - self.y_test))
        size=self.x_train.shape
        print(size)
        J=J/size[0]
        J=np.squeeze(np.asarray(J))
        return J**0.5

p=stochasticGradient(data) 
theta = np.zeros((p.x_train.shape[1], 1))
theta[0]=1
theta[1]=1
theta[2]=1
p.intialize(theta,0.005,0.001)
error_list=p.gradient()

"""
    To calculate R2 value of the model
"""
hypothesis=np.dot(p.x_test,p.w)
J = np.dot((hypothesis - p.y_test).transpose(), (hypothesis - p.y_test))
SSE=np.sum(J)

mean_test=np.mean(p.y_test)
p.y_test=p.y_test-mean_test
p.y_test=np.square(p.y_test)
TSE=np.sum(p.y_test)

print("RMSE:")
print(p.measure())

print("R2:")
print(1-SSE/TSE)

plt.plot(error_list[20:]) 
plt.xlabel("Number of iterations") 
plt.ylabel("Cost") 
plt.show()