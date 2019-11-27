import pickle
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import time

f=open("data.pkl",'rb')
data=pickle.load(f)

npdata = np.hstack((np.ones((data.shape[0], 1)), data))
error_list = []
# print(npdata)
# prfloat(data)

class gradient_descent:
    def __init__(self,npdata):
        '''
        Breaking the data set into training, testing and validation
        '''
        self.size_train = int(npdata.shape[0]*0.7)
        self.size_test = int(npdata.shape[0]*0.25)
        self.size_val = npdata.shape[0] - self.size_train - self.size_test
        self.x_train,self.y_train,self.t_train = npdata[:self.size_train,1:2],npdata[:self.size_train,2:3],npdata[:self.size_train,3:4]
        self.x_val = npdata[self.size_train:self.size_train+self.size_val,1:2]
        self.y_val = npdata[self.size_train:self.size_train+self.size_val,2:3]
        self.t_val = npdata[self.size_train:self.size_train+self.size_val,3:4]
        self.x_test = npdata[self.size_train+self.size_val:,1:2]
        self.y_test = npdata[self.size_train+self.size_val:,2:3]
        self.t_test = npdata[self.size_train+self.size_val:,3:4]
        # print("SFF")
        # print(np.sum(np.square(self.x_train)))
        # print((np.dot(self.x_train.T,self.x_train)))

        '''
        Taking input
        '''
        self.w0_init = float(input("enter w0:"))
        self.w1_init = float(input("enter w1:"))
        self.w2_init = float(input("enter w2:"))
        self.w0 = self.w0_init
        self.w1 = self.w1_init
        self.w2 = self.w2_init
        self.w = [self.w0,self.w1,self.w2]
        self.w = np.array(self.w)
        self.learning_rate = float(input("enter learning_rate:"))
        self.stoping_constant = float(input("enter stoping constant:"))
        self.start = time.time()
        print(self.start)
    
    '''
    gradient function for w0
    '''
    def sumw0(self):
        error_g = self.w0*self.x_train.shape[0]
        error_g += self.w1*np.sum(self.x_train)
        error_g += self.w2*np.sum(self.y_train)
        error_g -= np.sum(self.t_train)
        return error_g
    
    '''
    gradient function for w1
    '''
    def sumw1(self):
        error_g = self.w0*np.sum(self.x_train)
        error_g += self.w1*np.sum(np.square(self.x_train))
        error_g += self.w2*np.sum(np.dot(self.x_train.T,self.y_train))
        error_g -= np.sum(np.dot(self.t_train.T,self.x_train))
        return error_g
    
    '''
    gradient function for w2
    '''
    def sumw2(self):
        error_g = self.w0*np.sum(self.y_train)
        error_g += self.w1*np.sum(np.dot(self.x_train.T,self.y_train))
        error_g += self.w2*np.sum(np.dot(self.y_train.T,self.y_train))
        error_g -= np.sum(np.dot(self.t_train.T,self.y_train))
        return error_g

    '''
    Loss function 
    '''
    def error_fn(self):
        X = np.dot(self.x_train,self.w1)
        Y = np.dot(self.w2,self.y_train)
        Z = np.dot(-1,self.t_train)
        E = np.add(X,self.w0)
        E = np.add(Y,E)
        E = np.add(Z,E)
        error = np.sum(np.square(E))
        return error/2

    '''
    Regression without regularization
    '''
    def w_approximation(self):
        self.w0 = self.w0_init
        self.w1 = self.w1_init
        self.w2 = self.w2_init
        gradient0 = self.sumw0()
        gradient1 = self.sumw1()
        gradient2 = self.sumw2()
        previous_error = 0
        error = self.error_fn()
        it_no = 0

        while(abs(previous_error-error) > self.stoping_constant):
            it_no += 1
            if(it_no%20 == 0):
                it_no = 0
            self.w0 = self.w0 - gradient0*self.learning_rate
            self.w1 = self.w1 - gradient1*self.learning_rate
            self.w2 = self.w2 - gradient2*self.learning_rate

            if(it_no == 0):
                print(self.w0, self.w1, self.w2, error)
                error_list.append(error)
                # print(gradient0,gradient1,gradient2)

            previous_error = error
            gradient0 = self.sumw0()
            gradient1 = self.sumw1()
            gradient2 = self.sumw2()
            error = self.error_fn()
        
        print(self.w0, self.w1, self.w2, error)
        return

    '''
    Regression with l1 regularization
    '''
    def w_approximation_regularizationl1(self,lambda_coefficient):

        self.w0 = self.w0_init
        self.w1 = self.w1_init
        self.w2 = self.w2_init
        it_no = 0
        # lambda_coefficient = 100#float(input("Enter lamda:"))
        if(self.w0 > 0):
            gradient0 = self.sumw0() + lambda_coefficient
        else:
            gradient0 = self.sumw0() - lambda_coefficient
        if(self.w1 > 0):
            gradient1 = self.sumw1() + lambda_coefficient
        else:
            gradient1 = self.sumw1() - lambda_coefficient
        if(self.w2 > 0):
            gradient2 = self.sumw2() + lambda_coefficient
        else:
            gradient2 = self.sumw2() - lambda_coefficient
        
        reg_factor = abs(self.w0) + abs(self.w1) + abs(self.w2)
        reg_factor = reg_factor * lambda_coefficient

        error = self.error_fn() + reg_factor
        previous_error = 0
        while(abs(previous_error-error) > self.stoping_constant):
            it_no += 1
            if(it_no%20 == 0):
                it_no = 0
            self.w0 = self.w0 - gradient0*self.learning_rate
            self.w1 = self.w1 - gradient1*self.learning_rate
            self.w2 = self.w2 - gradient2*self.learning_rate

            gradient0 = self.sumw0() + lambda_coefficient
            gradient1 = self.sumw1() + lambda_coefficient
            gradient2 = self.sumw2() + lambda_coefficient

            reg_factor = abs(self.w0) + abs(self.w1) + abs(self.w2)
            reg_factor = reg_factor * lambda_coefficient

            previous_error = error
            error = self.error_fn() + reg_factor

            if(it_no == 0):
                print(self.w0, self.w1, self.w2, error)
                error_list.append(error)

        print(self.w0, self.w1, self.w2, error)
        return

    '''
    Regression with l2 regularization
    '''
    def w_approximation_regularizationl2(self,lambda_coefficient):
        
        self.w0 = self.w0_init
        self.w1 = self.w1_init
        self.w2 = self.w2_init
        it_no = 0
        # lambda_coefficient = 0.0001#float(input("Enter lamda:"))

        gradient0 = self.sumw0() + 2*lambda_coefficient*self.w0
        gradient1 = self.sumw1() + 2*lambda_coefficient*self.w1
        gradient2 = self.sumw2() + 2*lambda_coefficient*self.w2#*self.data.shape[0]

        ref_factor = self.w0*self.w0 + self.w1*self.w1 + self.w2*self.w2
        ref_factor = ref_factor*lambda_coefficient
        previous_error = 0
        error = self.error_fn() + ref_factor

        while(abs(previous_error-error) > self.stoping_constant):
            it_no += 1
            if(it_no%20 == 0):
                it_no = 0
            self.w0 = self.w0 - gradient0*self.learning_rate
            self.w1 = self.w1 - gradient1*self.learning_rate
            self.w2 = self.w2 - gradient2*self.learning_rate

            gradient0 = self.sumw0() + 2*lambda_coefficient*self.w0
            gradient1 = self.sumw1() + 2*lambda_coefficient*self.w1
            gradient2 = self.sumw2() + 2*lambda_coefficient*self.w2

            previous_error = error
            ref_factor = self.w0*self.w0 + self.w1*self.w1 + self.w2*self.w2
            ref_factor = ref_factor * lambda_coefficient
            error = self.error_fn() + ref_factor
            if(it_no == 0):
                print(self.w0, self.w1, self.w2, error)
                error_list.append(error)

        error_list.append(error)
        print(self.w0, self.w1, self.w2, error)
        return
    
    '''
    Function to calculate R square and RMSE errors 
    '''
    def r2_rmse(self,x,y,t,w0,w1,w2):
        rmse = np.sqrt(np.mean(np.square(w0 + w1*x + w2*y - t))) 
        mse = rmse*rmse
        SSR = mse*x.shape[0]
        SST = np.sum(np.square(t - np.mean(t)))
        print("SSR = ",SSR)
        print("SST = ",SST)
        R2 = 1-SSR/SST
        
        return R2,rmse
        
'''
call the gradient_descent constructor
'''
reg = gradient_descent(npdata)

'''
This segment is used for regression without regularization
'''
# reg.w_approximation()
# R2,RMSE = reg.r2_rmse(reg.x_val,reg.y_val,reg.t_val,reg.w0,reg.w1,reg.w2)
# print("R2:" + str(R2))
# print("RMSE:" + str(RMSE))


'''
This segment is used for lamda approximation
'''
lamda_list = []
for i in range(0,20):
    lamda_list.append(i*100)
mse_list_l1 = []
mse_list_l2 = []

'''
Lamda approximation with l1 regularization
'''
# for i in lamda_list:
#     reg.w_approximation_regularizationl1(i)
#     R2,RMSE = reg.r2_rmse(reg.x_val,reg.y_val,reg.t_val,reg.w0,reg.w1,reg.w2)
#     mse_list_l1.append(RMSE*RMSE)
#     print("R2:" + str(R2))
#     print("RMSE:" + str(RMSE))



# plt.plot(lamda_list,mse_list_l1)
# plt.xlabel("lamda")
# plt.ylabel("validation loss l1")
# plt.show()

'''
Lamda approximation with l2 regularization
'''
# for i in lamda_list:
#     reg.w_approximation_regularizationl2(i)
#     R2,RMSE = reg.r2_rmse(reg.x_val,reg.y_val,reg.t_val,reg.w0,reg.w1,reg.w2)
#     mse_list_l2.append(RMSE*RMSE)
#     print("R2:" + str(R2))
#     print("RMSE:" + str(RMSE))
# plt.plot(lamda_list,mse_list_l2)
# plt.xlabel("lamda")
# plt.ylabel("validation loss l2")
# plt.show()

'''
The best lamda from above approximations
'''
lamda_best_l1 = 1000
lamda_best_l2 = 1000

'''
R square and RMSE error with best lamda for l1 regularization
'''
# reg.w_approximation_regularizationl1(lamda_best_l1)
# R2,RMSE = reg.r2_rmse(reg.x_val,reg.y_val,reg.t_val,reg.w0,reg.w1,reg.w2)
# print("R2:" + str(R2))
# print("RMSE:" + str(RMSE))

'''
R square and RMSE error with best lamda for l2 regularization
'''
reg.w_approximation_regularizationl2(lamda_best_l2)
R2,RMSE = reg.r2_rmse(reg.x_val,reg.y_val,reg.t_val,reg.w0,reg.w1,reg.w2)
print("R2:" + str(R2))
print("RMSE:" + str(RMSE))

reg.end = time.time()
print(reg.end - reg.start)
'''
To plot error for any regression training(every 20 iterations)
'''
plt.plot(error_list) 
plt.xlabel("Number of iterations") 
plt.ylabel("Error") 
plt.show()
plt.xlabel("Number of iterations")
plt.ylabel("Error")



'''
Tests
'''
##w_app
# enter w0:1
# enter w1:1
# enter w2:1
# enter learning_rate:.0000008
# enter stoping constant:.1
# r2 = -0.02
# rmse = 0.147
#0.19618696325522425 0.10264752240580392 -0.07889452237017956 2573.6160546720807

##w_app_regl1
# enter w0:1
# enter w1:1
# enter w2:1
# enter learning_rate:.000000001
# enter stoping constant:1
#lamda = 100
#-0.10150549278495591 0.3378753461702664 0.3835193525844637 12311.888723726139

##w_app_regl2
# enter w0:1
# enter w1:1
# enter w2:1
# enter learning_rate:.0000008
# enter stoping constant:.1
# lamda = 100
# 0.19577889097266765 0.10148517441371772 -0.07681529190027742 2579.588042326759

