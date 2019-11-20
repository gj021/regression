import pickle
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

f=open("data.pkl",'rb')
data=pickle.load(f)

npdata = np.hstack((np.ones((data.shape[0], 1)), data))
error_list = []
print(npdata)
# prfloat(data)

class gradient_descent:
    def __init__(self,npdata):
        
        self.size_train = int(npdata.shape[0]*0.7)
        # self.size_test = npdata.shape[0] - size_train
        self.x_train,self.y_train,self.t_train = npdata[:self.size_train,1:2],npdata[:self.size_train,2:3],npdata[:self.size_train,3:4]
        self.w0 = float(input("enter w0:"))
        self.w1 = float(input("enter w1:"))
        self.w2 = float(input("enter w2:"))
        
        self.learning_rate = float(input("enter learning_rate:"))
        
        self.stoping_constant = float(input("enter stoping constant:"))

    def sumw0(self):
        error_g = self.w0*npdata.shape[0]
        error_g += self.w1*np.sum(self.x_train)
        error_g += self.w2*np.sum(self.y_train)
        error_g -= np.sum(self.t_train)
        return 2*error_g

    def sumw1(self):
        error_g = self.w0*np.sum(self.x_train)
        error_g += self.w1*np.sum(np.dot(self.x_train.T,self.x_train))
        error_g += self.w2*np.sum(np.dot(self.x_train.T,self.y_train))
        error_g -= np.sum(np.dot(self.t_train.T,self.x_train))
        return 2*error_g
    
    def sumw2(self):
        error_g = self.w0*np.sum(self.y_train)
        error_g += self.w1*np.sum(np.dot(self.x_train.T,self.y_train))
        error_g += self.w2*np.sum(np.dot(self.y_train.T,self.y_train))
        error_g -= np.sum(np.dot(self.t_train.T,self.y_train))
        return 2*error_g

    def error_fn(self):
        X = np.dot(self.x_train,self.w1)
        Y = np.dot(self.w2,self.y_train)
        Z = np.dot(-1,self.t_train)
        E = np.add(X,self.w0)
        E = np.add(Y,E)
        E = np.add(Z,E)
        error = np.sum(np.dot(E.T,E))
        return error

    #matrix operation and updating w without regularization
    def w_approximation(self):
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

    #l1 regularization
    def w_approximation_regularizationl1(self):
        
        it_no = 0
        lambda_coefficient = 100#float(input("Enter lamda:"))
        
        gradient0 = self.sumw0() + lambda_coefficient
        gradient1 = self.sumw1() + lambda_coefficient
        gradient2 = self.sumw2() + lambda_coefficient
        
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

    #l2 regularization
    def w_approximation_regularizationl2(self):
        it_no = 0
        lambda_coefficient = 0.0001#float(input("Enter lamda:"))

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
    
    def rmse(self):
        return error_list[-1]

        

reg = gradient_descent(npdata)
# reg.w_approximation()
reg.w_approximation_regularizationl1()
# reg.w_approximation_regularizationl2()
RMSE = reg.rmse()
plt.plot(error_list) 
plt.xlabel("Number of iterations") 
plt.ylabel("Error") 
plt.show()


# plt.xlabel("Number of iterations")
# plt.ylabel("Error")


##w_app
# enter w0:1
# enter w1:1
# enter w2:1
# enter learning_rate:.000000001
# enter stoping constant:1
#-0.10180387587736867 0.3385060930153827 0.3843130680744672 12260.565137626187




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
# enter learning_rate:.000000001
# enter stoping constant:1
# lamda = 100
# -0.10180387566069264 0.33850609271629895 0.38431306769813645 12260.565151113604
# reg2.png
