import pickle
import numpy

f=open("data.pkl",'rb')
data=pickle.load(f)
# prfloat(data)

class gradient_descent:
    def __init__(self,data):
        self.data = data
        self.w0 = float(input("enter w0:"))
        self.w1 = float(input("enter w1:"))
        self.w2 = float(input("enter w2:"))
        
        self.learning_rate = float(input("enter learning_rate:"))
        

        self.stoping_constant = float(input("enter stoping constant:"))

    def sumw0(self):
        error_g = self.w0*self.data.shape[0]
        # print(self.data.shape[0])
        for x in range(0,self.data.shape[0]):
            error_g += self.w1*self.data[x,0]
            error_g += self.w2*self.data[x,1]
            error_g -= self.data[x,2]
        return error_g
    def sumw1(self):
        error_g = self.w0*self.data.shape[0]
        for x in range(0,self.data.shape[0]):
            error_g += self.w1*self.data[x,0]
            error_g += self.w2*self.data[x,1]
            error_g -= self.data[x,2]
            error_g = error_g*self.data[x,0]
            return error_g
    def sumw2(self):
        error_g = self.w0*self.data.shape[0]
        for x in range(0,self.data.shape[0]):
            error_g += self.w1*self.data[x,0]
            error_g += self.w2*self.data[x,1]
            error_g -= self.data[x,2]
            error_g = error_g*self.data[x,1]
            return error_g

    def error_fn(self):
        error = 0
        for x in range(0,self.data.shape[0]):
            dev = self.w0
            dev += self.w1*self.data[x,0]
            dev += self.w2*self.data[x,1]
            dev -= self.data[x,2]
            dev = dev*dev
            dev = dev*0.5
        error += dev
        return error

    #matrix operation and updating w
    def w_approximation(self,error):
        g0 = self.sumw0()
        g1 = self.sumw1()
        g2 = self.sumw2()
        new_error = self.error_fn()
        if(abs(error-new_error) < self.stoping_constant):
            return
        else:
            error = new_error
        self.w0 = self.w0 - g0*self.learning_rate
        self.w1 = self.w1 - g1*self.learning_rate
        self.w2 = self.w2 - g2*self.learning_rate
        print(self.w0, self.w1, self.w2, error)
        self.w_approximation(error)

reg = gradient_descent(data)
reg.w_approximation(0)
#    self.data,self.learning_rate, self.stoping_constant,self.w0,self.w1,self.w1,self.w2 
