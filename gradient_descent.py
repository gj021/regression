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
        return 2.34
    def sumw1(self):
        return 3.32
    def sumw2(self):
        return 3.53

    #matrix operation and updating w
    def w_approximation(self):
        g0 = self.sumw0()
        g1 = self.sumw1()
        g2 = self.sumw2()
        if(abs(g0*self.learning_rate) < self.stoping_constant):# & g1*self.learning_rate < self.stoping_constant & g2*self.learning_rate < self.stoping_constant):
            return
        self.w0 = self.w0 - g0*self.learning_rate
        self.w1 = self.w1 - g1*self.learning_rate
        self.w2 = self.w2 - g2*self.learning_rate
        print(self.w0)
        self.w_approximation()

reg = gradient_descent(data)
reg.w_approximation()
#    self.data,self.learning_rate, self.stoping_constant,self.w0,self.w1,self.w1,self.w2 
