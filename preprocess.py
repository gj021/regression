import numpy as np
import pickle
from sklearn import preprocessing

data_set=open("data.txt",'r')

data=[]

for line in data_set:
    line=line.strip('\n')
    data_line=line.split(',')
    data_line=[float(i) for i in data_line]
    data.append(data_line[1:4])

data = np.array(data)

data = (data - np.min(data,axis = 0))/np.ptp(data,axis=0)

print(data)

f=open("data.pkl",'wb')
pickle.dump(data,f)
f.close()