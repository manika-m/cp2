
# -*- coding: utf-8 -*-
"""
#Tut 3 Problem 3
Write linear least-squares code to 
sines and cosines to evenly sampled data.
"""           
import numpy as np
import matplotlib.pyplot as plt

n=100

x=np.linspace(0,  2*np.pi, n)
y=np.sin(x)
z=np.cos(x)

data=y+np.random.randn(x.size)
order=10

A = np.zeros([x.size,order])
A[:,0]=1.0
for i in range(1,order):
    A[:,i]=A[:,i-1]*x
A = np.matrix(A)


d = np.matrix(data).transpose()
lhs = A.transpose()*A
rhs = A.transpose()*d
fitp = np.linalg.inv(lhs)*rhs #fit parameters
pred = A*fitp #predicted values 

plt.plot( fitp)
plt.title('fit parameters')
plt.show()

plt.plot(np.fft.fft(data))
plt.title('FFT of data ')
plt.show()

plt.plot(x,y)
plt.plot(x,z)
plt.plot(x,data,'x')
plt.plot(x,pred,'r')
plt.title('fit')
plt.show()