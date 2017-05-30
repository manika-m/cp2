
# -*- coding: utf-8 -*-
'''Manika

#Tut 3 Problem 3
Write linear least-squares code to fit
sine and cosines to evenly sampled data.
'''           
import numpy as np
import matplotlib.pyplot as plt

n=100

x=np.linspace(0,  2*np.pi, n)
y=np.sin(x)
z=np.cos(x)

data1=y+np.random.randn(x.size)
data2 = z +np.random.randn(x.size)
order=8

A = np.zeros([x.size,order])
A[:,0]=1.0
for i in range(1,order):
    A[:,i]=A[:,i-1]*x
A = np.matrix(A)


d1 = np.matrix(data1).transpose()
lhs = A.transpose()*A
rhs = A.transpose()*d1
fitp = np.linalg.inv(lhs)*rhs #fit parameters
pred = A*fitp #predicted values 

plt.plot( fitp)
plt.title('fit parameters')
plt.show()

plt.plot(np.fft.fft(data1))
plt.title('FFT of data ')
plt.show()

plt.plot(x,y)
plt.plot(x,z)
plt.plot(x,data,'x')
plt.plot(x,pred,'r')
plt.title('fit')
plt.show()
