#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manika 214582074
"""
#Question 1
import numpy as np
from numpy.fft import fft
from numpy.fft import ifft
from matplotlib import pyplot as plt


def Q1_shift(x,s): #takes in an array x, and amount to shift s
     s = [s] #array
     f=numpy.fft.fft(x)
     g=numpy.fft.fft(s)
     return np.real(ifft(f*g)) #convolution
if __name__=='__main__':
    x= np.arange(-20,20,0.1)
    sigma=1
    y= np.exp(-0.5*x**2/(sigma**2))
    y_shift= Q1_shift(y,y.size/2) 
    print 'Question 1'
    print 'Plot a gaussian that started in the centre of the array shifted by half the array length.'
    plt.plot(x,y)
    plt.plot(x,y_shift)
    plt.show()       
    
#Question 2


''' #Write a routine to take the correlation function of two
arrays. Plot the correlation function of a Gaussian with itself.'''
def Q2_corr(a,b):
    assert(a.size==b.size)
    f=fft(a)
    g=fft(b)
    g_conj=np.conj(g)
    return np.real(ifft(f*g_conj))
    
if __name__=='__main__':
   
    print 'Question2. Plot the correlation function of a Gaussian with itself:'
    a=np.arange(-20,20,0.1)
    sigma=2
    b=np.exp(-0.5*a**2/sigma**2)
    b_c = Q2_corr(b,b)
    plt.plot(a,b_c)
    plt.show()
    
#Question 3
#using methods from Q2 and Q1
a=np.arange(-20,20,0.1)
sigma=1
b=np.exp(-0.5*a**2/sigma**2)
bcorr=Q2_corr(b,b) #take correlation of gaussian with itself b*b
bshift=Q1_shift(b,4) #guaussian with arbitrary amount of shift =4
bshiftcorr = Q2_corr(bshift,bshift) #correlation of shifted gaussian fn
mean_error = np.mean(np.abs(bcorr-bshiftcorr))
print 'Question 3'
print 'The mean difference between the two correlation functions is ', mean_error


#Question 4 conv of 2 arrrays
def Q4_conv(a,b):#takes in 2 arrays

 f=fft(a)
 g=fft(b)
 conv=np.real(ifft(f*g))
 return conv

a = [0,1,1,1,1,0] #added zeros on end of array prevetn wrap around
b = a


print 'Question 4'
print 'array 1: size 6 ',a
print 'array 2: ',b

print 'convolution of arrays: \n', Q4_conv(a,b) 


#Question 5

#def