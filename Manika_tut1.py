#!/usr/bin/env python3
# -*- coding: utf-8 -*-
''' Manika Moodliar 214582074'''

import numpy as np
from scipy.integrate import simps
from matplotlib import pyplot as plt
#Problem 3,4,5,6 tut 1

pii = np.pi
#Question 3
def vector(n): 
    x=np.linspace(0,pii/2,n)
    return x

def simpleMethod(n):  #integrate cos 
    dx = (pii/2)/(n-1)
    y = np.cos(vector(n))
    summ =y.sum()*dx
    return summ

print('Question 3')
m=[10,30,100,300,1000,200000]
total_a = 0
for n in m:
    s =simpleMethod(n)
    err=abs(s-1) #abs value of integral-1
    a = np.log(err)/(np.log(n))
    print 'Simple integral with n=',n, ' points is ',s
    print 'Error for ',n,' points is ',err
    print 'a = ',a
    if n<5 :
        total_a+=a #only compute error for first 5 n's in the array not 200000
avg = total_a/5
print 'Average a is ',avg
print'if num points is n then error scales as ~ 1/n'


#Question 4:

print('\n Question 4')

x=np.arange(11)
print(x)
xodd=x[1::2]
print 'Only odd numbers from array: ',xodd
xeven=x[2:-2:2]
print 'Only even numbers from array skipping first and last elements: ',xeven
    
    
#Question 5

print '\n\nQuestion 5 /Simpson"s rule'

m=[11,31,101,301,1001]
total_a = 0
for n in m:
    x1=np.linspace(0,pii/2,n)
    y1 = np.cos(x1)
    simp = simps(y1,x1)
    err=abs(simp-1)
    alpha=np.log(err)/(np.log(n))
    print(alpha)
    print 'Simpsons rule integral with n=',n, ' points is ',simp
    print 'Error for ',n-1,' points is ',err
    print 'a = ',a
    total_a+=a
avg = total_a/5
print '\n Average a for simpson rule is ',avg
print 'needed ~200000 points to get the same error accuracy'


#Question 6:

print'\nQuestion 6\n'

m=[11,31,101,301,1001,3001,10001,30001,100001]
m=np.array(m)
simpson_err=np.zeros(m.size)
simple_err=np.zeros(m.size)
for ii in range(m.size):
    n=m[ii]
    x1=np.linspace(0,pii/2,n)
    y1 = np.cos(x1)
    simp = simps(y1,x1)
    err=abs(simp-1)
    simpson_err[ii]=err
    simple_err[ii]=np.abs(simpleMethod(n)-1)
plt.plot(m,simple_err)
plt.plot(m,simpson_err)
ax=plt.gca()
ax.set_yscale('log')
ax.set_xscale('log')
plt.xlabel('Number of points')
plt.ylabel('Error')
plt.title('Error against number of points for \n the simple method integration (blue) and for Simpsons rule')
plt.show()















