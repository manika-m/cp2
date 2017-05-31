# -*- coding: utf-8 -*-
"""


@author: Manika
"""
''' Manika Moodliar 214582074
email manika.moodliar@gmail.com'''
import numpy as np
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

class nbody:
    def __init__(self, n, G, soft, mass, dt ):
        self.m = mass
        self.n = n #number of particles
        self.G = G
        self.soft = soft #softening parameter
        self.dt = dt #time step
        self.x = np.random.randn(n)  #position in x space dimensions, randomly assigned float from gaussian dist
        self.y = np.random.randn(n)
        self.vx= np.zeros(n)  #velocity in x
        self.vy = np.zeros(n)
        # to get a 2D grid (grid dimension usually smaller than num particles)
        self.density = np.zeros([abs(np.round(np.max(self.x)))+1, abs(np.round(np.max(self.y)))+1 ])
    
    def particlePositions(self):
        self.x = np.random.randn(n)  #position in x space dimensions, randomly assigned float from gaussian dist
        self.y = np.random.randn(n)

        # to get a 2D grid (grid dimension usually smaller than num particles)

        
    def getDensity(self):
        for i in range(self.n):
            xi = self.x[i]
            yi = self.y[i]
            px = np.round(xi) #int value of x position is nearest grid point in x
            py = np.round(yi)
            self.density[px,py]+=1 #adds 1 to mass density at grid point nearest to particle position
        return self.density*self.m #returns a density grid 2d array
        
    def distance(self):
        '''scipy.spatial.distance.pdist gives 
        pairwise distances between observations in n-dimensional space.
        output: Returns a condensed distance matrix Y. 
        For each ii and jj (where i<j<m),where m is the number of original observations. 
        The metric dist(u=X[i], v=X[j]) is computed and stored in entry ij.'''
        
        r = pdist(self.getDensity())
        #print r
        return r
     
    def getPotential(self):    
        r=self.distance()
        potential1 = -1*self.G*self.m/np.sqrt(r**2 + (self.soft)**2)
        #getDensity is a 2D array  so the above code doesn't actually work /make sense
        #potFT=np.fft.fft(potential1)
        #densFT = np.fft.fft(self.getDensity())
        #conv =np.fft.ifft(densFT*potFT).real #this is the total potential: 
                                    #convolution of density & softened potential of a particle
        return potential1
        
    def getForce(self): #first got the softtened potential in grid and now calculate force
        #to get force (-grad of the potential) :   
        conv = self.getPotential()
        fx = -1*np.gradient(conv,axis=0) #axis 0 for 1st dimension of the potential array(x)
        fy = -1*np.gradient(conv,axis=1)
        '''numpy.gradient() Return the gradient of an N-dimensional array.
        The gradient is computed using second order accurate central differences 
        in the interior and either first differences or second order accurate one-sides 
        (forward or backwards) differences at the boundaries. The returned gradient 
        hence has the same shape as the input array.'''
        return fx,fy
        
    def partForce(self): #particle forces, from getForce forces in grid 
        fx, fy = self.getForce()
        fx1 = np.zeros(self.n)
        fy1 = np.zeros(self.n)
        for i in range(self.n):
            xi = np.round(self.x[i])
            yi = np.round(self.y[i]) #int values to get closest cell in index grid
            fx1[i] = fx[xi]
            fy1[i] = fy[yi]
        return fx1,fy1
        
    #to update the velocity and in the time evolved system and check the energy
        
    def update(self):
        self.fx,self.fy = self.getForce()
        self.x += self.vx*self.dt
        self.y +=  self.vy*self.dt
        self.vx += self.fx*self.dt #v ~ a*t
        self.vy += self.fy*self.dt
        KE = 0.5*self.m*np.sum((self.vx**2)+(self.vy**2)) #kinetic energy
        PE = self.getPotential() #potential energy
        return (KE+PE)
        
if __name__=='__main__':
    part = nbody(100000,1, 0.1, 0.1, 0.1) #parameters (num, G, soft, mass, dt ) 
    '''for i in range(10):
       print 'the energy of the system is', part.update()'''
    
    
    print 'this is just a plot of scattered particles'
    a =50
    for i in range(a):
            plt.clf()
            plt.plot(part.x,part.y,'+')
            plt.draw()
    plt.show()
    
    
        