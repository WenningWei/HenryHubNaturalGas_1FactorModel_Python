#do the calibration by random walk optimization

from __future__ import print_function
import math
import random
import numpy as np
import loglikelihood as llh
import time
import cProfile

def randomWalk(Nvariable,lower,upper,):
    N = 2**Nvariable # number of iterations
    x = [lower[i]+upper[i]*1/2 for i in range(Nvariable)]# initial point
    step = [(upper[i]-lower[i])/2 for i in range(Nvariable)] #initial step, choose a big step
    l=1e-3
    TOL = [(upper[i]-lower[i])*l for i in range(Nvariable)] #lower bound for the steps
    walk_num = 1
    # function to minimize
    def function(x):
        ll = llh.log_likelihood(x[0],x[1],x[2],[x[3]],[x[4]],[x[5]],[x[6]])
        return ll


    def domain(x,a,b):
        for i in range(len(x)):
            if x[i]<a[i]:
                x[i]=a[i]
            elif x[i]>b[i]:
                x[i]=b[i]
        return x

    while(step > TOL):
        k = 1 # counting
        while(k < N):
            u = [random.uniform(-1,1) for i in range(Nvariable)] # random variable
            #u1 is the standardized random variable with norm 1
            u1 = [u[i]/math.sqrt(sum([u[i]**2 for i in range(Nvariable)])) for i in range(Nvariable)]
            x1 = [x[i] + step[i]*u1[i] for i in range(Nvariable)]
            x1=domain(x1,lower,upper)
            if(function(x1) > function(x)): #a better point found
                k = 1
                x = x1
            else:
                k += 1
        step = [step[i]/2 for i in range(len(step))]
        print("%d random walk finished" % walk_num)
        print("maximum point in this step:", x1 )
        print("maximum:",function(x1))
        walk_num += 1

    print("numbers:",walk_num-1)
    print("maximum point:",x)
    print("maximum:",function(x))



cProfile.run('randomWalk(Nvariable=7,lower=[1.01,1.01,0.1,-3,0,0.01,0],upper=[30,30,10,3/2,1,10,2*np.pi])')
