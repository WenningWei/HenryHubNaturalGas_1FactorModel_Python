#This is a function of simulation of Jacobi process
import numpy as np
import random
import matplotlib.pyplot as plt

#backward Euler scheme to get the path of Jacobi process
def jacobi_simulation(a,b,sigma,t):
    TOL=1e-10      #tolerance for the Newton iteation
    Dt=np.diff(t) #Dt is the vector with dt=t_{i+1}-t_{i}, len(Dt)=len(t)-1
    dt=Dt[0]  #步长，也可用矩阵表示即 [t[i]-t[i-1]] i=1,...,T
    n=len(t)
    A=sigma**2*(2*a-1)*dt/8
    B=sigma**2*(1-2*b)*dt/8
    x0=random.betavariate(a,b)
    y=np.ones(n)*np.arcsin(np.sqrt(x0)) #use the substitution x=sin^2(y), then we get that dy=sigma/2*dw-t+[A*cot(y)+B*tan(y)]dt
    y_new=y[0]
    for i in range(n-1):
        y_old=y_new
        yN=y_new+1  #yN is the initial point for Newton iteration
        count=0
        C=sigma*np.random.randn()*np.sqrt(dt)/2
        while (abs(yN-y_new)>=TOL and count<=20):
            count+=1
            yN=y_new
            y_new=yN-(yN-A*1/np.tan(yN)-B*np.tan(yN)-y_old-C)/[1+A*1/np.sin(yN)**2-B*1/np.cos(yN)**2]
        y[i+1]=y_new
    x=np.sin(y)**2
    return x
plt.plot(np.arange(0,1,0.001),jacobi_simulation(3,6.8,6,np.arange(0,1,0.001)))
plt.show()