import numpy as np
import pandas as pd
import poly_inversepoly as p_ip    #polynomial function, inverse polynomial function and its derivatives
import transdensity as td  #transition density function
import fourier_function as ff  #Fourier function for the periodicity

#read the daily price data
data=pd.read_csv('Data/HenryHub_natureGas.csv',header=0)
dailyV=data.Value[0:10]
N=len(dailyV)

#get the time series for factor x by the inverse polynomial functions on spot price series
def xfactor_sequence(alpha,theta,c,d,A=1,t=np.arange(0,N/250,1/250)):
#    n=len(alpha)+len(theta)+1 #degree of the polynomial
#    m=len(c)+len(d)  #number of terms for Fourier function
    dpv=dailyV/ff.fourier(c,d,t) #periodicity adjusted price process
    a=max(dpv)
    #print(dpv/a)
    x=p_ip.inverse_poly(alpha,theta,dpv/a,A)  #S=fourier(t)*Phi(x), then x=inversePhi(S/fourier(t))
    return x

'''if __name__=='__main__':
    print(xfactor_sequence(alpha=[0.2],theta=[0.6],c=[2,1],d=[2,1],A=1,t=np.arange(0,N/250,1/250)))'''


#establish the log_likelihood function
def log_likelihood(a,b,sigma,alpha,theta,c,d,A=1,t=np.arange(0,N/250,1/250)):
    LL=0
    xseq=xfactor_sequence(alpha,theta,c,d,A,t)
    q = np.multiply(p_ip.poly_dpoly(alpha, theta, xseq, A)[1], ff.fourier(c,d,t)) #derivative of the polynomial function
    for i in range(N-1):
        p=td.trans_jacobi(a,b,sigma,xseq[i+1],xseq[i])
        LL+=np.log(abs(p))-np.log(q[i+1])
    return LL


if __name__=='__main__':
    print(log_likelihood(a=8,b=9,sigma=1,alpha=[0.2],theta=[0.6],c=[1,2],d=[2,1]))



