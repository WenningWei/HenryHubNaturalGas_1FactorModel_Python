import math
import numpy as np

#define the Jacobi Polynomial, J(i+1)=a*J(i-1)+b*J(i) (It is better to use vector)
def jacobi_poly(x,u,v,N):
    J=np.ones(N+1)
    J[1]=1-(u+1)*x/v
    for n in range(1,N):  #from 1 to n-1
        a = n*(v-u-n) / (u + 2 * n - 1) * 1 / (u + 2 * n )
        b= -x + (2*n*(u+n)+v * (u - 1)) / (u + 2 * n+1) * 1 / (u + 2 * n - 1)  # 1/(a*b) 要写成 1/a*1/b?
        c=(u+2*n+1)*(u+2*n)/(u+n)*1/(v+n)
        J[n+1]=(b*J[n]+a*J[n-1])*c
    return J

#if __name__=='__main__':
 #   print(jacobi_poly(0.1,10.5,2.5,26))

def notation(c,k):  #function (c)_n=gamma(c+n)/gamma(c)
    value=math.gamma(c+k)/math.gamma(c)
    return value
if __name__=='__main__':
    print(notation(3,165))

#define the trasition density for Jacobi process from (y,t) to (x,t+dt)
'''def trans_jacobi(a,b,sigma,x,y,dt=1/365):
    if a<1 or b<1:
        print('\nThe coefficients a and b are not in the interval!')
    n=0
    TOL=1e-7
    while np.exp(-n*sigma**2*dt/2*(a+b-1+n))>TOL:
        n+=1
    #print(n)
    # first term of transition function w(x)
    w=math.gamma(a+b)/math.gamma(a)*1/math.gamma(b)*np.power(x,a-1)*np.power(1-x,b-1)
    p=w
    for i in range(1,n):
        mu=(a+b-1+i)*i*sigma**2/2
        A=notation(a,i)/notation(b,i)*notation(a+b,i)/np.math.factorial(i)*np.exp(-mu*dt)
       # print(A)
        k=(a+b-1+2*i)/(a+b-1+i)*A
        p=p+k*w*jacobi_poly(x,a+b-1,a,n)[i]*jacobi_poly(y,a+b-1,a,n)[i]
        #print(k,w,jacobi_poly(x,a+b-1,a,3)[i],jacobi_poly(y,a+b-1,a,3)[i],mu,p)
    return(p)



if __name__=='__main__':
    print(trans_jacobi(1.05,1.05,.5,0.1,0.6))'''