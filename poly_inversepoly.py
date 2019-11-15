import numpy as np
import math
from scipy import integrate


#define the integrand of the polynomial
def f(t,alpha,theta): #elements of theta are in (0,1)
    m=len(alpha)+len(theta) #degree of the integrant
    n = math.floor(m / 2)  # 取整
    f=1
    if m==0:
        f=1
    elif m==1:
        beta= -1 / 2 + theta[0] * (1 / 2 - (-1 / 2))
        f=2*beta*(2*t-1)+1
    elif m%2==0:
        beta=np.ones(n)
        for i in range(n):
            if alpha[i]>=-3 and alpha[i]<=3/5:
                beta[i] = -(1 / 2 + alpha[i] / 6) + theta[i] * 2 * (1 / 2 + alpha[i] / 6)
            elif alpha[i]>3/5 and alpha[i]<=3/2:
                beta[i]=-np.sqrt(3/8-2/3*np.power(alpha[i]-3/4,2))+theta[i]*2*np.sqrt(3/8-2/3*np.power(alpha[i]-3/4,2))
            f=f*(alpha[i]*np.power((2*t-1),2)+2*beta[i]*(2*t-1)+1-2*alpha[i]/3)
    elif m%2==1:
        beta=np.ones(n)
        for i in range(n):
            if alpha[i]>=-3 and alpha[i]<=3/5:
                beta[i] = -(1 / 2 + alpha[i] / 6) + theta[i] * 2 * (1 / 2 + alpha[i] / 6)
            elif alpha[i]>3/5 and alpha[i]<=3/2:
                beta[i]=-np.sqrt(3/8-2/3*np.power(alpha[i]-3/4,2))+theta[i]*2*np.sqrt(3/8-2/3*np.power(alpha[i]-3/4,2))
            f=f*(alpha[i]*np.power((2*t-1),2)+2*beta[i]*(2*t-1)+1-2*alpha[i]/3)
        beta[n] = -1 / 2 + theta[n] * (1 / 2 - (-1 / 2))
        f=f*(2*beta[n]*(2*t-1)+1)
    return f

def poly_dpoly(alpha,theta,x,A=1): #alpha and theta are volume vector, N is the degree of the polynomial
    N=len(alpha)+len(theta)+1
    g=lambda t:f(t,alpha,theta)
    b=integrate.quad(g,0,1)[0]
    F=np.ones(len(x))
    dF=np.ones(len(x))
    for i in range(len(x)):
        F[i]=A*integrate.quad(g,0,x[i])[0]/b    #polynomial
        dF[i]=A*f(x[i],alpha,theta)/b              #derivative of polynomial

    return F,dF  #N is the degree of the polynomial

if __name__=='__main__':
    print(poly_dpoly([0.2],[.6],[0.05,0.06],1000))


# inverse polynomial via newton method. It does not work well. The reason could be that the derivative is close to 0
# at some point in [0,1]
'''def inverse_poly(alpha,theta,y,A=1):
    TOL = 1e-5
    xold = 1 / 2 * np.ones(len(y))
    xnew = 3 / 4 * np.ones(len(y))
    g = lambda t: f(t, alpha, theta)
    b = integrate.quad(g, 0, 1)[0]
    N=len(y)
    for i in range(N):
        count = 1
        while abs(xold[i] - xnew[i]) > TOL and count <= 20:
            count+=1
            xold[i] = xnew[i]
            xnew[i] = xold[i] - (A*integrate.quad(g, 0, xold[i])[0]-y[i]*b)/ f(xold[i], alpha, theta)*1/A
    return xnew'''


#inverse polynomial via Newton bisection method
def inverse_poly(alpha,theta,y,A=1):
    ToL=1e-4
    N=len(y)
    x=np.ones(N)
    for i in range(N):
        if y[i]<0 or y[i]>A:
            print('\n The seasonality adjust price is not in [0,1]')
        x_lower=0
        x_upper=1
        n=0
        while abs(x_lower-x_upper)>ToL:
            n+=1
            xnew=(x_lower+x_upper)/2
            val_xnew=poly_dpoly(alpha,theta,[xnew],A)[0]
            if val_xnew<y[i]:
                x_lower=xnew
            elif val_xnew>y[i]:
                x_upper=xnew
            elif val_xnew==y[i]:
                break
        x[i]=xnew
    return x


if __name__=='__main__':  #这个语句只有在这个文件被运行时，才运行。这个文件被调用时，在调用文件中不运行这个语句
    print(inverse_poly([0.2],[.6],[270,400],A=1000))