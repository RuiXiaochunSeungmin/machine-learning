import numpy as np
def find_minimum(f,f_prime,f_doubleprime,x_0):
    x = x_0
    while (f_prime(x)!=0):
        L = abs(f_doubleprime(x))
        n = 1.0/L
        x = x - n*f_prime(x)
    print('Minimum of f appears at x='+str(x)+', the minimum values is f(x)='+str(f(x)))
    print('First derivative at minimum point: '+str(f_prime(x)))
    return f(x),x
    
    
def main():
    f = lambda x: (x-3)**2 + np.exp(x)
    f_prime = lambda x: 2*x - 6 + np.exp(x)
    f_doubleprime = lambda x: 2 + np.exp(x)
    x_0 = 0
    min_value, argmin = find_minimum(f,f_prime,f_doubleprime,x_0)
    
    
main()