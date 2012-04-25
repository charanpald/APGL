
#Test some work on the cluster bound 
import numpy 

numpy.random.seed(21)
a = numpy.random.rand(5)
a = numpy.flipud(numpy.sort(a))
print("a=" + str(a))

epsilon = 0.3 
s = 0

q = (a[0:s+1]).sum()**2
r = ((a[s+1:])**2).sum()

print(q, a[0]**2)

b = numpy.zeros(5)
b[0] = -(epsilon+r)*(s+1)**2 - q*s - q
b[1] = 2*(epsilon+r)*(s+1)*(2*s+1) + 4*q*s + 2*q
b[2] = -6*epsilon*s**2 -6*epsilon*s - epsilon - 5*q*s - q - 5*r*s**2 - 4*r*s
b[3] = 4*epsilon*s**2 + 2*epsilon*s + 2*q*s + 2*r*s**2 
b[4] = -epsilon*s**2

print("b=" + str(b))
rhos = numpy.roots(b)
print(rhos)

for i in range(rhos.shape[0]): 
    sigma = numpy.zeros(5)
    rho = rhos[i] 
    
    if rho != 0: 
        sigma[0:s+1] = (rho/(rho*s - s + rho))*(a[0:s+1]).sum()
        sigma[s+1:] = a[s+1:]*rho/(rho-1)
        print("rho=" + str(rho))
        print("sigma=" + str(sigma))
        print("obj=" + str((sigma[1:]**2).sum()))
        
        #Check if expansion of polynomial is correct 
        t1 = (rho/(rho*s - s + rho))
        t2 = rho/(rho-1)
        val1 = (sigma*(sigma-2*a))
        
        poly = q*((t1**2)*(s+1) - t1*2) + r*(t2**2-2*t2)
        
        assert (q*((t1**2)*(s+1) - t1*2)) == val1[0:s+1].sum()
        assert abs((r*(t2**2-2*t2)) - val1[s+1:].sum()) <= 10**-3
        print("poly=" + str(poly))
        
        #Check bound holds 
        val = (sigma*(sigma-2*a)).sum() 
        print("bound=" + str(val))