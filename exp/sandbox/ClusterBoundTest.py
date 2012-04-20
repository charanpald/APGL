
#Test some work on the cluster bound 
import numpy 

numpy.random.seed(21)
a = numpy.random.rand(5)
a = numpy.flipud(numpy.sort(a))
print(a)

epsilon = 0.3 

ar = (a[1:]**2).sum()
br = -2*(a[1:]**2).sum()
cr = -(a[0]**2+epsilon)

coeff = [ar, br, cr]
alphas = numpy.roots(coeff)
print(alphas)

rhos = alphas/(1-alphas)

for i in range(alphas.shape[0]): 
    sigma = numpy.zeros(5)
    sigma[0] = a[0]
    sigma[1:] = a[1:]*alphas[i]
    
    print(sigma)
    print((sigma[1:]**2).sum())

    #Check bound holds 
    val = (sigma**2).sum() - 2*(sigma*a).sum() 
    print(val)