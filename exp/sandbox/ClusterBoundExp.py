
#Test some work on the cluster bound 
import sys 
import numpy 
import logging 
import sklearn.cluster
import matplotlib.pyplot as plt 

def computeBound(U, delta): 
    """
    Find the worse lower bound for a matrix U given a purtubation delta for a 
    two cluster problem. 
    """
    
    X, a, Y = numpy.linalg.svd(U)
    a = numpy.flipud(numpy.sort(a))
    epsilon = delta - numpy.trace(U.T.dot(U))
    logging.debug("a=" + str(a))
    logging.debug("epsilon=" + str(epsilon)) 
    
    s = 0    
    bestSigma = numpy.zeros(a.shape[0])
    bestSigma[-1] = 1
    
    while numpy.argmax(bestSigma) != 0 and s!=a.shape[0]-1: 
        q = (a[0:s+1]).sum()**2
        r = ((a[s+1:])**2).sum()
        logging.debug("q=" + str(q))
        logging.debug("r=" + str(r))
        logging.debug("s=" + str(s))
        
        b = numpy.zeros(5)
        b[0] = -(epsilon+r)*(s+1)**2 - q*s - q
        b[1] = 2*(epsilon+r)*(s+1)*(2*s+1) + 4*q*s + 2*q
        b[2] = -6*epsilon*s**2 -6*epsilon*s - epsilon - 5*q*s - q - 5*r*s**2 - 4*r*s
        b[3] = 4*epsilon*s**2 + 2*epsilon*s + 2*q*s + 2*r*s**2 
        b[4] = -epsilon*s**2
        
        logging.debug("b=" + str(b))
        rhos = numpy.roots(b)
        rhos = rhos[numpy.isreal(rhos)]
        rhos = rhos[rhos>0]
        logging.debug("rhos=" + str(rhos))
        
        #This is the objective when sigma[i] = a[i]
        bestObj = (a[1:]**2).sum() 

        for i in range(rhos.shape[0]): 
            sigma = numpy.zeros(a.shape[0])
            rho = rhos[i] 
            
            sigma[0:s+1] = (rho/(rho*s - s + rho))*(a[0:s+1]).sum()
            sigma[s+1:] = a[s+1:]*rho/(rho-1)
            obj = (sigma[1:]**2).sum()
            logging.debug("rho=" + str(rho))
            logging.debug("sigma=" + str(sigma))
            logging.debug("obj=" + str(obj))
            
            #Check if expansion of polynomial is correct 
            t1 = (rho/(rho*s - s + rho))
            t2 = rho/(rho-1)
            val1 = (sigma*(sigma-2*a))
            
            poly = q*((t1**2)*(s+1) - t1*2) + r*(t2**2-2*t2)
            
            assert abs((q*((t1**2)*(s+1) - t1*2))- val1[0:s+1].sum()) < 10**-3, "%s, %s" % (str((q*((t1**2)*(s+1) - t1*2))), str(val1[0:s+1].sum()))
            assert abs((r*(t2**2-2*t2)) - val1[s+1:].sum()) <= 10**-3, "%s" % str(abs((r*(t2**2-2*t2)) - val1[s+1:].sum()))
            logging.debug("poly=" + str(poly))
            
            #Check bound holds 
            val = (sigma*(sigma-2*a)).sum() 
            logging.debug("bound=" + str(val))
            
            if bestObj < obj: 
                bestObj = obj
                bestSigma = sigma.copy() 
        s += 1 
        
    return bestObj

def computeBound2(U, delta): 
    """
    Find the worse lower bound for a matrix U given a purtubation delta for a 
    two cluster problem. We use a simpler formulation and solve for y = rho/(rho-1). 
    """
    X, a, Y = numpy.linalg.svd(U)
    a = numpy.flipud(numpy.sort(a))
    epsilon = delta - numpy.trace(U.T.dot(U))
    logging.debug("a=" + str(a))
    logging.debug("epsilon=" + str(epsilon)) 
    tol = 10**-3
    
    s = 0    
    bestSigma = numpy.zeros(a.shape[0])
    bestSigma[-1] = 1
    
    while numpy.argmax(bestSigma) != 0 and s!=a.shape[0]-1: 
        q = (a[0:s+1]).sum()
        r = ((a[s+1:])**2).sum()
        logging.debug("q=" + str(q))
        logging.debug("r=" + str(r))
        logging.debug("s=" + str(s))
        
        b = numpy.zeros(5)
        b[0] = r
        b[1] = 2*r*s - 2*r
        b[2] = -epsilon + q**2*s - q**2 + r*s**2 - 4*r*s
        b[3] = -2*epsilon*s - 2*q**2*s - 2*r*s**2
        b[4] = -epsilon*s**2

        logging.debug("b=" + str(b))
        
        ys = numpy.roots(b)
        logging.debug("ys=" + str(ys))
        ys = ys[numpy.isreal(ys)]
        ys = ys[ys>0]
        logging.debug("ys=" + str(ys))
        
        #Compute epsilon from roots 
        for y in ys: 
            zero = (q**2*y**2*(s+1)) - (2*y*q**2*(y+s)) + (r*y**2*(y+s)**2) - (2*r*y*(y+s)**2) - (epsilon*(y+s)**2)
            assert abs(zero) <= tol, "%f" % (zero)
        
        #This is the objective when sigma[i] = a[i]
        bestObj = (a[1:]**2).sum() 

        for i in range(ys.shape[0]): 
            sigma = numpy.zeros(a.shape[0])
            y = ys[i] 
            
            sigma[0:s+1] = y*q/(y+s)
            sigma[s+1:] = a[s+1:]*y
    
            obj = s*y**2*q**2/(y+2)**2 + y**2 * r
            obj2 = (sigma[1:]**2).sum()
            assert abs(obj - obj2) <= tol, "%f %f" % (obj, obj2)
            
            logging.debug("sigma=" + str(sigma))
            logging.debug("obj=" + str(obj))
            
            #Check if expansion of polynomial is correct 
            #t1 = (rho/(rho*s - s + rho))
            #t2 = rho/(rho-1)
            #val1 = (sigma*(sigma-2*a))
            
            #poly = q*((t1**2)*(s+1) - t1*2) + r*(t2**2-2*t2)
            
            #assert abs((q*((t1**2)*(s+1) - t1*2))- val1[0:s+1].sum()) < 10**-3, "%s, %s" % (str((q*((t1**2)*(s+1) - t1*2))), str(val1[0:s+1].sum()))
            #assert abs((r*(t2**2-2*t2)) - val1[s+1:].sum()) <= 10**-3, "%s" % str(abs((r*(t2**2-2*t2)) - val1[s+1:].sum()))
            #logging.debug("poly=" + str(poly))
            
            #Check bound holds 
            val = (sigma*(sigma-2*a)).sum() 
            assert - val +  epsilon < 0.1, "%f" % (abs(val - epsilon))
            logging.debug("bound=" + str(val))
            
            if bestObj < obj: 
                bestObj = obj
                bestSigma = sigma.copy() 
        s += 1 
        
    return bestObj

def computeKClusterBound(U, delta): 
    """
    Find the worse lower bound for a matrix U given a purtubation delta for a 
    k-cluster problem. 
    """
    X, a, Y = numpy.linalg.svd(U)
    a = numpy.flipud(numpy.sort(a))
    epsilon = delta - numpy.trace(U.T.dot(U))
    logging.debug("a=" + str(a))
    logging.debug("epsilon=" + str(epsilon)) 
    
    s = 0    
    t = 0 
    bestSigma = numpy.zeros(a.shape[0])
    bestSigma[-1] = 1

    b = (a[0:k-1-s]**2).sum())
    c = (a[k+t+1:]**2).sum()
    q = a[k-s:k+t].sum()
    
    b = numpy.zeros(5)
    b[0] = c*s**2
    b[1] = -2*c*s**2 + 2*c*s*t + 2*c*s
    b[2] = -b*s**2 - 4*c*s*t - 4*c*s + c*t**2 + 2*c*t + c - epsilon*s**2 - q**2*s + q**2*t + q**2
    b[3] = -2*b*s*t - 2*b*s - 2*c*t**2 - 4*c*t - 2*c - 2*epsilon*s*t - 2*epsilon*s - 2*q**2*t - 2*q**2
    b[4] = -b*t**2 - 2*b*t - b - epsilon*t**2 - 2*epsilon*t - epsilon

    logging.debug("b=" + str(b))
    ys = numpy.roots(b)
    ys = rhos[numpy.isreal(ys)]
    ys = rhos[ys>0]
    logging.debug("ys=" + str(ys))
    
    

numpy.random.seed(21)
numpy.set_printoptions(suppress=True, precision=3)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

numExamples = 100 
numFeatures = 2

V = numpy.random.rand(numExamples, numFeatures)

V[0:80, :] += 2
U = V - numpy.mean(V)

UU = U.dot(U.T)
s, X = numpy.linalg.eig(UU)


#Lower and upper bounds on the cluster error 
print(numpy.trace(UU) - numpy.max(s), numpy.trace(UU))
print(numpy.linalg.norm(U)**2)
 
#Now compute true cluster error 
kmeans = sklearn.cluster.KMeans(2)
kmeans.fit(U)
error = 0

for i in range(numExamples): 
    #print(U[i, :])
    #print(kmeans.cluster_centers_[kmeans.labels_[i], :])
    error += numpy.linalg.norm(U[i, :] - kmeans.cluster_centers_[kmeans.labels_[i], :])**2

print(error)


deltas = numpy.arange(0, 150, 0.1)
worstLowerBounds = numpy.zeros(deltas.shape[0])
lowerBounds = numpy.zeros(deltas.shape[0])
upperBounds = numpy.zeros(deltas.shape[0])
realError = numpy.zeros(deltas.shape[0])

for i in range(deltas.shape[0]): 
    worstLowerBounds[i] = computeBound2(U, deltas[i])
    
    #Now add random matrix to U 
    E = numpy.random.randn(numExamples, numFeatures)
    E = E*numpy.sqrt(deltas[i])/numpy.linalg.norm(E)
    U2 = U + E
    
    #print(numpy.linalg.norm(U2 -U)**2, deltas[i])
    
    UU2 = U2.dot(U2.T)
    s, X = numpy.linalg.eig(UU2)
    
    lowerBounds[i] = numpy.trace(UU2) - numpy.max(s)
    upperBounds[i] = numpy.trace(UU2)
    
    kmeans = sklearn.cluster.KMeans(2)
    kmeans.fit(U2)
  
    for j in range(numExamples): 
        realError[i] += numpy.linalg.norm(U2[j, :] - kmeans.cluster_centers_[kmeans.labels_[j], :])**2
        
    
plt.plot(deltas, worstLowerBounds, label="Worst Continuous") 
#plt.plot(deltas, upperBounds, label="Upper") 
plt.plot(deltas, lowerBounds, label="Continuous Solution") 
plt.plot(deltas, realError, label="k-means")
plt.xlabel("delta")
plt.ylabel("J_k")
plt.legend(loc="upper left") 
plt.show()

#print("objective = " + str(computeBound(U, delta)))
