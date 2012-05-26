
import numpy 
import logging 

"""
Some bounds for the continuous k-cluster problem. 
"""

class ClusterBound(object): 
    
    @staticmethod 
    def compute2ClusterBound(U, delta): 
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
        
        bestSigma = numpy.zeros(a.shape[0])
        bestObj = -float("inf")
        
        for s in range(a.shape[0]): 
            logging.debug("s="+str(s))             
            
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
            
            for i in range(ys.shape[0]): 
                sigma = numpy.zeros(a.shape[0])
                y = ys[i] 
                
                sigma[0:s+1] = y*q/(y+s)
                sigma[s+1:] = a[s+1:]*y
        
                obj = s*y**2*q**2/(y+s)**2 + y**2 * r
                obj2 = (sigma[1:]**2).sum()
                assert abs(obj - obj2) <= tol, "%f %f" % (obj, obj2)
                
                logging.debug("sigma=" + str(sigma))
                logging.debug("obj=" + str(obj))
                            
                #Check bound holds 
                val = (sigma*(sigma-2*a)).sum() 
                assert - val +  epsilon < 0.1, "%f" % (abs(val - epsilon))
                logging.debug("bound=" + str(val))
                
                if obj > bestObj and (sigma[0] >= sigma[1:]).all(): 
                    bestObj = obj
                    bestSigma = sigma.copy() 
                    logging.debug("bestSigma=" + str(bestSigma))
            
        return bestObj, bestSigma 
    
    @staticmethod 
    def computeKClusterBound(U, delta, k, a=None): 
        """
        Find the worse lower bound for a matrix U given a purtubation delta for a 
        k-cluster problem. U is a centered matrix of examples. If a!=None then 
        this vector is used as the set of singular values of U. 
        """
        tol = 10**-6
        logging.debug("Computing cluster bound")
        if a==None: 
            X, a, Y = numpy.linalg.svd(U)
        a = numpy.flipud(numpy.sort(a))
        epsilon = delta - numpy.trace(U.T.dot(U))
        logging.debug("a=" + str(a))
        logging.debug("epsilon=" + str(epsilon)) 
        
        bestSigma = numpy.zeros(a.shape[0])
        bestSigma[-1] = 1
        bestObj = -float("inf") 
        
        for s in range(k): 
            for t in range(a.shape[0]-k+1):
                logging.debug("s="+str(s) + ", t="+str(t))
                r = (a[0:k-1-s]**2).sum()
                c = (a[k+t:]**2).sum()
                q = a[k-s-1:k+t].sum()
                        
                b = numpy.zeros(5)
                b[0] = c*s**2
                b[1] = -2*c*s**2 + 2*c*s*t + 2*c*s
                b[2] = -4*c*s*t - 4*c*s + c*t**2 + 2*c*t + c - epsilon*s**2 - q**2*s + q**2*t + q**2 - r*s**2
                b[3] = -2*c*t**2 - 4*c*t - 2*c - 2*epsilon*s*t - 2*epsilon*s - 2*q**2*t - 2*q**2 - 2*r*s*t - 2*r*s
                b[4] = - epsilon*t**2 - 2*epsilon*t - epsilon - r*t**2 - 2*r*t - r 
            
                logging.debug("b=" + str(b))
                ys = numpy.roots(b)
                ys = numpy.real_if_close(ys, 10**10)
                ys = ys[numpy.isreal(ys)].real
                #Note that y does not have to be positive - pick the largest abs(y)
                logging.debug("ys=" + str(ys))
                
                #Check solutions are correct 
                for y in ys: 
                    zero = -epsilon - r + ((1+s+t)*y**2*q**2)/(1+t+s*y)**2
                    zero += -2*y*q**2/(1+t+s*y) + c*(y**2-2*y)
                    assert abs(zero) < 10**-6 ,"%f" % (zero)
                            
                for i in range(ys.shape[0]): 
                    y = ys[i] 
                    obj = ((t+1)*y**2*q**2)/ (1+t+s*y)**2 + c * y**2
                    logging.debug("obj=" + str(obj))
                                
                    sigma = numpy.zeros(a.shape[0])    
                    sigma[0:k-s-1] = a[0:k-s-1]    
                    sigma[k-s-1:k+t] = y*q/(1+t+s*y)
                    sigma[k+t:] = y*a[k+t:]
                    logging.debug("sigma=" + str(sigma))
                    
                    #Check objective is correct 
                    obj2 = (sigma[k-1:]**2).sum()
                    assert abs(obj - obj2) <= tol, "%f %f" % (obj, obj2)
    
                    #Check bound holds 
                    val = (sigma*(sigma-2*a)).sum() 
                    assert - val +  epsilon < 0.1, "%f" % (abs(val - epsilon))
                    
                    #Only accept if constraints are satisfied 
                    if obj > bestObj and (sigma[0:k-1] >= sigma[k-1]).all() and (sigma[k:] <= sigma[k-1]).all(): 
                        bestObj = obj
                        bestSigma = sigma.copy()
                        logging.debug("bestSigma=" + str(bestSigma))
                
        return bestObj, bestSigma  
