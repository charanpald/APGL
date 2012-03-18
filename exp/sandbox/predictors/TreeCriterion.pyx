import numpy 
cimport numpy 
cimport cython
"""
Some Cython code to efficiently compute the best split. 
"""


@cython.boundscheck(False) # turn of bounds-checking for entire function
def findBestSplit(int minSplit, numpy.ndarray[numpy.float_t, ndim=2] X, numpy.ndarray[numpy.float_t, ndim=1] y): 
    """
    Given a set of examples and a particular feature, find the best split 
    of the data. 
    """
    if X.shape[0] == 0: 
        raise ValueError("Cannot split on 0 examples")
    
    #Loop variables
    cdef float error, var1, var2, val     
    cdef numpy.ndarray[numpy.float_t, ndim=1] x, tempX, tempY, cumY, cumY2, vals 
    cdef numpy.ndarray[numpy.int_t, ndim=1] insertInds, inds
    cdef unsigned int rightSize, insertInd, featureInd, i
    
    #Best values 
    cdef float bestError = float("inf")   
    cdef unsigned int bestFeatureInd = 0 
    cdef float bestThreshold = X[:, bestFeatureInd].min() 
    cdef numpy.ndarray[numpy.int_t, ndim=1] bestLeftInds = numpy.array([], numpy.int), bestRightInds  = numpy.array([], numpy.int)
    
    for featureInd in range(X.shape[1]): 
        x = X[:, featureInd] 
        vals = numpy.unique(x)
        vals = (vals[1:]+vals[0:<unsigned int> vals.shape[0]-1])/2.0
        
        inds = numpy.argsort(x)
        tempX = x[inds]
        tempY = y[inds]
        cumY = numpy.cumsum(tempY)
        cumY2 = numpy.cumsum(tempY**2)
        
        insertInds = numpy.searchsorted(tempX, vals)
        
        for i in range(vals.shape[0]): 
            val = vals[i]
            #Find index where val will be inserted before to preserve order 
            insertInd = insertInds[i]
            
            rightSize = (tempX.shape[0] - insertInd)
            if insertInd < minSplit or rightSize < minSplit: 
                continue 

            if insertInd!=1 and insertInd!=x.shape[0]: 
                cumYVal = cumY[insertInd-1]
                cumY2Val = cumY2[insertInd-1]
                var1 = cumY2Val - (cumYVal**2)/float(insertInd)
                var2 = (cumY2[cumY2.shape[0]-1]-cumY2Val) - (cumY[cumY.shape[0]-1]-cumYVal)**2/float(tempX.shape[0] - insertInd)
   
                #tol = 0.01
                #assert abs(var1 - y[x<val].var()*y[x<val].shape[0]) < tol  
                #assert abs(var2 - y[x>=val].var()*y[x>=val].shape[0]) < tol 
                
                error = var1 + var2 
                
                if error <= bestError: 
                    bestError = error 
                    bestFeatureInd = featureInd
                    bestThreshold = val 
                    bestLeftInds = inds[0:insertInd]
                    bestRightInds = inds[insertInd:]
                    
    bestLeftInds = numpy.sort(bestLeftInds) 
    bestRightInds = numpy.sort(bestRightInds)
    
    return bestError, bestFeatureInd, bestThreshold, bestLeftInds, bestRightInds
    
def findBestSplit2(minSplit, X, y): 
    """
    Give a set of examples and a particular feature, find the best split 
    of the data. 
    """
    if X.shape[0] == 0: 
        raise ValueError("Cannot split on 0 examples")
    
    bestError = float("inf")   
    bestFeatureInd = 0 
    bestThreshold = X[:, bestFeatureInd].min() 
    bestLeftInds = numpy.array([], numpy.int)
    bestRightInds = numpy.array([], numpy.int)
    
    for featureInd in range(X.shape[1]): 
        x = X[:, featureInd] 
        vals = numpy.unique(x)
        vals = (vals[1:]+vals[0:-1])/2.0
        
        inds = numpy.argsort(x)
        tempX = x[inds]
        tempY = y[inds]
        cumY = numpy.cumsum(tempY)
        cumY2 = numpy.cumsum(tempY**2)
        
        insertInds = numpy.searchsorted(tempX, vals)
        
        
        for i in range(vals.shape[0]): 
            val = vals[i]
            #Find index where val will be inserted before to preserve order 
            insertInd = insertInds[i]
            
            rightSize = (tempX.shape[0] - insertInd)
            if insertInd < minSplit or rightSize < minSplit: 
                continue 

            if insertInd!=1 and insertInd!=x.shape[0]: 
                cumYVal = cumY[insertInd-1]
                cumY2Val = cumY2[insertInd-1]
                var1 = cumY2Val - (cumYVal**2)/float(insertInd)
                var2 = (cumY2[-1]-cumY2Val) - (cumY[-1]-cumYVal)**2/float(tempX.shape[0] - insertInd)
   
                tol = 0.01
                assert abs(var1 - y[x<val].var()*y[x<val].shape[0]) < tol  
                assert abs(var2 - y[x>=val].var()*y[x>=val].shape[0]) < tol 
                
                error = var1 + var2 
                
                if error <= bestError: 
                    bestError = error 
                    bestFeatureInd = featureInd
                    bestThreshold = val 
                    bestLeftInds = inds[0:insertInd]
                    bestRightInds = inds[insertInd:]
                    
    bestLeftInds = numpy.sort(bestLeftInds) 
    bestRightInds = numpy.sort(bestRightInds)
    
    return bestError, bestFeatureInd, bestThreshold, bestLeftInds, bestRightInds