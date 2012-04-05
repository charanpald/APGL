# cython: profile=False
# cython: boundscheck=False
# cython: wraparound=False
# filename: TreeCriterion.pyx

import numpy 
cimport numpy 
cimport cython
"""
Some Cython code to efficiently compute the best split. 
"""

@cython.boundscheck(False) # turn of bounds-checking for entire function
def findBestSplit(int minSplit, numpy.ndarray[numpy.float_t, ndim=2] X, numpy.ndarray[numpy.float_t, ndim=1] y,  numpy.ndarray[numpy.int_t, ndim=1] nodeInds, numpy.ndarray[numpy.int_t, ndim=2] argsortX): 
    """
    Given a set of examples and a particular feature, find the best split 
    of the data. 
    """
    if X.shape[0] == 0: 
        raise ValueError("Cannot split on 0 examples")
    
    #Loop variables
    cdef float error, var1, var2, val     
    cdef numpy.ndarray[numpy.float_t, ndim=1] x, tempX, tempY, cumY, cumY2, vals
    cdef numpy.ndarray[numpy.int8_t, ndim=1] boolInds
    cdef numpy.ndarray[numpy.int_t, ndim=1] insertInds
    cdef unsigned int rightSize, insertInd, featureInd, i, j, insertIndm1, finalInd
    
    #Best values 
    cdef float bestError = float("inf")   
    cdef unsigned int bestFeatureInd = 0 
    cdef float bestThreshold = X[:, bestFeatureInd].min() 
    cdef numpy.ndarray[numpy.int_t, ndim=1] bestLeftInds = numpy.array([0]), bestRightInds  = numpy.array([0])
    
    for featureInd in range(X.shape[1]): 
        x = X[:, featureInd] 

        tempX = numpy.zeros(X.shape[0])
        tempY = numpy.zeros(X.shape[0])
        
        tempX[argsortX[:, featureInd][nodeInds]] = x[nodeInds]
        tempY[argsortX[:, featureInd][nodeInds]] = y[nodeInds]
        
        tempX2 = numpy.zeros(nodeInds.shape[0])
        tempY2 = numpy.zeros(nodeInds.shape[0])
        
        boolInds = numpy.zeros(X.shape[0], numpy.int8)
        boolInds[argsortX[:, featureInd][nodeInds]] = 1

        j = 0 
        
        for i in range(tempX.shape[0]):
            if boolInds[i] == 1: 
                tempX2[j] = tempX[i]       
                tempY2[j] = tempY[i]
                j += 1 
                
        tempX = tempX2 
        tempY = tempY2

        tol = 0.001
        
        argsInds = numpy.argsort(X[nodeInds, featureInd])
        tempX2 = X[nodeInds, featureInd][argsInds]
        tempY2 = y[nodeInds][argsInds]
        
        #assert (numpy.linalg.norm(tempX - tempX2) < tol), "%s %s, %s" % (str(tempX), str(tempX2), str(boolInds)) 
        #assert (numpy.linalg.norm(tempY - tempY2) < tol)        
        
        vals = numpy.unique(tempX)
        vals = (vals[1:]+vals[0:<unsigned int> vals.shape[0]-1])/2.0

        cumY = numpy.cumsum(tempY)
        cumY2 = numpy.cumsum(tempY**2)
        
        insertInds = numpy.searchsorted(tempX, vals)
        
        for i in range(vals.shape[0]): 
            val = vals[i]
            #Find index where val will be inserted before to preserve order 
            insertInd = insertInds[i]
            insertIndm1 = insertInd-1
            finalInd = cumY2.shape[0]-1
            
            rightSize = (tempX.shape[0] - insertInd)
            if insertInd < minSplit or rightSize < minSplit: 
                continue 

            if insertInd!=1 and insertInd!=tempX.shape[0]: 
                cumYVal = cumY[insertIndm1]
                cumY2Val = cumY2[insertIndm1]
                var1 = cumY2Val - (cumYVal**2)/float(insertInd)
                var2 = (cumY2[finalInd]-cumY2Val) - (cumY[finalInd]-cumYVal)**2/float(tempX.shape[0] - insertInd)
   
                #tol = 0.01
                #assert abs(var1 - y[x<val].var()*y[x<val].shape[0]) < tol  
                #assert abs(var2 - y[x>=val].var()*y[x>=val].shape[0]) < tol 
                
                error = var1 + var2 
                
                if error <= bestError: 
                    bestError = error 
                    bestFeatureInd = featureInd
                    bestThreshold = val 
                    
    bestLeftInds = numpy.sort(nodeInds[numpy.arange(nodeInds.shape[0])[X[:, bestFeatureInd][nodeInds]<bestThreshold]]) 
    bestRightInds = numpy.sort(nodeInds[numpy.arange(nodeInds.shape[0])[X[:, bestFeatureInd][nodeInds]>=bestThreshold]])
    
    return bestError, bestFeatureInd, bestThreshold, bestLeftInds, bestRightInds
    
    
def findBestSplit3(int minSplit, numpy.ndarray[numpy.float_t, ndim=2, mode="fortran"] X, numpy.ndarray[numpy.float_t, ndim=1, mode="c"] y,  numpy.ndarray[numpy.int_t, ndim=1, mode="c"] nodeInds,  numpy.ndarray[numpy.int_t, ndim=2, mode="fortran"] argsortX): 
    #print 1st col
    
    cdef numpy.float_t *x = NULL 
    cdef numpy.int_t *argsort_x = NULL 
    cdef int X_elem_stride = X.strides[0]
    cdef int X_col_stride = X.strides[1]
    cdef int X_stride = X_col_stride / X_elem_stride
    cdef int X_argsorted_elem_stride = argsortX.strides[0]
    cdef int X_argsorted_col_stride = argsortX.strides[1]
    cdef int X_argsorted_stride = X_argsorted_col_stride / X_argsorted_elem_stride
        
    #Loop 
    cdef float error, var1, var2, val, currentValue, cumYFinal, cumY2Final, tempYVal, tempY2Val     
    cdef numpy.ndarray[numpy.float_t, ndim=1] tempX = numpy.zeros(X.shape[0]), tempY = numpy.zeros(X.shape[0])
    cdef numpy.ndarray[numpy.float_t, ndim=1] tempX2 = numpy.zeros(nodeInds.shape[0]), tempY2 = numpy.zeros(nodeInds.shape[0])
    cdef numpy.ndarray[numpy.float_t, ndim=1] cumY = numpy.zeros(nodeInds.shape[0]), cumY2 = numpy.zeros(nodeInds.shape[0])
    cdef numpy.ndarray[numpy.int8_t, ndim=1] boolInds = numpy.zeros(X.shape[0], dtype=numpy.int8)
    cdef unsigned int rightSize, insertInd, featureInd, i, j, finalInd, variables, insertIndp1
    #Store unique values 
    cdef numpy.ndarray[numpy.int_t, ndim=1] sortedNodeInds = numpy.zeros(nodeInds.shape[0], dtype=int)
    cdef numpy.ndarray[numpy.float_t, ndim=1] uniqueVals = numpy.zeros(X.shape[0])
    #cdef int lastUniqueVal = 0
    cdef float tol = 10**-3   
    cdef unsigned int numInds = nodeInds.shape[0]
    
    #Best values 
    cdef float bestError = float("inf")   
    cdef unsigned int bestFeatureInd = 0 
    cdef float bestThreshold = X[:, bestFeatureInd].min() 
    cdef numpy.ndarray[numpy.int_t, ndim=1] bestLeftInds = numpy.array([0]), bestRightInds  = numpy.array([0])
    
    for featureInd in range(X.shape[1]): 
        #x = X[:, featureInd] 
        x = (<numpy.float_t *>X.data) + X_stride * featureInd
        argsort_x = (<numpy.int_t *>argsortX.data) + X_argsorted_stride * featureInd

        tempX.fill(0) 
        tempY.fill(0)
        boolInds.fill(0)
                
        j = 0 
        #Sort by values of x 
        for i in nodeInds:        
            tempX[argsort_x[i]] = x[i]
            tempY[argsort_x[i]] = y[i]
            sortedNodeInds[j] = argsort_x[i]  
            j += 1
        
     
        
        for i in range(j): 
            boolInds[sortedNodeInds[i]] = 1
        
        k = 0 
        sumY = 0 
        sumY2 = 0
        for i in range(X.shape[0]):
            if boolInds[i] == 1: 
                tempYVal = tempY[i]
                tempY2Val = tempY[i]**2
                tempX2[k] = tempX[i]
                tempY2[k] = tempY[i]
                cumY[k] = sumY + tempYVal
                cumY2[k] = sumY2 + tempY2Val
                
                k += 1
                sumY += tempYVal
                sumY2 += tempY2Val
                        
        #assert (numpy.linalg.norm(cumY - tempY2.cumsum()) <= tol), "%f" % numpy.linalg.norm(cumY - tempY2.cumsum())           
        #assert (numpy.linalg.norm(cumY2 - (tempY2**2).cumsum()) <= tol), "%f" % numpy.linalg.norm(cumY2 - (tempY2**2).cumsum())  
        
        currentValue = tempX2[0]
        finalInd = cumY2.shape[0]-1
        cumYFinal = cumY[finalInd]
        cumY2Final = cumY2[finalInd]
        
        for insertInd in range(numInds-1): 
            if insertInd < minSplit or insertInd < minSplit: 
                continue 
            
            val = tempX2[insertInd]
            insertIndp1 = insertInd+1
            rightSize = (numInds - insertIndp1)
            
            if insertInd!=1 and insertInd!=numInds: 
                cumYVal = cumY[insertInd]
                cumY2Val = cumY2[insertInd]
                var1 = cumY2Val - (cumYVal**2)/float(insertIndp1)
                var2 = (cumY2Final-cumY2Val) - (cumYFinal-cumYVal)**2/float(rightSize)
                
                error = var1 + var2 
                
                if error <= bestError: 
                    bestError = error 
                    bestFeatureInd = featureInd
                    bestThreshold = (val + tempX2[insertIndp1])/2
                    
    bestLeftInds = numpy.sort(nodeInds[numpy.arange(nodeInds.shape[0])[X[:, bestFeatureInd][nodeInds]<bestThreshold]]) 
    bestRightInds = numpy.sort(nodeInds[numpy.arange(nodeInds.shape[0])[X[:, bestFeatureInd][nodeInds]>=bestThreshold]])
    
    return bestError, bestFeatureInd, bestThreshold, bestLeftInds, bestRightInds