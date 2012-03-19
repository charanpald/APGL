import numpy 

def findBestSplit2(minSplit, X, y, nodeInds, argsortX): 
    """
    Give a set of examples and a particular feature, find the best split 
    of the data. This is a pure python version. Note that nodeInds must be 
    sorted. 
    """
    if X.shape[0] == 0: 
        raise ValueError("Cannot split on 0 examples")
        
    #nodeInds = numpy.sort(nodeInds)
    tol = 0.01
        
    bestError = float("inf")   
    bestFeatureInd = 0 
    bestThreshold = X[:, bestFeatureInd].min() 
    
    for featureInd in range(X.shape[1]): 
        x = X[:, featureInd] 

        tempX = numpy.zeros(X.shape[0])
        tempY = numpy.zeros(X.shape[0])
        
        tempX[argsortX[:, featureInd][nodeInds]] = x[nodeInds]
        tempY[argsortX[:, featureInd][nodeInds]] = y[nodeInds]
        
        boolInds = numpy.zeros(X.shape[0], numpy.bool)
        boolInds[argsortX[:, featureInd][nodeInds]] = True
        
        tempX = tempX[boolInds]       
        tempY = tempY[boolInds]   
        
        #argsInds = numpy.argsort(X[nodeInds, featureInd])
        #tempX2 = X[nodeInds, featureInd][argsInds]
        #tempY2 = y[nodeInds][argsInds]
        
        #assert (numpy.linalg.norm(tempX - tempX2) < tol), "%s %s, %s" % (str(tempX), str(tempX2), str(boolInds)) 
        #assert (numpy.linalg.norm(tempY - tempY2) < tol)
        
        vals = numpy.unique(tempX)
        vals = (vals[1:]+vals[0:-1])/2.0
        
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

            if insertInd!=1 and insertInd!=nodeInds.shape[0]: 
                cumYVal = cumY[insertInd-1]
                cumY2Val = cumY2[insertInd-1]
                var1 = cumY2Val - (cumYVal**2)/float(insertInd)
                var2 = (cumY2[-1]-cumY2Val) - (cumY[-1]-cumYVal)**2/float(tempX.shape[0] - insertInd)
   
                assert abs(var1 - y[nodeInds][x[nodeInds]<val].var()*y[nodeInds][x[nodeInds]<val].shape[0]) < tol, "%f %f" % (var1, y[nodeInds][x[nodeInds]<val].var()*y[nodeInds][x[nodeInds]<val].shape[0])
                assert abs(var2 - y[nodeInds][x[nodeInds]>=val].var()*y[nodeInds][x[nodeInds]>=val].shape[0]) < tol 
                
                error = var1 + var2 
                
                if error <= bestError: 
                    bestError = error 
                    bestFeatureInd = featureInd
                    bestThreshold = val 
    
    bestLeftInds = numpy.sort(nodeInds[numpy.arange(nodeInds.shape[0])[X[:, bestFeatureInd][nodeInds]<bestThreshold]]) 
    bestRightInds = numpy.sort(nodeInds[numpy.arange(nodeInds.shape[0])[X[:, bestFeatureInd][nodeInds]>=bestThreshold]])
    
    assert (numpy.union1d(bestLeftInds, bestRightInds) == nodeInds).all(), "%s %s, %s" % (str(bestLeftInds), str(bestRightInds), str(nodeInds.shape)) 
    
    return bestError, bestFeatureInd, bestThreshold, bestLeftInds, bestRightInds