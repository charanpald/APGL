import numpy 
from apgl.util.Util import Util 

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
    
    
def findBestSplitRand(minSplit, X, y, nodeInds, argsortX): 
    """
    Give a set of examples and a particular feature, find the best split 
    of the data using information gain. This is a pure python version. Note that 
    nodeInds must be sorted. We weight each node based on the probability of being random. 
    """
    if X.shape[0] == 0: 
        raise ValueError("Cannot split on 0 examples")
        
    #nodeInds = numpy.sort(nodeInds)
    tol = 0.01
        
    gains = numpy.zeros(X.shape[1])
    thresholds = numpy.zeros(X.shape[1])
    
    for featureInd in range(X.shape[1]): 
        gains[featureInd] = 0 
        x = X[:, featureInd] 

        tempX = numpy.zeros(X.shape[0])
        tempY = numpy.zeros(y.shape[0], y.dtype)
        
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
        counts = numpy.array(numpy.bincount(tempY), numpy.float)
        counts /= counts.sum()            
        tempCounts = counts + (counts == 0)
        parentEntropy = -(counts * numpy.log2(tempCounts)).sum()     
        
        vals = numpy.unique(tempX)
        vals = (vals[1:]+vals[0:-1])/2.0
        
        insertInds = numpy.searchsorted(tempX, vals)
        
        for i in range(vals.shape[0]): 
            val = vals[i]
            #Find index where val will be inserted before to preserve order 
            insertInd = insertInds[i]
            
            rightSize = (tempX.shape[0] - insertInd)
            if insertInd < minSplit or rightSize < minSplit: 
                continue 

            if insertInd!=1 and insertInd!=nodeInds.shape[0]: 
                counts = numpy.array(numpy.bincount(tempY[tempX<val]), numpy.float)
                counts /= counts.sum()  
                tempCounts1 = counts + (counts == 0)
                entropy1 = -(counts * numpy.log2(tempCounts1)).sum()                
                
                counts = numpy.array(numpy.bincount(tempY[tempX>=val]), numpy.float)
                counts /= counts.sum()  
                tempCounts2 = counts + (counts == 0)
                entropy2 = -(counts * numpy.log2(tempCounts2)).sum()
                
                gain = parentEntropy - (tempCounts1.shape[0]*entropy1 + tempCounts2.shape[0]*entropy2)/tempY.shape[0]
                if gain >= gains[featureInd]: 
                    gains[featureInd] = gain 
                    thresholds[featureInd] = val 
    
    return gains, thresholds
    
def findBestSplitRisk(minSplit, X, y, nodeInds, argsortX): 
    """
    Give a set of examples and a particular feature, find the best split 
    of the data according to which minimises the risk. This is a pure python version. Note that 
    nodeInds must be sorted. 
    """
    if X.shape[0] == 0: 
        raise ValueError("Cannot split on 0 examples")
        
    #nodeInds = numpy.sort(nodeInds)        
    accuracies = numpy.zeros(X.shape[1])
    thresholds = numpy.zeros(X.shape[1])
    minY = numpy.min(y)
    
    for featureInd in range(X.shape[1]): 
        accuracies[featureInd] = 0 
        x = X[:, featureInd] 

        tempX = numpy.zeros(X.shape[0])
        tempY = numpy.zeros(y.shape[0], y.dtype)
        
        tempX[argsortX[:, featureInd][nodeInds]] = x[nodeInds]
        tempY[argsortX[:, featureInd][nodeInds]] = y[nodeInds]
        
        boolInds = numpy.zeros(X.shape[0], numpy.bool)
        boolInds[argsortX[:, featureInd][nodeInds]] = True
        
        tempX = tempX[boolInds]       
        tempY = tempY[boolInds]   
                
        vals = numpy.unique(tempX)
        vals = (vals[1:]+vals[0:-1])/2.0
        
        insertInds = numpy.searchsorted(tempX, vals)
        parentAccuracy = numpy.max(numpy.bincount(tempY - minY))
        
        for i in range(vals.shape[0]): 
            val = vals[i]
            #Find index where val will be inserted before to preserve order 
            insertInd = insertInds[i]
            
            rightSize = (tempX.shape[0] - insertInd)
            if insertInd < minSplit or rightSize < minSplit: 
                continue 

            if insertInd!=1 and insertInd!=nodeInds.shape[0]: 
                accuracy1 = numpy.max(numpy.bincount(tempY[tempX<val] - minY))
                accuracy2 = numpy.max(numpy.bincount(tempY[tempX>=val] - minY))
 
                totalAccuracy = (accuracy1 + accuracy2 - parentAccuracy)/float(tempY.shape[0])
                if totalAccuracy >= accuracies[featureInd] and totalAccuracy > 0: 
                    accuracies[featureInd] = totalAccuracy 
                    thresholds[featureInd] = val 
    
    return accuracies, thresholds