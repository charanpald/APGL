import numpy 

class DecisionNode(object): 
    def __init__(self, trainInds, value): 
        #All nodes 
        self.value = value 
        self.trainInds = trainInds
        #Internal nodes 
        self.featureInd = None 
        self.threshold = None 
        self.error = None 
        #Used for making predictions 
        self.testInds = None 
        self.testError = None
        #Used for pruning 
        self.alpha = 0 

    def setTrainInds(self, trainInds): 
        self.trainInds = trainInds        
        
    def getTrainInds(self): 
        return self.trainInds
        
    def getValue(self): 
        return self.value 
        
    def setError(self, error): 
        """
        The training error for internal nodes. 
        """
        self.error = error 
    
    def getError(self): 
        return self.error 
        
    def setFeatureInd(self, featureInd): 
        self.featureInd = featureInd 
        
    def getFeatureInd(self): 
        return self.featureInd 
        
    def setThreshold(self, threshold): 
        self.threshold = threshold 
        
    def getThreshold(self): 
        return self.threshold
        
    def setValue(self, value): 
        self.value = value 
          
    def setTestInds(self, testInds): 
        self.testInds = testInds
        
    def getTestInds(self): 
        return self.testInds
        
    def setTestError(self, testError): 
        self.testError = testError 
        
    def getTestError(self): 
        return self.testError 
        
    def setAlpha(self, alpha): 
        self.alpha = alpha 
            
    def __str__(self): 
        outputStr = "Size: " + str(self.trainInds.shape[0]) + ", " 
        outputStr += "featureInd: " + str(self.featureInd) + ", " 
        if self.threshold != None: 
            outputStr += "threshold: %.3f" % self.threshold + ", "
        if self.error != None: 
            outputStr += "err: %.3f" % self.error + ", "
        outputStr += "val: %.3f" % self.value + " "
        outputStr += "alpha: %05f" % self.alpha + " "
        return outputStr 