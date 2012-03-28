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
            
    def __str__(self): 
        outputStr = "Size: " + str(self.trainInds.shape[0]) + ", " 
        outputStr += "featureInd: " + str(self.featureInd) + ", " 
        outputStr += "threshold: " + str(self.threshold) + ", "
        outputStr += "err: " + str(self.error) + ", "
        outputStr += "val: " + str(self.value) + " "
        return outputStr 