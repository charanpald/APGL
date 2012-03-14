import numpy 
from apgl.predictors.AbstractPredictor import AbstractPredictor
from apgl.graph.DictTree import DictTree


class DecisionNode(): 
    def __init__(self, exampleInds): 
        self.featureInd = None 
        self.threshold = None 
        self.value = None 
        self.error = None 
        self.exampleInds = exampleInds 
        
    def getExampleInds(self): 
        return self.exampleInds
        
    def setError(self, error): 
        self.error = error 
        
    def setFeatureInd(self, featureInd): 
        self.featureInd = featureInd 
        
    def setThreshold(self, threshold): 
        self.threshold = threshold 
        
    def __str__(self): 
        outputStr = "Size: " + str(self.exampleInds.shape[0]) + ", " 
        outputStr += "Featureind: " + str(self.featureInd) + ", " 
        outputStr += "threshold: " + str(self.threshold) + ", "
        outputStr += "error: " + str(self.error) + " "
        return outputStr 
    

class DecisionTreeLearner(AbstractPredictor): 
    def __init__(self, criterion="mse", maxDepth=10, minSplit=30, type="class"):
        super(DecisionTreeLearner, self).__init__()
        self.maxDepth = maxDepth
        self.minSplit = minSplit
        self.criterion = criterion
        self.type = type
        
        self.maxDepths = numpy.arange(1, 10)
        self.minSplits = numpy.arange(10, 51, 10)
        
    def meanSqError(self, y1, y2): 
        """
        Given a split that results in the labels y1 \in \mathbb^n1, y2 \in \mathbb{R}^n2, 
        compute the impurity using the mean squared error. This is just computed 
        as ||y1 - 1/n1 y1^T 1 1^T||^2 + ||y2 - 1/n2 y2^T 1 1^T||^2. 
        """
        error = y1.shape[0]*y1.var() + y2.shape[0]*y2.var()  
        return error 
        
    def findBestSplit(self, X, y): 
        """
        Give a set of examples and a particular feature, find the best split 
        of the data. 
        """
        if X.shape[0] == 0: 
            raise ValueError("Cannot split on 0 examples")
        
        bestError = float("inf")   
        
        for featureInd in range(X.shape[1]): 
            x = X[:, featureInd] 
            vals = numpy.unique(x)

            for val in vals: 
                inds1 = x<val
                inds2 = x>=val
                
                error = self.meanSqError(y[inds1], y[inds2])

                if error < bestError: 
                    bestError = error 
                    bestFeatureInd = featureInd
                    bestThreshold = val 
                    bestSplitInds = (inds1, inds2)
                        
        return bestError, bestFeatureInd, bestThreshold, bestSplitInds 
    
    def learnModel(self, X, y):
        #Let's create a tree 

        nodeId = (0, )         
        self.tree = DictTree()
        self.tree.setVertex(nodeId, DecisionNode(numpy.arange(X.shape[0])))
        self.recursiveSplit(X, y, nodeId)
        
    def recursiveSplit(self, X, y, nodeId): 
        """
        Give a sample of data and a node index, we find the best split and 
        add children to the tree accordingly. 
        """
        node = self.tree.getVertex(nodeId)
        tempX = X[node.getExampleInds(), :]
        tempY = y[node.getExampleInds()]

        print(node.getExampleInds().shape[0])

        bestError, bestFeatureInd, bestThreshold, bestSplitInds = self.findBestSplit(tempX, tempY)
    
        #The split may have 0 items in one set, so don't split 
        if bestSplitInds[0].sum() != 0 and bestSplitInds[1].sum() != 0: 
            node.setError(bestError)
            node.setFeatureInd(bestFeatureInd)
            node.setThreshold(bestThreshold)
            
            leftChildId = list(nodeId)
            leftChildId.append(0)
            leftChildId = tuple(leftChildId)
            
            rightChildId = list(nodeId)
            rightChildId.append(1)
            rightChildId = tuple(rightChildId) 
            
            leftChild = DecisionNode(node.getExampleInds()[bestSplitInds[0]])
            self.tree.addChild(nodeId, leftChildId, leftChild)
            
            rightChild = DecisionNode(node.getExampleInds()[bestSplitInds[1]])
            self.tree.addChild(nodeId, rightChildId, rightChild)
            
            if leftChild.getExampleInds().shape[0] >= self.minSplit: 
                self.recursiveSplit(X, y, leftChildId)
                
            if rightChild.getExampleInds().shape[0] >= self.minSplit: 
                self.recursiveSplit(X, y, rightChildId)
        
                
        
        
        
        
        
        
        
        