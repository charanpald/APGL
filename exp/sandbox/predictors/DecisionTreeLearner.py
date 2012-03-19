import numpy 
from apgl.predictors.AbstractPredictor import AbstractPredictor
from apgl.graph.DictTree import DictTree
from exp.sandbox.predictors.TreeCriterion import findBestSplit
from exp.sandbox.predictors.TreeCriterionPy import findBestSplit2


class DecisionNode(): 
    def __init__(self, trainInds, value): 
        #All nodes 
        self.value = value 
        self.trainInds = trainInds
        #Internal nodes 
        self.featureInd = None 
        self.threshold = None 
        self.error = None 
        #Used for sorting predictions 
        self.testInds = None 
        
    def getTrainInds(self): 
        return self.trainInds
        
    def getValue(self): 
        return self.value 
        
    def setError(self, error): 
        self.error = error 
        
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
    
    def isLeaf(self): 
        return self.error == None
        
    def __str__(self): 
        outputStr = "Size: " + str(self.trainInds.shape[0]) + ", " 
        outputStr += "featureInd: " + str(self.featureInd) + ", " 
        outputStr += "threshold: " + str(self.threshold) + ", "
        outputStr += "error: " + str(self.error) + ", "
        outputStr += "value: " + str(self.value) + " "
        return outputStr 
    
class DecisionTreeLearner(AbstractPredictor): 
    def __init__(self, criterion="mse", maxDepth=10, minSplit=30, type="class"):
        """
        Need a minSplit for the internal nodes and one for leaves. 
        """
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
        if y1.shape[0]==0 or y2.shape[0]==0: 
            raise ValueError("Cannot work with one-sided split")
        error = y1.shape[0]*y1.var() + y2.shape[0]*y2.var()  
        return error 
         
    #@profile 
    def learnModel(self, X, y):
        nodeId = (0, )         
        self.tree = DictTree()
        rootNode = DecisionNode(numpy.arange(X.shape[0]), y.mean())
        self.tree.setVertex(nodeId, rootNode)

        #We computer a sorted version of X 
        argsortX = numpy.zeros(X.shape, numpy.int)
        for i in range(X.shape[1]): 
            argsortX[:, i] = numpy.argsort(X[:, i])
            argsortX[:, i] = numpy.argsort(argsortX[:, i])
        
        self.recursiveSplit(X, y, argsortX, nodeId)
     
    def getLeftChildId(self, nodeId): 
        leftChildId = list(nodeId)
        leftChildId.append(0)
        leftChildId = tuple(leftChildId)
        return leftChildId

    def getRightChildId(self, nodeId): 
        rightChildId = list(nodeId)
        rightChildId.append(1)
        rightChildId = tuple(rightChildId) 
        return rightChildId
   
    def getTree(self): 
        return self.tree 
   
    #@profile 
    def recursiveSplit(self, X, y, argsortX, nodeId): 
        """
        Give a sample of data and a node index, we find the best split and 
        add children to the tree accordingly. 
        """
        if len(nodeId)-1 >= self.maxDepth: 
            return 
        
        node = self.tree.getVertex(nodeId)
        bestError, bestFeatureInd, bestThreshold, bestLeftInds, bestRightInds = findBestSplit(self.minSplit, X, y, node.getTrainInds(), argsortX)
    
        #The split may have 0 items in one set, so don't split 
        if bestLeftInds.sum() != 0 and bestRightInds.sum() != 0: 
            node.setError(bestError)
            node.setFeatureInd(bestFeatureInd)
            node.setThreshold(bestThreshold)
            
            leftChildId = self.getLeftChildId(nodeId)
            rightChildId = self.getRightChildId(nodeId)

            leftChild = DecisionNode(bestLeftInds, y[bestLeftInds].mean())
            self.tree.addChild(nodeId, leftChildId, leftChild)
            
            rightChild = DecisionNode(bestRightInds, y[bestRightInds].mean())
            self.tree.addChild(nodeId, rightChildId, rightChild)
            
            if leftChild.getTrainInds().shape[0] >= self.minSplit: 
                self.recursiveSplit(X, y, argsortX, leftChildId)
                
            if rightChild.getTrainInds().shape[0] >= self.minSplit: 
                self.recursiveSplit(X, y, argsortX, rightChildId)
        
    def predict(self, X): 
        """
        Make a prediction for the set of examples given in the matrix X. 
        """
        rootId = (0,)
        predY = numpy.zeros(X.shape[0])
        self.tree.getVertex(rootId).setTestInds(numpy.arange(X.shape[0]))
        predY = self.recursivePredict(X, predY, rootId)
        
        return predY 
        
    def recursivePredict(self, X, y, nodeId): 
        node = self.tree.getVertex(nodeId)
        testInds = node.getTestInds()
        
        if node.isLeaf(): 
            y[testInds] = node.getValue()
        else: 
             
            leftChildId = self.getLeftChildId(nodeId)
            if self.tree.vertexExists(leftChildId):
                leftChild = self.tree.getVertex(leftChildId)
                leftChildInds = X[testInds, node.getFeatureInd()] < node.getThreshold() 
                leftChild.setTestInds(testInds[leftChildInds])
                y = self.recursivePredict(X, y, leftChildId)
                
            rightChildId = self.getRightChildId(nodeId)
            if self.tree.vertexExists(rightChildId): 
                rightChild = self.tree.getVertex(rightChildId)
                rightChildInds = X[testInds, node.getFeatureInd()] >= node.getThreshold()
                rightChild.setTestInds(testInds[rightChildInds])
                y = self.recursivePredict(X, y, rightChildId)
                
        return y
        
        
        
        
        
        