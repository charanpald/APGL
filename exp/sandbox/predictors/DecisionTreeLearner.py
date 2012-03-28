import numpy 
from apgl.predictors.AbstractPredictor import AbstractPredictor
from apgl.graph.DictTree import DictTree
from exp.sandbox.predictors.TreeCriterion import findBestSplit
from exp.sandbox.predictors.TreeCriterionPy import findBestSplit2
from exp.sandbox.predictors.DecisionNode import DecisionNode


    
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
        
        if self.tree.isLeaf(nodeId): 
            y[testInds] = node.getValue()
        else: 
             
            leftChildId = self.getLeftChildId(nodeId)
            if self.tree.vertexExists(leftChildId):
                leftChild = self.tree.getVertex(leftChildId)
                leftChildInds = X[testInds, node.getFeatureInd()] < node.getThreshold() 
                tempInds = testInds[leftChildInds]
                leftChild.setTestInds(tempInds)
                
                y = self.recursivePredict(X, y, leftChildId)
                
            rightChildId = self.getRightChildId(nodeId)
            if self.tree.vertexExists(rightChildId): 
                rightChild = self.tree.getVertex(rightChildId)
                rightChildInds = X[testInds, node.getFeatureInd()] >= node.getThreshold()
                tempInds = testInds[rightChildInds]
                rightChild.setTestInds(tempInds)
                y = self.recursivePredict(X, y, rightChildId)
                
        return y
        
    def prune(self, validX, validY, alphaThreshold=0): 
        """
        Prune the decision tree using reduced error pruning. 
        """
        rootId = (0,)
        self.tree.getVertex(rootId).setTestInds(numpy.arange(validX.shape[0]))
        self.recursiveSetPrune(validX, validY, rootId)        
        
        for vertexId in self.tree.getAllVertexIds(): 
            currentNode = self.tree.getVertex(vertexId)            
            subtreeLeaves = self.tree.leaves(vertexId)

            testErrorSum = 0 
            for leaf in subtreeLeaves: 
                testErrorSum += self.tree.getVertex(leaf).getTestError()
            
            #Alpha is normalised difference in error 
            currentNode.alpha = (testErrorSum - currentNode.getTestError())/float(currentNode.getTestInds().shape[0])
                  
        self.recursivePrune(rootId, alphaThreshold)
        
    def recursiveSetPrune(self, X, y, nodeId):
        """
        This computes test errors on nodes by passing in the test X and y. 
        """
        node = self.tree.getVertex(nodeId)
        testInds = node.getTestInds()
        node.setTestError(numpy.sum((y[testInds] - node.getValue())**2))
    
        leftChildId = self.getLeftChildId(nodeId)
        if self.tree.vertexExists(leftChildId):
            leftChild = self.tree.getVertex(leftChildId)
            leftChildInds = X[testInds, node.getFeatureInd()] < node.getThreshold() 
            leftChild.setTestInds(testInds[leftChildInds])
            self.recursiveSetPrune(X, y, leftChildId)
            
        rightChildId = self.getRightChildId(nodeId)
        if self.tree.vertexExists(rightChildId): 
            rightChild = self.tree.getVertex(rightChildId)
            rightChildInds = X[testInds, node.getFeatureInd()] >= node.getThreshold()
            rightChild.setTestInds(testInds[rightChildInds])
            self.recursiveSetPrune(X, y, rightChildId)
        
    def recursivePrune(self, nodeId, alphaThresh): 
        """
        We compute alpha values and prune as early as possible.   
        """
        node = self.tree.getVertex(nodeId)

        if node.alpha > alphaThresh: 
            self.tree.pruneVertex(nodeId)
        else: 
            leftChildId = self.getLeftChildId(nodeId)
            if self.tree.vertexExists(leftChildId):
                self.recursivePrune(leftChildId, alphaThresh)
                
            rightChildId = self.getRightChildId(nodeId)
            if self.tree.vertexExists(rightChildId): 
                self.recursivePrune(rightChildId, alphaThresh)
    
        
        
        
        
        
        