import numpy 
from apgl.predictors.AbstractPredictor import AbstractPredictor
from apgl.graph.DictTree import DictTree
from exp.sandbox.predictors.TreeCriterion import findBestSplit
from exp.sandbox.predictors.TreeCriterionPy import findBestSplit2
from exp.sandbox.predictors.DecisionNode import DecisionNode
from apgl.util.Sampling import Sampling
from apgl.util.Parameter import Parameter
from apgl.util.Evaluator import Evaluator
    
class DecisionTreeLearner(AbstractPredictor): 
    def __init__(self, criterion="mse", maxDepth=10, minSplit=30, type="reg", pruneType="none", alphaThreshold=0.0, folds=5):
        """
        Need a minSplit for the internal nodes and one for leaves. 
        """
        super(DecisionTreeLearner, self).__init__()
        self.maxDepth = maxDepth
        self.minSplit = minSplit
        self.criterion = criterion
        self.type = type
        self.pruneType = pruneType 
        self.alphaThreshold = alphaThreshold
        self.folds = 5
    
    def learnModel(self, X, y):
        nodeId = (0, )         
        self.tree = DictTree()
        rootNode = DecisionNode(numpy.arange(X.shape[0]), y.mean())
        self.tree.setVertex(nodeId, rootNode)

        #We compute a sorted version of X 
        argsortX = numpy.zeros(X.shape, numpy.int)
        for i in range(X.shape[1]): 
            argsortX[:, i] = numpy.argsort(X[:, i])
            argsortX[:, i] = numpy.argsort(argsortX[:, i])
        
        self.growSkLearn(X, y)
        #self.recursiveSplit(X, y, argsortX, nodeId)
        
        if self.pruneType == "REP": 
            self.repPrune(X, y)
        elif self.pruneType == "REP-CV":
            self.cvPrune(X, y)
        elif self.pruneType == "none": 
            pass
        else:
            raise ValueError("Unknown pruning type " + self.pruneType)
     
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
            leftChild = DecisionNode(bestLeftInds, y[bestLeftInds].mean())
            self.tree.addChild(nodeId, leftChildId, leftChild)
            
            if leftChild.getTrainInds().shape[0] >= self.minSplit: 
                self.recursiveSplit(X, y, argsortX, leftChildId)
            
            rightChildId = self.getRightChildId(nodeId)
            rightChild = DecisionNode(bestRightInds, y[bestRightInds].mean())
            self.tree.addChild(nodeId, rightChildId, rightChild)
            
            if rightChild.getTrainInds().shape[0] >= self.minSplit: 
                self.recursiveSplit(X, y, argsortX, rightChildId)
    
    def growSkLearn(self, X, y): 
        """
        Grow a decision tree from sklearn. 
        """
        
        from sklearn.tree import DecisionTreeRegressor
        regressor = DecisionTreeRegressor(max_depth = self.maxDepth, min_samples_split=self.minSplit)
        regressor.fit(X, y)
        
        #Convert the sklearn tree into our tree 
        nodeId = (0, )          
        nodeStack = [(nodeId, 0)] 
        
        node = DecisionNode(numpy.arange(X.shape[0]), regressor.tree_.value[0])
        self.tree.setVertex(nodeId, node)
        
        while len(nodeStack) != 0: 
            nodeId, nodeInd = nodeStack.pop()
            
            node = self.tree.getVertex(nodeId)
            node.setError(regressor.tree_.best_error[nodeInd])
            node.setFeatureInd(regressor.tree_.feature[nodeInd])
            node.setThreshold(regressor.tree_.threshold[nodeInd])
                
            if regressor.tree_.children[nodeInd, 0] != -1: 
                leftChildInds = node.getTrainInds()[X[node.getTrainInds(), node.getFeatureInd()] < node.getThreshold()] 
                leftChildId = self.getLeftChildId(nodeId)
                leftChild = DecisionNode(leftChildInds, regressor.tree_.value[regressor.tree_.children[nodeInd, 0]])
                self.tree.addChild(nodeId, leftChildId, leftChild)
                nodeStack.append((self.getLeftChildId(nodeId), regressor.tree_.children[nodeInd, 0]))
                
            if regressor.tree_.children[nodeInd, 1] != -1: 
                rightChildInds = node.getTrainInds()[X[node.getTrainInds(), node.getFeatureInd()] >= node.getThreshold()]
                rightChildId = self.getRightChildId(nodeId)
                rightChild = DecisionNode(rightChildInds, regressor.tree_.value[regressor.tree_.children[nodeInd, 1]])
                self.tree.addChild(nodeId, rightChildId, rightChild)
                nodeStack.append((self.getRightChildId(nodeId), regressor.tree_.children[nodeInd, 1]))

    
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
        """
        Recurse through the tree and assign examples to the correct vertex. 
        """        
        node = self.tree.getVertex(nodeId)
        testInds = node.getTestInds()
        
        if self.tree.isLeaf(nodeId): 
            y[testInds] = node.getValue()
        else: 
             
            for childId in [self.getLeftChildId(nodeId), self.getRightChildId(nodeId)]:
                if self.tree.vertexExists(childId):
                    child = self.tree.getVertex(childId)
    
                    if childId[-1] == 0: 
                        childInds = X[testInds, node.getFeatureInd()] < node.getThreshold() 
                    else:
                        childInds = X[testInds, node.getFeatureInd()] >= node.getThreshold()
                    
                    child.setTestInds(testInds[childInds])   
                    y = self.recursivePredict(X, y, childId)
                
        return y
        
    def recursiveSetPrune(self, X, y, nodeId):
        """
        This computes test errors on nodes by passing in the test X and y. 
        """
        node = self.tree.getVertex(nodeId)
        testInds = node.getTestInds()
        node.setTestError(self.vertexTestError(y[testInds], node.getValue()))
    
        for childId in [self.getLeftChildId(nodeId), self.getRightChildId(nodeId)]:
            if self.tree.vertexExists(childId):
                child = self.tree.getVertex(childId)
                
                if childId[-1] == 0: 
                    childInds = X[testInds, node.getFeatureInd()] < node.getThreshold() 
                else:
                    childInds = X[testInds, node.getFeatureInd()] >= node.getThreshold()
                child.setTestInds(testInds[childInds])
                self.recursiveSetPrune(X, y, childId)
    
    def vertexTestError(self, trueY, predY):
        """
        This is the error used for pruning. We compute it at each node. 
        """
        return numpy.sum((trueY - predY)**2)
    
    def computeAlphas(self): 
        for vertexId in self.tree.getAllVertexIds(): 
            currentNode = self.tree.getVertex(vertexId)
            subtreeLeaves = self.tree.leaves(vertexId)

            testErrorSum = 0 
            for leaf in subtreeLeaves: 
                testErrorSum += self.tree.getVertex(leaf).getTestError()
            
            #Alpha is normalised difference in error 
            if currentNode.getTestInds().shape[0] != 0: 
                currentNode.alpha = (testErrorSum - currentNode.getTestError())/float(currentNode.getTestInds().shape[0])        
        
    def repPrune(self, validX, validY): 
        """
        Prune the decision tree using reduced error pruning. 
        """
        rootId = (0,)
        self.tree.getVertex(rootId).setTestInds(numpy.arange(validX.shape[0]))
        self.recursiveSetPrune(validX, validY, rootId)        
        self.computeAlphas()        
        self.recursivePrune(rootId)
                            
    def recursivePrune(self, nodeId): 
        """
        We prune as early as possible.   
        """
        node = self.tree.getVertex(nodeId)

        if node.alpha > self.alphaThreshold: 
            self.tree.pruneVertex(nodeId)
        else: 
            for childId in [self.getLeftChildId(nodeId), self.getRightChildId(nodeId)]: 
                if self.tree.vertexExists(childId):
                    self.recursivePrune(childId)
                
    def cvPrune(self, validX, validY): 
        """
        We do something like reduced error pruning but we use cross validation 
        to decide which nodes to prune. 
        """
        
        #First set the value of the vertices using the training set. 
        #Reset all alphas to zero 
        inds = Sampling.crossValidation(self.folds, validX.shape[0])
        
        for i in self.tree.getAllVertexIds(): 
            self.tree.getVertex(i).setAlpha(0.0)
            self.tree.getVertex(i).setTestError(0.0)
        
        for trainInds, testInds in inds:             
            rootId = (0,)
            root = self.tree.getVertex(rootId)
            root.setTrainInds(trainInds)
            root.setTestInds(testInds)
            root.tempValue = numpy.mean(validY[trainInds])
            
            nodeStack = [(rootId, root.tempValue)]
            
            while len(nodeStack) != 0: 
                (nodeId, value) = nodeStack.pop()
                node = self.tree.getVertex(nodeId)
                tempTrainInds = node.getTrainInds()
                tempTestInds = node.getTestInds()
                node.setTestError(numpy.sum((validY[tempTestInds] - node.tempValue)**2) + node.getTestError())
                childIds = [self.getLeftChildId(nodeId), self.getRightChildId(nodeId)]
                
                for childId in childIds:                 
                    if self.tree.vertexExists(childId): 
                        child = self.tree.getVertex(childId)
                        
                        if childId[-1] == 0: 
                            childInds = validX[tempTrainInds, node.getFeatureInd()] < node.getThreshold()
                        else: 
                            childInds = validX[tempTrainInds, node.getFeatureInd()] >= node.getThreshold()
                        
                        if childInds.sum() !=0:   
                            value = numpy.mean(validY[tempTrainInds[childInds]])
                            
                        child.tempValue = value 
                        child.setTrainInds(tempTrainInds[childInds])
                        nodeStack.append((childId, value))
                        
                        if childId[-1] == 0: 
                            childInds = validX[tempTestInds, node.getFeatureInd()] < node.getThreshold() 
                        else: 
                            childInds = validX[tempTestInds, node.getFeatureInd()] >= node.getThreshold()  
                         
                        child.setTestInds(tempTestInds[childInds])
        
        self.computeAlphas()
        self.recursivePrune(rootId)
        
    def copy(self): 
        """
        Copies parameter values only 
        """
        newLearner = DecisionTreeLearner(self.criterion, self.maxDepth, self.minSplit, self.type, self.pruneType, self.alphaThreshold, self.folds)
        return newLearner 
        
    def getMetricMethod(self): 
        if self.type == "reg": 
            return Evaluator.rootMeanSqError
        else:
            return Evaluator.binaryError      
            
    def getAlphaThreshold(self): 
        return self.alphaThreshold
    
    def setAlphaThreshold(self, alphaThreshold): 
        Parameter.checkFloat(alphaThreshold, -float("inf"), float("inf"))
        self.alphaThreshold = alphaThreshold
        
    def setPruneCV(self, folds): 
        Parameter.checkInt(folds, 1, float("inf"))
        self.folds = folds
        
    def getPruneCV(self): 
        return self.folds
        
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
