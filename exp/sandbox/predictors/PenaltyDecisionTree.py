import numpy 
from apgl.graph.DictTree import DictTree 
from apgl.util.Util import Util 
from apgl.util.Evaluator import Evaluator 
from apgl.util.Parameter import Parameter 
from exp.sandbox.predictors.DecisionNode import DecisionNode
from exp.sandbox.predictors.TreeCriterionPy import findBestSplit2, findBestSplitRisk
from apgl.predictors.AbstractPredictor import AbstractPredictor

class PenaltyDecisionTree(AbstractPredictor): 
    def __init__(self, criterion="gain", maxDepth=10, minSplit=30, learnType="reg", pruning=True, gamma=0.01, sampleSize=10):
        """
        Learn a decision tree with penalty proportional to the root of the size 
        of the tree as in Nobel 2002. We use a stochastic approach in which we 
        learn a set of trees randomly and choose the best one. 

        :param criterion: The splitting criterion which is only informaiton gain currently 

        :param maxDepth: The maximum depth of the tree 
        :type maxDepth: `int`

        :param minSplit: The minimum size of a node for it to be split. 
        :type minSplit: `int`
        
        :param type: The type of learning to perform. Currently only regression 
        
        :param pruning: Whether to perform pruning or not. 
        :type pruning: `boolean`
        
        :param gamma: The weight on the penalty factor between 0 and 1
        :type gamma: `float`
        
        :param sampleSize: The number of trees to learn in the stochastic search. 
        :type sampleSize: `int`
        """
        super(PenaltyDecisionTree, self).__init__()
        self.maxDepth = maxDepth
        self.minSplit = minSplit
        self.criterion = criterion
        self.learnType = learnType
        self.setGamma(gamma)
        self.setSampleSize(sampleSize) 
        self.pruning = pruning 
        self.alphaThreshold = 0.0
                
    def setGamma(self, gamma): 
        Parameter.checkFloat(gamma, 0.0, 1.0)
        self.gamma = gamma   
        
    def setSampleSize(self, sampleSize):
        Parameter.checkInt(sampleSize, 1, float("inf"))
        self.sampleSize = sampleSize                

    def setAlphaThreshold(self, alphaThreshold): 
        Parameter.checkFloat(alphaThreshold, -float("inf"), float("inf"))
        self.alphaThreshold = alphaThreshold
   
    def getAlphaThreshold(self): 
        return self.alphaThreshold
    
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
                
    def learnModel(self, X, y):
        if numpy.unique(y).shape[0] != 2: 
            raise ValueError("Must provide binary labels")
        if y.dtype != numpy.int: 
            raise ValueError("Labels must be integers")
        
        self.shapeX = X.shape  
        argsortX = numpy.zeros(X.shape, numpy.int)
        for i in range(X.shape[1]): 
            argsortX[:, i] = numpy.argsort(X[:, i])
            argsortX[:, i] = numpy.argsort(argsortX[:, i])
        
            
        rootId = (0,)
        idStack = [rootId]
        self.tree = DictTree()
        rootNode = DecisionNode(numpy.arange(X.shape[0]), Util.mode(y))
        self.tree.setVertex(rootId, rootNode)
        bestError = float("inf")
        bestTree = self.tree 
        
        while len(idStack) != 0:
            #Prune the current node away and grow from that node 
            nodeId = idStack.pop()
            
            for i in range(self.sampleSize):   
                self.tree = bestTree.deepCopy()
                node = self.tree.getVertex(nodeId)
                self.tree.pruneVertex(nodeId)            
                self.growTree(X, y, argsortX, nodeId)
                self.prune(X, y)
                error = self.treeObjective(X, y)
            
                if error < bestError: 
                    bestError = error
                    bestTree = self.tree.deepCopy()
            
            children = bestTree.children(nodeId)            
            idStack.extend(children)               
            
        self.tree = bestTree 

    def growTree(self, X, y, argsortX, startId): 
        """
        Grow a tree using a stack. Give a sample of data and a node index, we 
        find the best split and add children to the tree accordingly. We perform 
        pre-pruning based on the penalty. 
        """
        eps = 10**-4 
        idStack = [startId]
        
        while len(idStack) != 0: 
            nodeId = idStack.pop()
            node = self.tree.getVertex(nodeId)
            accuracies, thresholds = findBestSplitRisk(self.minSplit, X, y, node.getTrainInds(), argsortX)
        
            #Choose best feature based on gains 
            accuracies += eps 
            bestFeatureInd = Util.randomChoice(accuracies)[0]
            bestThreshold = thresholds[bestFeatureInd]
        
            nodeInds = node.getTrainInds()    
            bestLeftInds = numpy.sort(nodeInds[numpy.arange(nodeInds.shape[0])[X[:, bestFeatureInd][nodeInds]<bestThreshold]]) 
            bestRightInds = numpy.sort(nodeInds[numpy.arange(nodeInds.shape[0])[X[:, bestFeatureInd][nodeInds]>=bestThreshold]])
            
            #The split may have 0 items in one set, so don't split 
            if bestLeftInds.sum() != 0 and bestRightInds.sum() != 0 and self.tree.depth() < self.maxDepth: 
                node.setError(1-accuracies[bestFeatureInd])
                node.setFeatureInd(bestFeatureInd)
                node.setThreshold(bestThreshold)            
                            
                leftChildId = self.getLeftChildId(nodeId)
                leftChild = DecisionNode(bestLeftInds, Util.mode(y[bestLeftInds]))
                self.tree.addChild(nodeId, leftChildId, leftChild)
                
                if leftChild.getTrainInds().shape[0] >= self.minSplit: 
                    idStack.append(leftChildId)
                
                rightChildId = self.getRightChildId(nodeId)
                rightChild = DecisionNode(bestRightInds, Util.mode(y[bestRightInds]))
                self.tree.addChild(nodeId, rightChildId, rightChild)
                
                if rightChild.getTrainInds().shape[0] >= self.minSplit: 
                    idStack.append(rightChildId)
        
    def predict(self, X, y=None): 
        """
        Make a prediction for the set of examples given in the matrix X.  If 
        one passes in a label vector y then we set the errors for each node. On 
        the other hand if y=None, no errors are set. 
        """ 
        rootId = (0,)
        predY = numpy.zeros(X.shape[0])
        self.tree.getVertex(rootId).setTestInds(numpy.arange(X.shape[0]))
        idStack = [rootId]

        while len(idStack) != 0:
            nodeId = idStack.pop()
            node = self.tree.getVertex(nodeId)
            testInds = node.getTestInds()
            if y!=None: 
                node.setTestError(self.vertexTestError(y[testInds], node.getValue()))
        
            if self.tree.isLeaf(nodeId): 
                predY[testInds] = node.getValue()
            else: 
                 
                for childId in [self.getLeftChildId(nodeId), self.getRightChildId(nodeId)]:
                    if self.tree.vertexExists(childId):
                        child = self.tree.getVertex(childId)
        
                        if childId[-1] == 0: 
                            childInds = X[testInds, node.getFeatureInd()] < node.getThreshold() 
                        else:
                            childInds = X[testInds, node.getFeatureInd()] >= node.getThreshold()
                        
                        child.setTestInds(testInds[childInds])   
                        idStack.append(childId)
                
        return predY

    def treeObjective(self, X, y): 
        """
        Return the empirical risk plus penalty for the tree. 
        """
        predY = self.predict(X)
        (n, d) = X.shape
        return (1-self.gamma)*numpy.sum(predY!=y)/float(n) + self.gamma*numpy.sqrt(self.tree.getNumVertices())

    def prune(self, X, y): 
        """
        Do some post pruning greedily. 
        """
        self.predict(X, y)  
        self.computeAlphas()
        
        #Do the pruning, recomputing alpha along the way 
        rootId = (0,)
        idStack = [rootId]

        while len(idStack) != 0:        
            nodeId = idStack.pop()
            node = self.tree.getVertex(nodeId)
    
            if node.alpha > self.alphaThreshold: 
                self.tree.pruneVertex(nodeId)
                self.computeAlphas()
            else: 
                for childId in [self.getLeftChildId(nodeId), self.getRightChildId(nodeId)]: 
                    if self.tree.vertexExists(childId):
                        idStack.append(childId)
        
    def vertexTestError(self, trueY, predY):
        """
        This is the error used for pruning. We compute it at each node. 
        """
        return numpy.sum(trueY != predY)
        
    def computeAlphas(self): 
        """
        The alpha value at each vertex is the improvement in the objective by 
        pruning at that vertex.  
        """
        n = self.shapeX[0]    
        
        for vertexId in self.tree.getAllVertexIds(): 
            currentNode = self.tree.getVertex(vertexId)            
            subtreeLeaves = self.tree.leaves(vertexId)
    
            subtreeError = 0 
            for leaf in subtreeLeaves: 
                subtreeError += self.tree.getVertex(leaf).getTestError()
        
            T = self.tree.getNumVertices()
            T2 = T - len(self.tree.subtreeIds(vertexId)) + 1 
            currentNode.alpha = (1-self.gamma)*(subtreeError - currentNode.getTestError())
            currentNode.alpha /= n
            currentNode.alpha += self.gamma * numpy.sqrt(T)
            currentNode.alpha -= self.gamma * numpy.sqrt(T2)

    def copy(self): 
        """
        Create a new tree with the same parameters. 
        """
        newLearner = PenaltyDecisionTree(criterion=self.criterion, maxDepth=self.maxDepth, minSplit=self.minSplit, learnType=self.learnType, pruning=self.pruning, gamma=self.gamma, sampleSize=self.sampleSize)
        return newLearner 
        
    def getMetricMethod(self):
        """ 
        Returns a way to measure the performance of the classifier.
        """
        return Evaluator.binaryError
