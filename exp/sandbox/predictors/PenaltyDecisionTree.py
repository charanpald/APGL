import numpy 

from apgl.graph.DictTree import DictTree 
from apgl.util.Util import Util 
from exp.sandbox.predictors.DecisionTreeLearner import DecisionTreeLearner
from exp.sandbox.predictors.DecisionNode import DecisionNode
from exp.sandbox.predictors.TreeCriterionPy import findBestSplit2, findBestSplitRand

class PenaltyDecisionTree(DecisionTreeLearner): 
    def __init__(self, criterion="mse", maxDepth=10, minSplit=30, type="reg", pruning=True, gamma=0.1, sampleSize=10):
        """
        Need a minSplit for the internal nodes and one for leaves. A PenaltyDecisionTree
        is one created such that the penalty on the empirical risk is proportional 
        to the root of the size of the tree as in Nobel 2002. 
        
        :param gamma: The weight on the penalty factor.  
        """
        super(PenaltyDecisionTree, self).__init__()
        self.maxDepth = maxDepth
        self.minSplit = minSplit
        self.criterion = criterion
        self.type = type
        self.gamma = gamma 
        self.sampleSize = sampleSize 
        self.pruning = pruning 
                
    def learnModel(self, X, y):
        if numpy.unique(y).shape[0] != 2: 
            raise ValueError("Must provide binary labels")
        
        self.shapeX = X.shape
        rootId = (0, )         
        argsortX = numpy.zeros(X.shape, numpy.int)
        for i in range(X.shape[1]): 
            argsortX[:, i] = numpy.argsort(X[:, i])
            argsortX[:, i] = numpy.argsort(argsortX[:, i])
        
        bestError = float("inf")        
        
        for i in range(self.sampleSize): 
            self.tree = DictTree()
            rootNode = DecisionNode(numpy.arange(X.shape[0]), Util.mode(y))
            self.tree.setVertex(rootId, rootNode)
            self.recursiveSplit(X, y, argsortX, rootId)
            if self.pruning: 
                self.prune(X, y)
            error = self.treeObjective(X, y)
            
            if error < bestError: 
                bestError = error
                bestTree = self.tree.copy()
        
        self.tree = bestTree 
        
    def treeObjective(self, X, y): 
        """
        Return the empirical risk plus penalty for the tree. 
        """
        predY = self.predict(X)
        T = self.tree.getNumVertices()
        (n, d) = X.shape
        error = numpy.sum(predY!=y)/float(X.shape[0]) + self.gamma*numpy.sqrt(32*(T*d*numpy.log(n) + T*numpy.log(2) + 2*numpy.log(T))/n)
        return error 

    def prune(self, X, y): 
        """
        Do some post pruning greedily. 
        """
        rootId = (0,)
        self.tree.getVertex(rootId).setTestInds(numpy.arange(X.shape[0]))
        self.recursiveSetPrune(X, y, rootId)  
        self.computeAlphas()
        self.recursivePrune(rootId)
        
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
        d = self.shapeX[1]        
        
        for vertexId in self.tree.getAllVertexIds(): 
            currentNode = self.tree.getVertex(vertexId)            
            subtreeLeaves = self.tree.leaves(vertexId)
    
            subtreeError = 0 
            for leaf in subtreeLeaves: 
                subtreeError += self.tree.getVertex(leaf).getTestError()
        
            T = self.tree.getNumVertices()
            T2 = T - len(self.tree.subtreeIds(vertexId)) + 1 
            currentNode.alpha = (subtreeError - currentNode.getTestError())
            currentNode.alpha += self.gamma * numpy.sqrt(32*n*(T*d*numpy.log(n) + T*numpy.log(2) + 2*numpy.log(T)))
            currentNode.alpha -= self.gamma * numpy.sqrt(32*n*(T2*d*numpy.log(n) + T2*numpy.log(2) + 2*numpy.log(T2)))
            currentNode.alpha /= n

    def recursivePrune(self, nodeId): 
        """
        We prune as early as possible and recompute the alphas after each pruning.   
        """
        node = self.tree.getVertex(nodeId)

        if node.alpha > self.alphaThreshold: 
            self.tree.pruneVertex(nodeId)
            self.computeAlphas()
        else: 
            for childId in [self.getLeftChildId(nodeId), self.getRightChildId(nodeId)]: 
                if self.tree.vertexExists(childId):
                    self.recursivePrune(childId)

    def recursiveSplit(self, X, y, argsortX, nodeId): 
        """
        Give a sample of data and a node index, we find the best split and 
        add children to the tree accordingly. We perform a pre-pruning 
        based on the penalty. 
        """
        if len(nodeId)-1 >= self.maxDepth: 
            return 
        
        node = self.tree.getVertex(nodeId)
        gains, thresholds = findBestSplitRand(self.minSplit, X, y, node.getTrainInds(), argsortX)
        
        #Choose best feature based on gains 
        eps = 10**-4 
        gains += eps 
        bestFeatureInd = Util.randomChoice(gains)[0]
        bestThreshold = thresholds[bestFeatureInd]
    
        nodeInds = node.getTrainInds()    
        bestLeftInds = numpy.sort(nodeInds[numpy.arange(nodeInds.shape[0])[X[:, bestFeatureInd][nodeInds]<bestThreshold]]) 
        bestRightInds = numpy.sort(nodeInds[numpy.arange(nodeInds.shape[0])[X[:, bestFeatureInd][nodeInds]>=bestThreshold]])
    
        #The split may have 0 items in one set, so don't split 
        if bestLeftInds.sum() != 0 and bestRightInds.sum() != 0: 
            node.setError(gains[bestFeatureInd])
            node.setFeatureInd(bestFeatureInd)
            node.setThreshold(bestThreshold)
            
            leftChildId = self.getLeftChildId(nodeId)
            leftChild = DecisionNode(bestLeftInds, Util.mode(y[bestLeftInds]))
            self.tree.addChild(nodeId, leftChildId, leftChild)
            
            if leftChild.getTrainInds().shape[0] >= self.minSplit: 
                self.recursiveSplit(X, y, argsortX, leftChildId)
            
            rightChildId = self.getRightChildId(nodeId)
            rightChild = DecisionNode(bestRightInds, Util.mode(y[bestRightInds]))
            self.tree.addChild(nodeId, rightChildId, rightChild)
            
            if rightChild.getTrainInds().shape[0] >= self.minSplit: 
                self.recursiveSplit(X, y, argsortX, rightChildId)