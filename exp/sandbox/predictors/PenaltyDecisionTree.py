import numpy 

from apgl.graph.DictTree import DictTree 
from apgl.util.Util import Util 
from exp.sandbox.predictors.DecisionTreeLearner import DecisionTreeLearner
from exp.sandbox.predictors.DecisionNode import DecisionNode
from exp.sandbox.predictors.TreeCriterionPy import findBestSplit2, findBestSplitRand

class PenaltyDecisionTree(DecisionTreeLearner): 
    def __init__(self, criterion="mse", maxDepth=10, minSplit=30, type="reg"):
        """
        Need a minSplit for the internal nodes and one for leaves. 
        """
        super(PenaltyDecisionTree, self).__init__()
        self.maxDepth = maxDepth
        self.minSplit = minSplit
        self.criterion = criterion
        self.type = type
                
    def learnModel(self, X, y):
        nodeId = (0, )         
        self.tree = DictTree()
        rootNode = DecisionNode(numpy.arange(X.shape[0]), y.mean())
        self.tree.setVertex(nodeId, rootNode)

        argsortX = numpy.zeros(X.shape, numpy.int)
        for i in range(X.shape[1]): 
            argsortX[:, i] = numpy.argsort(X[:, i])
            argsortX[:, i] = numpy.argsort(argsortX[:, i])
        
        self.recursiveSplit(X, y, argsortX, nodeId)

        #Next add empirical risk plus penalty 

    def recursiveSplit(self, X, y, argsortX, nodeId): 
        """
        Give a sample of data and a node index, we find the best split and 
        add children to the tree accordingly. 
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
            leftChild = DecisionNode(bestLeftInds, y[bestLeftInds].mean())
            self.tree.addChild(nodeId, leftChildId, leftChild)
            
            if leftChild.getTrainInds().shape[0] >= self.minSplit: 
                self.recursiveSplit(X, y, argsortX, leftChildId)
            
            rightChildId = self.getRightChildId(nodeId)
            rightChild = DecisionNode(bestRightInds, y[bestRightInds].mean())
            self.tree.addChild(nodeId, rightChildId, rightChild)
            
            if rightChild.getTrainInds().shape[0] >= self.minSplit: 
                self.recursiveSplit(X, y, argsortX, rightChildId)