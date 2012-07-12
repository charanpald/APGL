import numpy 
import itertools 
import multiprocessing 
import logging 
from apgl.predictors.AbstractPredictor import AbstractPredictor
from apgl.graph.DictTree import DictTree
from exp.sandbox.predictors.TreeCriterion import findBestSplit
from exp.sandbox.predictors.TreeCriterionPy import findBestSplit2
from exp.sandbox.predictors.DecisionNode import DecisionNode
from apgl.util.Sampling import Sampling
from apgl.util.Parameter import Parameter
from apgl.util.Evaluator import Evaluator

def computePenaltyTree(args):
    """
    Used in conjunction with the parallel model selection.
    """
    (X, y, idx, learner, Cv) = args
    penalty = 0
    treeSize = 0
    folds = len(idx)

    for idxtr, idxts in idx:
        trainX, trainY = X[idxtr, :], y[idxtr]
        learner.learnModel(trainX, trainY)
        predY = learner.predict(X)
        predTrainY = learner.predict(trainX)
        penalty += learner.getMetricMethod()(predY, y) - learner.getMetricMethod()(predTrainY, trainY)
        treeSize += learner.getUnprunedTreeSize()
            
    penalty *= Cv/folds
    treeSize /= float(folds)
    
    if treeSize < learner.getGamma(): 
        penalty = float("inf")
    
    return penalty
    
def computePenalisedErrorTree(args):
    """
    Used in conjunction with the parallel model selection. It returns the
    binary error on the whole training set and the penalty
    """
    (X, y, idx, learner, Cv) = args
    penalty = computePenaltyTree(args)
    learner.learnModel(X, y)
    predY = learner.predict(X)
    return learner.getMetricMethod()(predY, y), penalty    
    
class DecisionTreeLearner(AbstractPredictor): 
    def __init__(self, criterion="mse", maxDepth=10, minSplit=30, type="reg", pruneType="none", gamma=1000, folds=5, processes=None):
        """
        Need a minSplit for the internal nodes and one for leaves. 
        
        :param gamma: A value between 0 (no pruning) and 1 (full pruning) which decides how much pruning to do. 
        """
        super(DecisionTreeLearner, self).__init__()
        self.maxDepth = maxDepth
        self.minSplit = minSplit
        self.criterion = criterion
        self.type = type
        self.pruneType = pruneType 
        self.setGamma(gamma)
        self.folds = 5
        self.processes = processes
        self.alphas = numpy.array([])
    
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
        self.unprunedTreeSize = self.tree.size
        
        if self.pruneType == "REP": 
            #Note: This should be a seperate validation set 
            self.repPrune(X, y)
        elif self.pruneType == "REP-CV":
            self.cvPrune(X, y)
        elif self.pruneType == "CART": 
            self.cartPrune(X, y)
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
        self.minAlpha = float("inf")
        self.maxAlpha = -float("inf")        
        
        for vertexId in self.tree.getAllVertexIds(): 
            currentNode = self.tree.getVertex(vertexId)
            subtreeLeaves = self.tree.leaves(vertexId)

            testErrorSum = 0 
            for leaf in subtreeLeaves: 
                testErrorSum += self.tree.getVertex(leaf).getTestError()
            
            #Alpha is normalised difference in error 
            if currentNode.getTestInds().shape[0] != 0: 
                currentNode.alpha = (testErrorSum - currentNode.getTestError())/float(currentNode.getTestInds().shape[0])       
                
                if currentNode.alpha < self.minAlpha:
                    self.minAlpha = currentNode.alpha 
                
                if currentNode.alpha > self.maxAlpha: 
                    self.maxAlpha = currentNode.alpha
                    
    def computeCARTAlphas(self, X):
        """
        Solve for the CART complexity based pruning. 
        """
        self.minAlpha = float("inf")
        self.maxAlpha = -float("inf")      
        alphas = [] 
        
        for vertexId in self.tree.getAllVertexIds(): 
            currentNode = self.tree.getVertex(vertexId)
            subtreeLeaves = self.tree.leaves(vertexId)

            testErrorSum = 0 
            for leaf in subtreeLeaves: 
                testErrorSum += self.tree.getVertex(leaf).getTestError()
            
            subtreeSize = len(self.tree.subtreeIds(vertexId))            
            
            #Alpha is reduction in error per leaf - larger alphas are better 
            if currentNode.getTestInds().shape[0] != 0 and subtreeSize != 1: 
                currentNode.alpha = (currentNode.getTestError() - testErrorSum)/float(X.shape[0]*(subtreeSize-1))
                #Flip alpha so that pruning works 
                currentNode.alpha = -currentNode.alpha
                
                alphas.append(currentNode.alpha)
                
                """
                if currentNode.alpha < self.minAlpha:
                    self.minAlpha = currentNode.alpha 
                
                if currentNode.alpha > self.maxAlpha: 
                    self.maxAlpha = currentNode.alpha   
                """
        alphas = numpy.array(alphas)
        self.alphas = numpy.unique(alphas)
        self.minAlpha = numpy.min(self.alphas)
        self.maxAlpha = numpy.max(self.alphas)

    def repPrune(self, validX, validY): 
        """
        Prune the decision tree using reduced error pruning. 
        """
        rootId = (0,)
        self.tree.getVertex(rootId).setTestInds(numpy.arange(validX.shape[0]))
        self.recursiveSetPrune(validX, validY, rootId)        
        self.computeAlphas()        
        self.prune()
                            
    def prune(self): 
        """
        We prune as early as possible and make sure the final tree has at most 
        gamma vertices. 
        """
        i = self.alphas.shape[0]-1 
        #print(self.alphas)
        
        while self.tree.getNumVertices() > self.gamma and i >= 0: 
            #print(self.alphas[i], self.tree.getNumVertices())
            alphaThreshold = self.alphas[i] 
            toPrune = []
            
            for vertexId in self.tree.getAllVertexIds(): 
                if self.tree.getVertex(vertexId).alpha >= alphaThreshold: 
                    toPrune.append(vertexId)

            for vertexId in toPrune: 
                if self.tree.vertexExists(vertexId):
                    self.tree.pruneVertex(vertexId)                    
                    
            i -= 1

                    
    def cartPrune(self, trainX, trainY): 
        """
        Prune the tree according to the CART algorithm. Here, the chosen 
        tree is selected by thresholding alpha. In CART itself the best 
        tree is selected by using an independent pruning set. 
        """
        rootId = (0,)
        self.tree.getVertex(rootId).setTestInds(numpy.arange(trainX.shape[0]))
        self.recursiveSetPrune(trainX, trainY, rootId)        
        self.computeCARTAlphas(trainX)    
        self.prune()
                
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
        self.prune()
        
    def copy(self): 
        """
        Copies parameter values only 
        """
        newLearner = DecisionTreeLearner(self.criterion, self.maxDepth, self.minSplit, self.type, self.pruneType, self.gamma, self.folds)
        return newLearner 
        
    def getMetricMethod(self): 
        if self.type == "reg": 
            #return Evaluator.rootMeanSqError
            return Evaluator.meanAbsError
            #return Evaluator.meanSqError
        else:
            return Evaluator.binaryError      
            
    def getAlphaThreshold(self): 
        #return self.maxAlpha - (self.maxAlpha - self.minAlpha)*self.gamma
        #A more natural way of defining gamma 
        return self.alphas[numpy.round((1-self.gamma)*(self.alphas.shape[0]-1))]        
        
    def setGamma(self, gamma): 
        """
        Gamma is an upper bound on the number of nodes in the tree. 
        """
        Parameter.checkInt(gamma, 1, float("inf"))
        self.gamma = gamma
        
    def getGamma(self): 
        return self.gamma 
        
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
        
    def complexity(self): 
        return self.tree.size
        
    def getBestLearner(self, meanErrors, paramDict, X, y, idx=None): 
        """
        Given a grid of errors, paramDict and examples, labels, find the 
        best learner and train it. In this case we set gamma to the real 
        size of the tree as learnt using CV. If idx == None then we simply 
        use the gamma corresponding to the lowest error. 
        """
        if idx == None: 
            return super(DecisionTreeLearner, self).getBestLearner(meanErrors, paramDict, X, y, idx)
        
        bestInds = numpy.unravel_index(numpy.argmin(meanErrors), meanErrors.shape)
        currentInd = 0    
        learner = self.copy()         
    
        for key, val in paramDict.items():
            method = getattr(learner, key)
            method(val[bestInds[currentInd]])
            currentInd += 1 
         
        treeSizes = []
        for trainInds, testInds in idx: 
            validX = X[trainInds, :]
            validY = y[trainInds]
            learner.learnModel(validX, validY)
            
            treeSizes.append(learner.tree.getNumVertices())
        
        bestGamma = int(numpy.round(numpy.array(treeSizes).mean()))
        
        learner.setGamma(bestGamma)
        learner.learnModel(X, y)            
        return learner 
        
    def getUnprunedTreeSize(self): 
        """
        Return the size of the tree before pruning was performed. 
        """
        return self.unprunedTreeSize

    def parallelPen(self, X, y, idx, paramDict, Cvs):
        """
        Perform parallel penalisation using any learner. 
        Using the best set of parameters train using the whole dataset. In this 
        case if gamma > max(treeSize) the penalty is infinite. 

        :param X: The examples as rows
        :type X: :class:`numpy.ndarray`

        :param y: The binary -1/+1 labels 
        :type y: :class:`numpy.ndarray`

        :param idx: A list of train/test splits

        :param paramDict: A dictionary index by the method name and with value as an array of values
        :type X: :class:`dict`

        """
        return super(DecisionTreeLearner, self).parallelPen(X, y, idx, paramDict, Cvs, computePenalisedErrorTree)
        