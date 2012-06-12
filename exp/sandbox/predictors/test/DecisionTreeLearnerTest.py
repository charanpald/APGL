import numpy 
import unittest
import numpy.testing as nptst
from exp.sandbox.predictors.DecisionTreeLearner import DecisionTreeLearner
from apgl.data.ExamplesGenerator import ExamplesGenerator
from apgl.data.Standardiser import Standardiser    
from sklearn.tree import DecisionTreeRegressor 
import sklearn.datasets as data 
from apgl.util.Evaluator import Evaluator

class DecisionTreeLearnerTest(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(21)
        numpy.seterr("raise")
        self.numExamples = 20
        self.numFeatures = 5
        
        generator = ExamplesGenerator() 
        self.X, self.y = generator.generateBinaryExamples(self.numExamples, self.numFeatures)
        self.y = numpy.array(self.y, numpy.float)
        
        
    def testInit(self): 
        learner = DecisionTreeLearner() 
         
    def testLearnModel(self): 
        #First check the integrety of the trees 
        generator = ExamplesGenerator()         
        
        for i in range(5):        
            numExamples = numpy.random.randint(1, 200)
            numFeatures = numpy.random.randint(1, 10)
            minSplit = numpy.random.randint(1, 50)
            maxDepth = numpy.random.randint(1, 10)
            
            X, y = generator.generateBinaryExamples(numExamples, numFeatures)
            y = numpy.array(y, numpy.float)
        
            learner = DecisionTreeLearner(minSplit=minSplit, maxDepth=maxDepth) 
            learner.learnModel(X, y)        
            tree = learner.getTree() 
            
            for vertexId in tree.getAllVertexIds(): 
                vertex = tree.getVertex(vertexId)
                if vertex.getFeatureInd() != None: 
                    meanValue = y[vertex.getTrainInds()].mean()
                    self.assertEquals(meanValue, vertex.getValue())
                    if tree.isNonLeaf(vertexId): 
                        self.assertTrue(0 <= vertex.getFeatureInd() < X.shape[1]) 
                        self.assertTrue(X[:, vertex.getFeatureInd()].min() <= vertex.getThreshold() <= X[:, vertex.getFeatureInd()].max())
                    self.assertTrue(vertex.getTrainInds().shape[0] >= 1)
            
            
            self.assertTrue(tree.depth() <= maxDepth)
            #Check that each split contains indices from parent 
            root = tree.getRootId()
            vertexStack = [root]
            
            while len(vertexStack) != 0: 
                vertexId = vertexStack.pop()
                neighbours = tree.children(vertexId)
                
                if len(neighbours) > 2: 
                    self.fail("Cannot have more than 2 children") 
                elif len(neighbours) > 0: 
                    inds1 = tree.getVertex(neighbours[0]).getTrainInds()
                    inds2 = tree.getVertex(neighbours[1]).getTrainInds()
                    
                    nptst.assert_array_equal(numpy.union1d(inds1, inds2), numpy.unique(tree.getVertex(vertexId).getTrainInds()))
                    
                    vertexStack.append(neighbours[0])
                    vertexStack.append(neighbours[1])
        
        #Try a tree of depth 0 
        #learner = DecisionTreeLearner(minSplit=10, maxDepth=0) 
        #learner.learnModel(self.X, self.y)        
        #tree = learner.getTree()
        
        #self.assertEquals(tree.depth(), 0)
        
        #Try minSplit > numExamples 
        #learner = DecisionTreeLearner(minSplit=self.numExamples+1, maxDepth=0) 
        #learner.learnModel(self.X, self.y)        
        #tree = learner.getTree()
        
        #self.assertEquals(tree.getNumVertices(), 1)
        
        #Try a simple tree of depth 1 
        learner = DecisionTreeLearner(minSplit=1, maxDepth=1) 
        learner.learnModel(self.X, self.y)     
        
        bestFeature = 0 
        bestError = 10**6 
        bestThreshold = 0         
        
        for i in range(numFeatures): 
            vals = numpy.unique(self.X[:, i])
            
            for j in range(vals.shape[0]-1):             
                threshold = (vals[j+1]+vals[j])/2
                leftInds = self.X[:, i] <= threshold
                rightInds = self.X[:, i] > threshold
                
                valLeft = numpy.mean(self.y[leftInds])
                valRight = numpy.mean(self.y[rightInds])
                
                error = ((self.y[leftInds] - valLeft)**2).sum() + ((self.y[rightInds] - valRight)**2).sum()
                
                if error < bestError: 
                    bestError = error 
                    bestFeature = i 
                    bestThreshold = threshold 
        
        self.assertAlmostEquals(bestThreshold, learner.tree.getRoot().getThreshold())
        self.assertAlmostEquals(bestError, learner.tree.getRoot().getError(), 5)
        self.assertEquals(bestFeature, learner.tree.getRoot().getFeatureInd())
        
        #Now we will test pruning works 
        learner = DecisionTreeLearner(minSplit=1, maxDepth=10) 
        learner.learnModel(X, y)
        numVertices1 = learner.getTree().getNumVertices()       
        
        learner = DecisionTreeLearner(minSplit=1, maxDepth=10, pruneType="REP-CV") 
        learner.learnModel(X, y) 
        numVertices2 = learner.getTree().getNumVertices()   
        
        self.assertTrue(numVertices1 >= numVertices2)
        
    @staticmethod
    def printTree(tree):
        """
        Some code to print the sklearn tree. 
        """
        
        children = tree.children
        
        depth = 0
        nodeIdStack = [(0, depth)] 
         
        
        while len(nodeIdStack) != 0:
            vertexId, depth = nodeIdStack.pop()
            
            if vertexId != tree.LEAF: 
                outputStr = "\t"*depth +str(vertexId) + ": Size: " + str(tree.n_samples[vertexId]) + ", "
                outputStr += "featureInd: " + str(tree.feature[vertexId]) + ", "
                outputStr += "threshold: " + str(tree.threshold[vertexId]) + ", "
                outputStr += "error: " + str(tree.best_error[vertexId]) + ", "
                outputStr += "value: " + str(tree.value[vertexId])
                print(outputStr)
            
                rightChildId = children[vertexId, 1]
                nodeIdStack.append((rightChildId, depth+1))
                
                leftChildId = children[vertexId, 0]
                nodeIdStack.append((leftChildId, depth+1))
        
    def testPredict(self): 
        
        generator = ExamplesGenerator()         
        
        for i in range(10):        
            numExamples = numpy.random.randint(1, 200)
            numFeatures = numpy.random.randint(1, 20)
            minSplit = numpy.random.randint(1, 50)
            maxDepth = numpy.random.randint(0, 10)
            
            X, y = generator.generateBinaryExamples(numExamples, numFeatures)   
            y = numpy.array(y, numpy.float)
                
            learner = DecisionTreeLearner(minSplit=minSplit, maxDepth=maxDepth) 
            learner.learnModel(X, y)    
            
            predY = learner.predict(X)
            
            tree = learner.tree            
            
            for vertexId in tree.getAllVertexIds(): 
                
                nptst.assert_array_equal(tree.getVertex(vertexId).getTrainInds(), tree.getVertex(vertexId).getTestInds())
                
            #Compare against sklearn tree  
            regressor = DecisionTreeRegressor(min_samples_split=minSplit, max_depth=maxDepth, min_density=0.0)
            regressor.fit(X, y)
            
            sktree = regressor.tree_
            
            #Note that the sklearn algorithm appears to combine nodes with same value 
            #self.assertEquals(sktree.node_count, tree.getNumVertices())
            self.assertEquals(sktree.feature[0], tree.getRoot().getFeatureInd())
            self.assertEquals(sktree.value[0], tree.getRoot().getValue())
            self.assertAlmostEquals(sktree.threshold[0], tree.getRoot().getThreshold(), 3)
            
            predY2 = regressor.predict(X)
            
            #Note that this is not always precise because if two thresholds give the same error we choose the largest 
            #and not sure how it is chosen in sklearn (or if the code is correct)
            self.assertTrue(abs(numpy.linalg.norm(predY-y)- numpy.linalg.norm(predY2-y))/numExamples < 0.05)  

    def testRecursiveSetPrune(self): 
        numExamples = 1000
        X, y = data.make_regression(numExamples)  
        
        y = Standardiser().normaliseArray(y)
        
        numTrain = numpy.round(numExamples * 0.66)     
        
        trainX = X[0:numTrain, :]
        trainY = y[0:numTrain]
        testX = X[numTrain:, :]
        testY = y[numTrain:]
        
        learner = DecisionTreeLearner()
        learner.learnModel(trainX, trainY)
        
        rootId = (0,)
        learner.tree.getVertex(rootId).setTestInds(numpy.arange(testX.shape[0]))
        learner.recursiveSetPrune(testX, testY, rootId)
        
        for vertexId in learner.tree.getAllVertexIds(): 
            tempY = testY[learner.tree.getVertex(vertexId).getTestInds()]
            predY = numpy.ones(tempY.shape[0])*learner.tree.getVertex(vertexId).getValue()
            error = numpy.sum((tempY-predY)**2)
            self.assertAlmostEquals(error, learner.tree.getVertex(vertexId).getTestError())
            
        #Check leaf indices form all indices 
        inds = numpy.array([])        
        
        for vertexId in learner.tree.leaves(): 
            inds = numpy.union1d(inds, learner.tree.getVertex(vertexId).getTestInds())
            
        nptst.assert_array_equal(inds, numpy.arange(testY.shape[0]))
        
        
    
    def testPrune(self):
        numExamples = 500
        X, y = data.make_regression(numExamples)  
        
        y = Standardiser().standardiseArray(y)
        
        numTrain = numpy.round(numExamples * 0.66)     
        
        trainX = X[0:numTrain, :]
        trainY = y[0:numTrain]
        testX = X[numTrain:, :]
        testY = y[numTrain:]
        
        #In this case we set alpha = maxAlpha which is 0.0 
        learner = DecisionTreeLearner(gamma=0.0)
        learner.learnModel(trainX, trainY)
        
        vertexIds = learner.tree.getAllVertexIds()         
        
        learner.repPrune(trainX, trainY)
        self.assertEquals(learner.maxAlpha, 0.0)
        
        vertexIds2 = learner.tree.getAllVertexIds() 
        
        #No pruning if we test using training set 
        self.assertEquals(vertexIds, vertexIds2)
        
        #Now prune using test set 
        learner.setGamma(1.0)
        learner.repPrune(testX, testY)
        toPrune = []
        
        for  vertexId in learner.tree.getAllVertexIds(): 
            if learner.tree.getVertex(vertexId).alpha > 0: 
                toPrune.append(vertexId)       
        
        learner.setGamma(0.0)
        learner.repPrune(testX, testY)
        
        self.assertTrue((0, 0, 1, 0) not in learner.tree.getAllVertexIds())
        
        #Now try max pruning 
        learner.setGamma(1.0)
        learner.repPrune(testX, testY)
        self.assertEquals(learner.tree.getNumVertices(), 1)
    
    def testRecursivePrune(self): 
        learner = DecisionTreeLearner(minSplit=5)
        learner.learnModel(self.X, self.y)
        
        unprunedTree = learner.getTree().copy()
             
        learner.minAlpha = float("inf")
        learner.maxAlpha = -float("inf")                  
             
        #Now randomly assign alpha values 
        for vertexId in learner.tree.getAllVertexIds(): 
            learner.tree.getVertex(vertexId).alpha = numpy.random.randn()
            
            if learner.tree.getVertex(vertexId).alpha < learner.minAlpha:
                    learner.minAlpha = learner.tree.getVertex(vertexId).alpha 
                
            if learner.tree.getVertex(vertexId).alpha > learner.maxAlpha: 
                learner.maxAlpha = learner.tree.getVertex(vertexId).alpha

        learner.recursivePrune((0,))
        
        for vertexId in learner.tree.getAllVertexIds(): 
            if learner.tree.getVertex(vertexId).alpha > learner.getAlphaThreshold(): 
                self.assertTrue(learner.tree.isLeaf(vertexId))
                
        self.assertTrue(learner.tree.isSubtree(unprunedTree))
        
    
    def testCvPrune(self): 
        numExamples = 500
        X, y = data.make_regression(numExamples)  
        
        y = Standardiser().standardiseArray(y)
        
        numTrain = numpy.round(numExamples * 0.33)     
        numValid = numpy.round(numExamples * 0.33) 
        
        trainX = X[0:numTrain, :]
        trainY = y[0:numTrain]
        validX = X[numTrain:numTrain+numValid, :]
        validY = y[numTrain:numTrain+numValid]
        testX = X[numTrain+numValid:, :]
        testY = y[numTrain+numValid:]
        
        learner = DecisionTreeLearner()
        learner.learnModel(trainX, trainY)
        error1 = Evaluator.rootMeanSqError(learner.predict(testX), testY)
        
        #print(learner.getTree())
        unprunedTree = learner.tree.copy() 
        learner.setGamma(0.0)
        learner.cvPrune(trainX, trainY)
        
        self.assertEquals(unprunedTree.getNumVertices(), learner.tree.getNumVertices())
        learner.setGamma(0.5)
        learner.cvPrune(trainX, trainY)
        
        #Test if pruned tree is subtree of current: 
        for vertexId in learner.tree.getAllVertexIds(): 
            self.assertTrue(vertexId in unprunedTree.getAllVertexIds())
            
        #The error should be better after pruning 
        learner.learnModel(trainX, trainY)
        #learner.cvPrune(validX, validY, 0.0, 5)
        learner.repPrune(validX, validY)
      
        error2 = Evaluator.rootMeanSqError(learner.predict(testX), testY)
        
        self.assertTrue(error1 >= error2)

        

        

     
if __name__ == "__main__":
    unittest.main()