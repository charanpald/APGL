import numpy 
import unittest
import numpy.testing as nptst
from exp.sandbox.predictors.PenaltyDecisionTree import PenaltyDecisionTree
from apgl.data.ExamplesGenerator import ExamplesGenerator
from apgl.util.Evaluator import Evaluator    
import sklearn.datasets as data 
from apgl.util.Util import Util
from apgl.graph.DictTree import DictTree 
from exp.sandbox.predictors.DecisionNode import DecisionNode

class PenaltyDecisionTreeLearnerTest(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(21)
        numpy.seterr("raise")
        self.numExamples = 200
        self.numFeatures = 10
        
        generator = ExamplesGenerator() 
        self.X, self.y = generator.generateBinaryExamples(self.numExamples, self.numFeatures)
        self.y += 1 
    
    def testLearnModel(self): 
        minSplit = 20
        maxDepth = 3
        gamma = 0.00
            
        X, y = self.X, self.y
                
        testX = X[100:, :]
        testY = y[100:]
        X = X[0:100, :]
        y = y[0:100]
         
        learner = PenaltyDecisionTree(minSplit=minSplit, maxDepth=maxDepth, gamma=gamma) 
        learner.learnModel(X, y)                  
        tree = learner.getTree() 
        
        #Work out penalty cost 
        predY = learner.predict(X)
        predTestY = learner.predict(testX)
        
        n = float(X.shape[0])
        d = X.shape[1]
        T = tree.getNumVertices()
        error = (1-gamma)*numpy.sum(predY!=y)/n
        testError = numpy.sum(predTestY!=testY)/float(testY.shape[0])
        error += gamma*numpy.sqrt(T)
        
        self.assertEquals(error, learner.treeObjective(X, y)) 
                
        #Check if the values in the tree nodes are correct 
        for vertexId in tree.getAllVertexIds(): 
            vertex = tree.getVertex(vertexId)
            
            self.assertTrue(vertex.getValue()==2.0 or vertex.getValue()==0.0)
            if tree.isNonLeaf(vertexId): 
                self.assertTrue(0 <= vertex.getFeatureInd() <= X.shape[1])
                self.assertTrue(0 <= vertex.getError() <= 1)

    @unittest.skip("")    
    def testLearnModel2(self): 
        #We want to make sure the learnt tree with gamma = 0 maximise the 
        #empirical risk 
        minSplit = 20
        maxDepth = 3
        gamma = 0.00
        learner = PenaltyDecisionTree(minSplit=minSplit, maxDepth=maxDepth, gamma=gamma, pruning=False) 
        
        #Vary sampleSize
        numpy.random.seed(21)
        learner.setSampleSize(5)           
        learner.learnModel(self.X, self.y)        
        predY = learner.predict(self.X)
        error1 = Evaluator.binaryError(self.y, predY)

        numpy.random.seed(21)
        learner.setSampleSize(10)        
        learner.learnModel(self.X, self.y)
        predY = learner.predict(self.X)
        error2 = Evaluator.binaryError(self.y, predY)

        numpy.random.seed(21)                
        learner.setSampleSize(30)       
        learner.learnModel(self.X, self.y)
        predY = learner.predict(self.X)
        error3 = Evaluator.binaryError(self.y, predY)
        
        self.assertTrue(error1 >= error2)
        self.assertTrue(error2 >= error3)
        
        #Now vary max depth 
        numpy.random.seed(21)
        learner.setSampleSize(10) 
        learner.minSplit = 1
        learner.maxDepth = 3 
        learner.learnModel(self.X, self.y)
        predY = learner.predict(self.X)
        error1 = Evaluator.binaryError(self.y, predY)
        
        numpy.random.seed(21)
        learner.maxDepth = 5 
        learner.learnModel(self.X, self.y)
        predY = learner.predict(self.X)
        error2 = Evaluator.binaryError(self.y, predY)
        
        numpy.random.seed(21)
        learner.maxDepth = 10 
        learner.learnModel(self.X, self.y)
        predY = learner.predict(self.X)
        error2 = Evaluator.binaryError(self.y, predY)        
        
        self.assertTrue(error1 >= error2)
        #print(error1, error2, error3)
        #self.assertTrue(error2 >= error3)


    def testComputeAlphas(self): 
        minSplit = 20
        maxDepth = 3
        gamma = 0.1
            
        X, y = self.X, self.y
                
        testX = X[100:, :]
        testY = y[100:]
        X = X[0:100, :]
        y = y[0:100]
         
        learner = PenaltyDecisionTree(minSplit=minSplit, maxDepth=maxDepth, gamma=gamma, pruning=False) 
        learner.learnModel(X, y)                  
        tree = learner.getTree()    
        
        rootId = (0,)
        learner.tree.getVertex(rootId).setTestInds(numpy.arange(X.shape[0]))
        learner.predict(X, y)  
        learner.computeAlphas()
        
        #See if the alpha values of the nodes are correct 
        for vertexId in tree.getAllVertexIds(): 
            subtreeLeaves = tree.leaves(vertexId)
            
            subtreeError = 0 
            for subtreeLeaf in subtreeLeaves: 
                subtreeError += (1-gamma)*tree.getVertex(subtreeLeaf).getTestError()
            
            n = float(X.shape[0])
            d = X.shape[1]
            T = tree.getNumVertices() 
            subtreeError /= n 
            subtreeError += gamma * numpy.sqrt(T)
            
            T2 = T - len(tree.subtreeIds(vertexId)) + 1 
            vertexError = (1-gamma)*tree.getVertex(vertexId).getTestError()/n
            vertexError +=  gamma * numpy.sqrt(T2)
            
            self.assertAlmostEquals((subtreeError - vertexError), tree.getVertex(vertexId).alpha)
            
            if tree.isLeaf(vertexId): 
                self.assertEquals(tree.getVertex(vertexId).alpha, 0.0)
                
        #Let's check the alpha of the root node via another method 
        rootId = (0,)
        
        T = 1 
        (n, d) = X.shape
        n = float(n)
        vertexError = (1-gamma)*numpy.sum(y != Util.mode(y))/n
        pen = gamma*numpy.sqrt(T)
        vertexError += pen 
        
        T = tree.getNumVertices() 
        treeError = (1-gamma)*numpy.sum(y != learner.predict(X))/n         
        pen = gamma*numpy.sqrt(T)
        treeError += pen 
        
        alpha = treeError - vertexError 
        self.assertAlmostEqual(alpha, tree.getVertex(rootId).alpha)
        
    def testPrune(self): 
        startId = (0, )
        minSplit = 20
        maxDepth = 5
        gamma = 0.05
        learner = PenaltyDecisionTree(minSplit=minSplit, maxDepth=maxDepth, gamma=gamma, pruning=False) 
        
        trainX = self.X[100:, :]
        trainY = self.y[100:]
        testX = self.X[0:100, :]
        testY = self.y[0:100]    
        
        argsortX = numpy.zeros(trainX.shape, numpy.int)
        for i in range(trainX.shape[1]): 
            argsortX[:, i] = numpy.argsort(trainX[:, i])
            argsortX[:, i] = numpy.argsort(argsortX[:, i])
        
        learner.tree = DictTree()
        rootNode = DecisionNode(numpy.arange(trainX.shape[0]), Util.mode(trainY))
        learner.tree.setVertex(startId, rootNode)        
        learner.growTree(trainX, trainY, argsortX, startId)    
        learner.shapeX = trainX.shape 
        learner.predict(trainX, trainY)
        learner.computeAlphas()
        
        obj1 = learner.treeObjective(trainX, trainY)        
        size1 = learner.tree.getNumVertices()
        
        #Now we'll prune 
        learner.prune(trainX, trainY)
        
        obj2 = learner.treeObjective(trainX, trainY)
        size2 = learner.tree.getNumVertices()
        
        self.assertTrue(obj1 >= obj2)    
        self.assertTrue(size1 >= size2)        
        
        #Check there are no nodes with alpha>alphaThreshold 
        for vertexId in learner.tree.getAllVertexIds(): 
            self.assertTrue(learner.tree.getVertex(vertexId).alpha <= learner.alphaThreshold)
    
    def testGrowTree(self):
        startId = (0, )
        minSplit = 20
        maxDepth = 3
        gamma = 0.01
        learner = PenaltyDecisionTree(minSplit=minSplit, maxDepth=maxDepth, gamma=gamma, pruning=False) 
        
        trainX = self.X[100:, :]
        trainY = self.y[100:]
        testX = self.X[0:100, :]
        testY = self.y[0:100]    
        
        argsortX = numpy.zeros(trainX.shape, numpy.int)
        for i in range(trainX.shape[1]): 
            argsortX[:, i] = numpy.argsort(trainX[:, i])
            argsortX[:, i] = numpy.argsort(argsortX[:, i])
        
        learner.tree = DictTree()
        rootNode = DecisionNode(numpy.arange(trainX.shape[0]), Util.mode(trainY))
        learner.tree.setVertex(startId, rootNode)        
        
        #Note that this matches with the case where we create a new tree each time 
        numpy.random.seed(21)
        bestError = float("inf")        
        
        for i in range(20): 
            learner.tree.pruneVertex(startId)
            learner.growTree(trainX, trainY, argsortX, startId)
            
            predTestY = learner.predict(testX)
            error = Evaluator.binaryError(predTestY, testY)
            #print(Evaluator.binaryError(predTestY, testY), learner.tree.getNumVertices())
            
            if error < bestError: 
                bestError = error 
                bestTree = learner.tree.copy() 
            
            self.assertTrue(learner.tree.depth() <= maxDepth)
            
            for vertexId in learner.tree.nonLeaves(): 
                self.assertTrue(learner.tree.getVertex(vertexId).getTrainInds().shape[0] >= minSplit)
        
        bestError1 = bestError               
        learner.tree = bestTree    
        
        #Now we test growing a tree from a non-root vertex 
        numpy.random.seed(21)
        for i in range(20): 
            learner.tree.pruneVertex((0, 1)) 
            learner.growTree(trainX, trainY, argsortX, (0, 1))
            
            self.assertTrue(learner.tree.getVertex((0,)) == bestTree.getVertex((0,)))
            self.assertTrue(learner.tree.getVertex((0,0)) == bestTree.getVertex((0,0)))
            
            
            predTestY = learner.predict(testX)
            error = Evaluator.binaryError(predTestY, testY)
            
            if error < bestError: 
                bestError = error 
                bestTree = learner.tree.copy() 
            #print(Evaluator.binaryError(predTestY, testY), learner.tree.getNumVertices())
        print(bestError1, bestError)
        self.assertTrue(bestError1 >= bestError )
        
if __name__ == '__main__':
    unittest.main()
