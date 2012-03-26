import numpy 
import unittest
import numpy.testing as nptst
from exp.sandbox.predictors.DecisionTreeLearner import DecisionTreeLearner
from apgl.data.ExamplesGenerator import ExamplesGenerator  
from sklearn.tree import DecisionTreeRegressor 

class DecisionTreeLearnerTest(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(21)
        numpy.seterr("raise")
        self.numExamples = 20
        self.numFeatures = 5
        
        generator = ExamplesGenerator() 
        self.X, self.y = generator.generateBinaryExamples(self.numExamples, self.numFeatures)
        
        
    def testInit(self): 
        learner = DecisionTreeLearner() 
         
    def testLearnModel(self): 
        #First check the integrety of the trees 
        generator = ExamplesGenerator()         
        
        for i in range(5):        
            numExamples = numpy.random.randint(1, 200)
            numFeatures = numpy.random.randint(1, 20)
            minSplit = numpy.random.randint(1, 50)
            maxDepth = numpy.random.randint(0, 10)
            
            X, y = generator.generateBinaryExamples(numExamples, numFeatures)
        
            learner = DecisionTreeLearner(minSplit=minSplit, maxDepth=maxDepth) 
            learner.learnModel(X, y)        
            tree = learner.getTree() 
            
            for vertexId in tree.getAllVertexIds(): 
                vertex = tree.getVertex(vertexId)
                if vertex.getFeatureInd() != None: 
                    meanValue = y[vertex.getTrainInds()].mean()
                    self.assertEquals(meanValue, vertex.getValue())
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
        learner = DecisionTreeLearner(minSplit=10, maxDepth=0) 
        learner.learnModel(self.X, self.y)        
        tree = learner.getTree()
        
        self.assertEquals(tree.depth(), 0)
        
        #Try minSplit > numExamples 
        learner = DecisionTreeLearner(minSplit=self.numExamples+1, maxDepth=0) 
        learner.learnModel(self.X, self.y)        
        tree = learner.getTree()
        
        self.assertEquals(tree.getNumVertices(), 1)
        
        
    
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
                
            learner = DecisionTreeLearner(minSplit=minSplit, maxDepth=maxDepth) 
            learner.learnModel(X, y)    
            
            predY = learner.predict(X)
            
            tree = learner.tree 
            
            for vertexId in tree.getAllVertexIds(): 
                nptst.assert_array_equal(tree.getVertex(vertexId).getTrainInds(), tree.getVertex(vertexId).getTestInds())
                
            #Compare against sklearn tree  
            regressor = DecisionTreeRegressor(min_split=minSplit, max_depth=maxDepth, min_density=0.0)
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

     
if __name__ == "__main__":
    unittest.main()