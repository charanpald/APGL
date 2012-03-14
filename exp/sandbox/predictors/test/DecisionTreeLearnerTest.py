import numpy 
import unittest
import numpy.testing as nptst
from exp.sandbox.predictors.DecisionTreeLearner import DecisionTreeLearner
from apgl.data.ExamplesGenerator import ExamplesGenerator  

class DecisionTreeLearnerTest(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(21)
        self.numExamples = 20
        self.numFeatures = 5
        
        generator = ExamplesGenerator() 
        self.X, self.y = generator.generateBinaryExamples(self.numExamples, self.numFeatures)
        
        
    def testInit(self): 
        learner = DecisionTreeLearner() 
        
    def testMeanSqError(self): 
        inds1 = numpy.arange(self.numFeatures/2)
        inds2 = numpy.arange(self.numFeatures/2,self.numFeatures)
        
        y1 = self.y[inds1]
        y2 = self.y[inds2]
        
        learner = DecisionTreeLearner() 
        error = learner.meanSqError(y1, y2)
        
        leftError = y1.var()
        rightError = y2.var() 
        
        error2 = y1.shape[0]*leftError + y2.shape[0]*rightError
        
        self.assertEquals(error, error2)
        
        #MSE should be zero with no variation 
        y1.fill(1)
        y2.fill(2)
        
        error = learner.meanSqError(y1, y2)
        self.assertEquals(error, 0.0)
        
    
    def testFindBestSplit(self): 
        learner = DecisionTreeLearner(minSplit=1) 
        
        X = numpy.zeros((20, 10))
        y = numpy.ones(20)
        
        X[0:10, 2] = numpy.arange(10)
        X[10:, 2] = numpy.arange(10)+10 
        y[0:10] = -1 
        
        bestError, bestFeatureInd, bestThreshold, bestSplitInds = learner.findBestSplit(X, y)
        
        
        self.assertEquals(bestError, 0.0)
        self.assertEquals(bestFeatureInd, 2)
        self.assertEquals(bestThreshold, 10)
        
        self.assertTrue((bestSplitInds[0][0:10]).all())
        self.assertTrue((bestSplitInds[1][10:]).all())
        
        #Test case where all values are the same 
        X = numpy.zeros((20, 10))
         
        bestError, bestFeatureInd, bestThreshold, bestSplitInds = learner.findBestSplit(X, y)
        self.assertTrue((bestSplitInds[1]).all())
        
    def testLearnModel(self): 
        learner = DecisionTreeLearner(minSplit=1, maxDepth=1) 

        learner.learnModel(self.X, self.y)        
        
        #print(learner.tree)
        
    def testPredict(self): 
        learner = DecisionTreeLearner(minSplit=1, maxDepth=1) 
        learner.learnModel(self.X, self.y)    
        
        predY = learner.predict(self.X)
        
        tree = learner.tree 
        
        for vertexId in tree.getAllVertexIds(): 
            nptst.assert_array_equal(tree.getVertex(vertexId).getTrainInds(), tree.getVertex(vertexId).getTestInds())
        
if __name__ == "__main__":
    unittest.main()