import numpy 
import unittest
import numpy.testing as nptst
from exp.sandbox.predictors.PenaltyDecisionTree import PenaltyDecisionTree
from apgl.data.ExamplesGenerator import ExamplesGenerator
from apgl.data.Standardiser import Standardiser    
import sklearn.datasets as data 
from apgl.util.Evaluator import Evaluator

class PenaltyDecisionTreeLearnerTest(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(21)
        numpy.seterr("raise")
        self.numExamples = 20
        self.numFeatures = 5
        
        generator = ExamplesGenerator() 
        self.X, self.y = generator.generateBinaryExamples(self.numExamples, self.numFeatures)
    
    def testLearnModel(self): 
        generator = ExamplesGenerator()         
        
        numExamples = 200
        numFeatures = 10
        minSplit = 20
        maxDepth = 3
            
        X, y = generator.generateBinaryExamples(numExamples, numFeatures)
        y+=1        
        
        testX = X[100:, :]
        testY = y[100:]
        X = X[0:100, :]
        y = y[0:100]
         
        
        gamma = 0.1
        learner = PenaltyDecisionTree(minSplit=minSplit, maxDepth=maxDepth, gamma=gamma) 
        learner.learnModel(X, y)   
                  
        tree = learner.getTree() 
        
        #Work out penalty cost 
        predY = learner.predict(X)
        predTestY = learner.predict(testX)
        
        n = float(X.shape[0])
        d = X.shape[1]
        T = tree.getNumVertices()
        error = numpy.sum(predY!=y)/n
        testError = numpy.sum(predTestY!=testY)/float(testY.shape[0])
        error += gamma*numpy.sqrt(32*(T*d*numpy.log(n) + T*numpy.log(2) + 2*numpy.log(T))/n)
        
        print("error=" + str(error))
        print("testError=" + str(testError))            
        print(tree)        
        
if __name__ == '__main__':
    unittest.main()
