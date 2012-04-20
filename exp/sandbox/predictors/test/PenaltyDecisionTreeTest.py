import numpy 
import unittest
import numpy.testing as nptst
from exp.sandbox.predictors.PenaltyDecisionTree import PenaltyDecisionTree
from apgl.data.ExamplesGenerator import ExamplesGenerator
from apgl.data.Standardiser import Standardiser    
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
    
    def testLearnModel(self): 
        generator = ExamplesGenerator()         
        
        numExamples = numpy.random.randint(1, 200)
        numFeatures = numpy.random.randint(1, 10)
        minSplit = numpy.random.randint(1, 50)
        maxDepth = numpy.random.randint(0, 10)
            
        X, y = generator.generateBinaryExamples(numExamples, numFeatures)
        y+=1 
        
        learner = PenaltyDecisionTree(minSplit=minSplit, maxDepth=maxDepth) 
        
        for i in range(10):         
            learner.learnModel(X, y)        
            tree = learner.getTree() 
        
            print(tree)        
        
if __name__ == '__main__':
    unittest.main()
