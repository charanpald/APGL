'''
Created on 22 Jul 2009

@author: charanpal
'''

from exp.sandbox.predictors.NaiveBayes import NaiveBayes
from apgl.util.Sampling import Sampling
from apgl.util.Evaluator import Evaluator
import unittest
import numpy
from numpy.random import rand

#TODO: Probably ought to have more test cases
#TODO: Test case where classifying feature not present in training data 

class NaiveBayesTest(unittest.TestCase):
    """Set up some useful test variables"""
    def setUp(self):
        numExamples = 20
        numFeatures = 5
        
        self.X = numpy.round_(rand(numExamples, numFeatures)*5)
        self.y = numpy.round_(rand(numExamples, 1))*2-1
        self.y = self.y.ravel()
    
        #Create a simpler matrix 
        self.X2 = numpy.array([[1, 1], [1, 1],[1, 1],[1, 1],[1, 1],[1, 1],[1, 1],[1, 1], [1, 2], [0, 1]])
        self.y2 = numpy.array([[1],[1],[1],[1],[1],[1],[1],[1],[1],[-1]])
        
        #Another example with more classes for the 2nd feature 
        self.X3 = numpy.array([[0, 0], [0, 1],[1, 0],[1, 1],[1, 2]])
        self.y3 = numpy.array([[1],[1],[-1],[-1],[-1]])
        
        #Test the empty examples list 
        self.X4 = numpy.array([[]])
        self.y4 = numpy.array([])

        self.numExamples = numExamples 
        
        
    def testLearnModel(self):
        nb = NaiveBayes()
        nb.learnModel(self.X2, self.y2)
        
        self.assertTrue((nb.getLabelSet() == numpy.array([-1, 1])).all())
        self.assertTrue((nb.getFeatureSet(0) == numpy.array([0, 1])).all())
        self.assertTrue((nb.getFeatureSet(1) == numpy.array([1, 2])).all())
        
        f0CondMatrix = numpy.array([[1,0],[0,1]])
        self.assertTrue((nb.getCondMatrix(0) == f0CondMatrix).all())
        
        f1CondMatrix = numpy.array([[float(1)/9, 0],[float(8)/9, 1]])
        self.assertTrue((nb.getCondMatrix(1) == f1CondMatrix).all())
        
        #Test case where we learn twice 
        self.assertRaises(ValueError, nb.learnModel, self.X2, self.y2)
        
        
        
    def testLearnModel2(self):
        nb = NaiveBayes()
        nb.learnModel(self.X3, self.y3)
        
        self.assertTrue((nb.getLabelSet() == numpy.array([-1, 1])).all())
        self.assertTrue((nb.getFeatureSet(0) == numpy.array([0, 1])).all())
        self.assertTrue((nb.getFeatureSet(1) == numpy.array([0, 1, 2])).all())
        
        f0CondMatrix = numpy.array([[0,1],[1,0]])
        self.assertTrue((nb.getCondMatrix(0) == f0CondMatrix).all())
        
        
        f1CondMatrix = numpy.array([[float(1)/2, float(1)/2, 1],[float(1)/2, float(1)/2, 0]])
        self.assertTrue((nb.getCondMatrix(1) == f1CondMatrix).any())
        
        
    def testLearnModelAndClassify(self):
        nb = NaiveBayes()
        nb.learnModel(self.X4, self.y4)
        #nb.classify(self.X4)
        
    def testClassify(self):
        nb = NaiveBayes()
        
        self.assertRaises(ValueError, nb.classify, self.X2)
        
        nb.learnModel(self.X2, self.y2)
        testY = nb.classify(self.X2)
        testPys = nb.getProbabilities()
        
        eon = float(8)/9
        
        y = numpy.array([1,1,1,1,1,1,1,1,1,-1])
        pys = numpy.array([eon,eon,eon,eon,eon,eon,eon,eon,1,float(1)/9])
        
        self.assertTrue((testY == y).all())
        self.assertTrue((testPys == pys).all())
        
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(NaiveBayesTest)
    unittest.TextTestRunner(verbosity=2).run(suite)
