
import unittest
import logging
from apgl.predictors.BinomialClassifier import BinomialClassifier
from apgl.data.ExamplesGenerator import ExamplesGenerator 


class BinomialClassifierTest(unittest.TestCase):
    def setUp(self):
        examplesGenerator = ExamplesGenerator()
        self.X, self.y = examplesGenerator.generateBinaryExamples(1000)

    def testLearnModel(self):
        p = 0.1 
        binomialClassifier = BinomialClassifier(p)
        binomialClassifier.learnModel(self.X, self.y)

    def testClassify(self):
        p = 0.1
        binomialClassifier = BinomialClassifier(p)
        y = binomialClassifier.classify(self.X)

        positives = float(sum(y==1))/self.X.shape[0]
        
        logging.debug(positives)

        self.assertAlmostEqual(p, positives, 1)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
