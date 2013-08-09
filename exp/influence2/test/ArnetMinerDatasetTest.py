import numpy 
import unittest
import logging
import sys 
from exp.influence2.ArnetMinerDataset import ArnetMinerDataset 

class  ArnetMinerDatasetTest(unittest.TestCase):
    def setUp(self): 
        numpy.random.seed(22) 
        numpy.set_printoptions(suppress=True, precision=3)
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        
    def testVectoriseDocuments(self): 
        field = "Boosting"
        dataset = ArnetMinerDataset(field)
        
        #Check document is correct as well as authors 
        dataset.vectoriseDocuments()
        
    def testFindSimilarDocuments(self): 
        field = "Boosting"
        dataset = ArnetMinerDataset(field)
        
        #Check document is correct as well as authors 
        dataset.vectoriseDocuments()
        dataset.findSimilarDocuments()
        
    def testFindCoauthors(self): 
        field = "Boosting"
        dataset = ArnetMinerDataset(field)
        
        #Check document is correct as well as authors 
        dataset.vectoriseDocuments()
        dataset.findSimilarDocuments()
        dataset.coauthorsGraph()
        
        
if __name__ == '__main__':
    unittest.main()
