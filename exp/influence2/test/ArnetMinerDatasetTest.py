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
        
    def testFindAuthorsInField(self): 
        field = "Boosting"
        dataset = ArnetMinerDataset(field)
        
        dataset.findAuthorsInField()
        
        
if __name__ == '__main__':
    unittest.main()
