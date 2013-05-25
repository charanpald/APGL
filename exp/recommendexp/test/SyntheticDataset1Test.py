
import numpy
import unittest
import pickle 
import logging 
import time 
from datetime import datetime 
from exp.recommendexp.SyntheticDataset1 import SyntheticDataset1 


class SyntheticDataset1Test(unittest.TestCase):
    def setUp(self): 
        pass
    
    def testTrainValues(self): 
        dataset = SyntheticDataset1()
        iterator = dataset.getTrainIteratorFunc()()
        
        X = next(iterator)
        print(X)
    

if __name__ == '__main__':
    unittest.main()





