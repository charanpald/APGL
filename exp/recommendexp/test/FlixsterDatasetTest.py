import numpy
import unittest
import pickle 
import logging 
import time 
import numpy.testing as nptst 
from datetime import datetime 
from exp.recommendexp.FlixsterDataset import FlixsterDataset 


class FlixsterDatasetTest(unittest.TestCase):
    def setUp(self): 
        pass 
    
    def testIteratorFunc2(self):
        dataset = FlixsterDataset()

        trainIterator = dataset.getTrainIteratorFunc()        
        testIterator = dataset.getTestIteratorFunc()
        
        for trainX in trainIterator: 
            testX = testIterator.next() 
            
            print(trainX.shape, trainX.nnz, testX.nnz)
            self.assertEquals(trainX.shape, testX.shape)

if __name__ == '__main__':
    unittest.main()

