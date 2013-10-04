
import unittest
import logging 
import sys 
from exp.recommendexp.Static2IdValDataset import Static2IdValDataset 
from apgl.util.PathDefaults import PathDefaults


class Static2IdValDatasetTest(unittest.TestCase):
    def setUp(self): 
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG) 
    
    def testGetTrainIteratorFunc(self):
        dataFilename = PathDefaults.getDataDir() + "reference/author_document_count" 
        dataset = Static2IdValDataset(dataFilename)

        trainIterator = dataset.getTrainIteratorFunc()()      
        testIterator = dataset.getTestIteratorFunc()()
        
        for trainX in trainIterator: 
            testX = testIterator.next() 
            
            print(trainX.shape, trainX.nnz, testX.nnz)
            self.assertEquals(trainX.shape, testX.shape)

if __name__ == '__main__':
    unittest.main()

