import numpy
import unittest
import pickle 
import logging 
from datetime import datetime 
from exp.recommendexp.NetflixDataset import NetflixDataset 


class  GenerateToyDataTest(unittest.TestCase):
    def setUp(self): 
        pass 

    #@unittest.skip("")
    def testGetTrainIteratorFunc(self):
        dataset = NetflixDataset()
        #iterator = dataset.getTrainIteratorFunc()
        iterator = dataset.getTrainIteratorFunc()
      
    #@unittest.skip("")
    def testGetTestIteratorFunc(self):
        dataset = NetflixDataset()

        trainIterator = dataset.getTrainIteratorFunc()        
        testIterator = dataset.getTestIteratorFunc()
        
        for trainX in trainIterator: 
            testX = testIterator.next() 
            
            print(trainX.shape, trainX.nnz, testX.nnz)
            self.assertEquals(trainX.shape, testX.shape)
        
    #@unittest.skip("")    
    def testData(self): 
        dataset = NetflixDataset()
        custIdDict = pickle.load(open(dataset.custDictFileName))             
        dataArr = numpy.load(dataset.ratingFileName)
        movieInds, custInds, ratings, dates = dataArr["arr_0"], dataArr["arr_1"], dataArr["arr_2"], dataArr["arr_3"]
        
        #Check a few entries of the read data 
        self.assertEquals(movieInds[546], 0)
        self.assertEquals(movieInds[547], 1)
        self.assertEquals(movieInds[691], 1)
        self.assertEquals(movieInds[692], 2)
        
        self.assertEquals(custIdDict[1488844], 0)
        self.assertEquals(custIdDict[893988], 5)
        self.assertEquals(custIdDict[1503895], 10)
        
        self.assertEquals(ratings[0], 3)
        self.assertEquals(ratings[9], 3)
        self.assertEquals(ratings[10], 4)
        self.assertEquals(ratings[547], 4)
        self.assertEquals(ratings[550], 5)
        
        self.assertEquals(dates[0], int((datetime(2005,9,6)-dataset.startDate).total_seconds()))
        self.assertEquals(dates[9], int((datetime(2005,5,11)-dataset.startDate).total_seconds()))
        self.assertEquals(dates[10], int((datetime(2005,5,19)-dataset.startDate).total_seconds()))
        self.assertEquals(dates[547], int((datetime(2005,9,5)-dataset.startDate).total_seconds()))
        
        isTrainRating = numpy.load(dataset.isTrainRatingsFileName)["arr_0"]
        logging.debug("Train/test indicator loaded")  
        
        self.assertEquals(isTrainRating[3], False)
        self.assertEquals(isTrainRating[47], False)
        self.assertEquals(isTrainRating[59], False)
        self.assertEquals(isTrainRating[numpy.logical_and(movieInds==9, custInds==custIdDict[1952305])], False)
        self.assertEquals(isTrainRating[numpy.logical_and(movieInds==9, custInds==custIdDict[1531863])], False) 
        self.assertEquals((1-isTrainRating[movieInds==9]).sum(), 2)
        
        
        
if __name__ == '__main__':
    unittest.main()





