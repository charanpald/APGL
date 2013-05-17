import numpy
import unittest
import pickle 
import logging 
import time 
from datetime import datetime 
from exp.recommendexp.NetflixDataset import NetflixDataset 


class  NetflixDatasetTest(unittest.TestCase):
    def setUp(self): 
        pass 

    #@unittest.skip("")
    def testIteratorFunc(self):
        iterStartTimeStamp = time.mktime(datetime(2005,12,31).timetuple())
        dataset = NetflixDataset(iterStartTimeStamp=iterStartTimeStamp)
        #iterator = dataset.getTrainIteratorFunc()
        trainIterator = dataset.getTrainIteratorFunc()
        testIterator = dataset.getTestIteratorFunc()
        
        trainX = trainIterator.next() 
        testX = testIterator.next()
        self.assertEquals(trainX.shape, testX.shape)
        self.assertEquals(trainX.nnz + testX.nnz, dataset.numRatings)
        
        try: 
            trainIterator.next()
            self.fail()
        except StopIteration: 
            pass 
      
    #@unittest.skip("")
    def testIteratorFunc2(self):
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
        
        self.assertEquals(dates[0], int(time.mktime(datetime(2005,9,6).timetuple())))
        self.assertEquals(dates[9], int(time.mktime(datetime(2005,5,11).timetuple())))
        self.assertEquals(dates[10], int(time.mktime(datetime(2005,5,19).timetuple())))
        self.assertEquals(dates[547], int(time.mktime(datetime(2005,9,5).timetuple())))
        
        isTrainRating = numpy.load(dataset.isTrainRatingsFileName)["arr_0"]
        logging.debug("Train/test indicator loaded")  
        """
        self.assertEquals(isTrainRating[3], False)
        self.assertEquals(isTrainRating[47], False)
        self.assertEquals(isTrainRating[59], False)
        self.assertEquals(isTrainRating[numpy.logical_and(movieInds==9, custInds==custIdDict[1952305])], False)
        self.assertEquals(isTrainRating[numpy.logical_and(movieInds==9, custInds==custIdDict[1531863])], False) 
        self.assertEquals((1-isTrainRating[movieInds==9]).sum(), 2)
        """
        
        
        
if __name__ == '__main__':
    unittest.main()





