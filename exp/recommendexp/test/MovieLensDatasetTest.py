import numpy
import unittest
import pickle 
import logging 
import time 
import numpy.testing as nptst 
from datetime import datetime 
from exp.recommendexp.MovieLensDataset import MovieLensDataset 


class  MovieLensDatasetTest(unittest.TestCase):
    def setUp(self): 
        pass 

    @unittest.skip("")
    def testIteratorFunc(self):
        iterStartTimeStamp = time.mktime(datetime(2009,01,25).timetuple())
        dataset = MovieLensDataset(iterStartTimeStamp=iterStartTimeStamp)
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
      
    @unittest.skip("")
    def testIteratorFunc2(self):
        dataset = MovieLensDataset()

        trainIterator = dataset.getTrainIteratorFunc()        
        testIterator = dataset.getTestIteratorFunc()
        
        for trainX in trainIterator: 
            testX = testIterator.next() 
            
            print(trainX.shape, trainX.nnz, testX.nnz)
            self.assertEquals(trainX.shape, testX.shape)

    def testIteratorFunc3(self):
        #Check to see if repeated calls generate new matrices 
        iterStartTimeStamp = time.mktime(datetime(2005,1,1).timetuple())
        dataset = MovieLensDataset(iterStartTimeStamp=iterStartTimeStamp)
        trainIterator = dataset.getTrainIteratorFunc()   
        
        X = next(trainIterator)
        X.data += 1 
        
        trainIterator = dataset.getTrainIteratorFunc()  
        X2 = next(trainIterator)
        
        nptst.assert_array_almost_equal(X.data, X2.data+1)
        
        
        
    #@unittest.skip("")    
    def testData(self): 
        dataset = MovieLensDataset()
        custIdDict = pickle.load(open(dataset.custDictFileName))
        movieIdDict = pickle.load(open(dataset.movieDictFileName))                 
        dataArr = numpy.load(dataset.ratingFileName)
        movieInds, custInds, ratings, dates = dataArr["arr_0"], dataArr["arr_1"], dataArr["arr_2"], dataArr["arr_3"]
        
        #Check a few entries of the read data 
        self.assertEquals(custInds[21], 0)
        self.assertEquals(custInds[22], 1)
        self.assertEquals(custInds[41], 1)
        self.assertEquals(custInds[42], 2)
        
        self.assertEquals(movieIdDict[122], 0)
        self.assertEquals(movieIdDict[329], 5)
        self.assertEquals(movieIdDict[370], 10)
        
        self.assertEquals(ratings[0], 5.0)
        self.assertEquals(ratings[9], 5)
        self.assertEquals(ratings[10], 5)
        self.assertEquals(ratings[547], 2.5)
        self.assertEquals(ratings[550], 5)
        
        self.assertEquals(dates[0], 838985046)
        self.assertEquals(dates[9], 838983707)
        self.assertEquals(dates[10], 838984596)
        self.assertEquals(dates[547], 1116547122)
        
        
if __name__ == '__main__':
    unittest.main()





