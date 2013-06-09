
import numpy
import unittest
import pickle 
import logging 
import time 
import numpy.testing as nptst 
from datetime import datetime 
from exp.recommendexp.MendeleyGroupsDataset import MendeleyGroupsDataset 


class  MovieLensDatasetTest(unittest.TestCase):
    def setUp(self): 
        pass 

    @unittest.skip("")
    def testIteratorFunc(self):
        iterStartTimeStamp = time.mktime(datetime(2009,01,25).timetuple())
        dataset = MendeleyGroupsDataset(iterStartTimeStamp=iterStartTimeStamp)
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
    
    @unittest.skip("") 
    def testIteratorFunc2(self):
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
        dataset = MendeleyGroupsDataset()
        userIdDict = pickle.load(open(dataset.userDictFileName))
        groupIdDict = pickle.load(open(dataset.groupDictFileName))                 
        dataArr = numpy.load(dataset.ratingFileName)
        movieInds, custInds, dates = dataArr["arr_0"], dataArr["arr_1"], dataArr["arr_2"]
        
        #Check a few entries of the read data 
        self.assertEquals(userIdDict[1852], 0)
        self.assertEquals(userIdDict[442481], 1)    
        self.assertEquals(userIdDict[2028911], 2) 
        
        self.assertEquals(custInds[0], 0)
        self.assertEquals(custInds[1], 1)
        self.assertEquals(custInds[2], 2)
        self.assertEquals(custInds[98673], 1)
        
        self.assertEquals(groupIdDict[481831], 0)
        self.assertEquals(groupIdDict[481841], 1)
        self.assertEquals(groupIdDict[481851], 2)
        
        
        self.assertEquals(dates[0], int(time.mktime(datetime.strptime("2010-10-05", "%Y-%m-%d").timetuple())))
        self.assertEquals(dates[2], int(time.mktime(datetime.strptime("2011-12-26", "%Y-%m-%d").timetuple())))
        self.assertEquals(dates[9], int(time.mktime(datetime.strptime("2010-11-17", "%Y-%m-%d").timetuple())))
        self.assertEquals(dates[10], int(time.mktime(datetime.strptime("2011-03-03", "%Y-%m-%d").timetuple())))
        
        
if __name__ == '__main__':
    unittest.main()





