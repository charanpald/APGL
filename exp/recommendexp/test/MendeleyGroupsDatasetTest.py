import sys 
import numpy
import unittest
import pickle 
import logging 
import time 
import numpy.testing as nptst 
import scipy.sparse 
from datetime import datetime 
from exp.recommendexp.MendeleyGroupsDataset import MendeleyGroupsDataset 


class  MendeleyGroupsDatasetTest(unittest.TestCase):
    def setUp(self): 
        numpy.random.seed(21)
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        numpy.set_printoptions(suppress=True, linewidth=60)
        numpy.seterr("raise", under="ignore") 

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
      
    #@unittest.skip("")
    def testIteratorFunc2(self):
        dataset = MendeleyGroupsDataset()

        trainIterator = dataset.getTrainIteratorFunc()        
        testIterator = dataset.getTestIteratorFunc()
        
        for trainX in trainIterator: 
            testX = testIterator.next() 
            
            print(trainX.shape, trainX.nnz, testX.nnz)
            self.assertEquals(trainX.shape, testX.shape)
    
    #@unittest.skip("") 
    def testIteratorFunc3(self):
        #Check to see if repeated calls generate new matrices 
        dataset = MendeleyGroupsDataset()
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
        user1Inds, user2Inds, dates = dataArr["arr_0"], dataArr["arr_1"], dataArr["arr_2"]
        

        #Check a few entries of the read data 
        self.assertEquals(dates[0], int(time.mktime(datetime.strptime("2011-03-17", "%Y-%m-%d").timetuple())))
        self.assertEquals(dates[1], int(time.mktime(datetime.strptime("2011-12-26", "%Y-%m-%d").timetuple())))
        

    def testBipartiteToUni(self):
        userInds = numpy.array([0, 1, 2, 0, 4])
        itemInds = numpy.array([0, 0, 0, 1, 1])
        dates = numpy.array([1, 2, 3, 4, 5])
        
        dataset = MendeleyGroupsDataset()
        user1Inds, user2Inds, newDates = dataset.bipartiteToUni(userInds, itemInds, dates)
        
        X = scipy.sparse.csc_matrix((newDates, (user1Inds, user2Inds)))        
        
        nptst.assert_array_equal(X[0, :].nonzero()[1], numpy.array([1, 2, 4]))
        nptst.assert_array_equal(X[1, :].nonzero()[1], numpy.array([0, 2]))
        nptst.assert_array_equal(X[2, :].nonzero()[1], numpy.array([0, 1]))
        nptst.assert_array_equal(X[3, :].nonzero()[1], numpy.array([]))
        nptst.assert_array_equal(X[4, :].nonzero()[1], numpy.array([0]))
        
if __name__ == '__main__':
    unittest.main()





