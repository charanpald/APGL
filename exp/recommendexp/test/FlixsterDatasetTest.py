import numpy
import unittest
import pickle 
import logging 
import time 
import numpy.testing as nptst 
from datetime import datetime 
from exp.recommendexp.FlixsterDataset import FlixsterDataset 
import matplotlib 
matplotlib.use("GTK3Agg")
import matplotlib.pyplot as plt 
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)


class FlixsterDatasetTest(unittest.TestCase):
    def setUp(self): 
        pass 
    
    @unittest.skip("")
    def testIteratorFunc2(self):
        dataset = FlixsterDataset()

        trainIterator = dataset.getTrainIteratorFunc()        
        testIterator = dataset.getTestIteratorFunc()
        
        for trainX in trainIterator: 
            testX = testIterator.next() 
            
            print(trainX.shape, trainX.nnz, testX.nnz)
            self.assertEquals(trainX.shape, testX.shape)

    @unittest.skip("")
    def testIteratorFunc3(self):
        #Check to see if repeated calls generate new matrices 
        iterStartTimeStamp = time.mktime(datetime(2005,1,1).timetuple())
        dataset = FlixsterDataset(iterStartTimeStamp=iterStartTimeStamp)
        trainIterator = dataset.getTrainIteratorFunc()   
        
        X = next(trainIterator)
        X.data += 1 
        
        trainIterator = dataset.getTrainIteratorFunc()  
        X2 = next(trainIterator)
        
        nptst.assert_array_almost_equal(X.data, X2.data+1)

    def testData(self): 
        dataset = FlixsterDataset()
        custIdDict = pickle.load(open(dataset.custDictFileName))
        movieIdDict = pickle.load(open(dataset.itemDictFileName))                 
        dataArr = numpy.load(dataset.ratingFileName)
        movieInds, custInds, ratings, dates = dataArr["arr_0"], dataArr["arr_1"], dataArr["arr_2"], dataArr["arr_3"]
        
        #Check a few entries of the read data 
        self.assertEquals(custInds[0], 0)
        self.assertEquals(custInds[1], 0)
        self.assertEquals(custInds[71], 0)
        self.assertEquals(custInds[72], 1)
        
        self.assertEquals(movieIdDict[81], 0)
        self.assertEquals(movieIdDict[926], 1)
        self.assertEquals(movieIdDict[5124], 10)
        
        self.assertEquals(ratings[0], 1.5)
        self.assertEquals(ratings[9], 0.5)
        self.assertEquals(ratings[10], 1)
        self.assertEquals(ratings[250], 0.5)
        self.assertEquals(ratings[550], 4.5)
        
        self.assertEquals(time.strftime("%D", time.localtime(dates[0])), "10/10/07")
        self.assertEquals(time.strftime("%D", time.localtime(dates[4])), "12/29/07")
        self.assertEquals(time.strftime("%D", time.localtime(dates[5])), "11/13/07")
        self.assertEquals(time.strftime("%D", time.localtime(dates[7])), "10/10/07")

    def testData2(self): 
        #Check numbers of rows/cols of matrix 
        
        dataset = FlixsterDataset()
        trainIterator = dataset.getTrainIteratorFunc()   
        X = trainIterator.next() 
        
        rowInds, colInds = X.nonzero() 
        print("Counting rows")
        rowCounts = numpy.bincount(rowInds)
        colCounts = numpy.bincount(colInds)
        print("Done counting rows")
        print((colCounts<5000).sum(), X.shape[1])
        
        #plt.hist(rowCounts, bins=1000, range=) 
        #plt.show() 


if __name__ == '__main__':
    unittest.main()

