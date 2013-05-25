import numpy
import unittest
import time 
import numpy.testing as nptst 
from exp.recommendexp.CenterMatrixIterator import CenterMatrixIterator 
from exp.util.SparseUtils import SparseUtils 


class  CenterMatrixIteratorTest(unittest.TestCase):
    def setUp(self): 
        numpy.set_printoptions(suppress=True, precision=3, linewidth=150)
        numpy.random.seed(21)
        shape = (20, 10)
        r = 5 
        k = 100         
        
        #Create an iterator 
        matrixList = [] 
        matrixList.append(SparseUtils.generateSparseLowRank(shape, r, k))
        matrixList.append(SparseUtils.generateSparseLowRank(shape, r, k))
        matrixList.append(SparseUtils.generateSparseLowRank(shape, r, k))
        
        self.matrixList = matrixList
        self.testMatrixList = []
        
        for X in matrixList: 
            self.testMatrixList.append(X.copy())
      
    def getIterator(self): 
        return iter(self.matrixList)
    
    def testNext(self): 
        iterator = CenterMatrixIterator(self.getIterator())
        
        for X in iterator: 
            Y = X.todense()
            nptst.assert_array_almost_equal(numpy.array(Y.sum(1)).flatten(), numpy.zeros(Y.shape[0]))
    
    #@unittest.skip("")        
    def testcenterMatrix(self): 
        #Try centering a test set 
        iterator = CenterMatrixIterator(self.getIterator())
        testIterator = iter(self.testMatrixList) 
        
        for X in iterator: 
            Y = X.todense()
            X2 = next(testIterator)
            Y2 = iterator.centerMatrix(X2).todense()
            nptst.assert_array_almost_equal(Y, Y2)
            
    def testUnCenter(self): 
        iterator = CenterMatrixIterator(self.getIterator())
        testIterator = iter(self.testMatrixList) 
        
        for X in iterator:
            Y = iterator.uncenter(X).todense()
            Y2 = next(testIterator).todense()
            
            for i in range(Y.shape[0]): 
                if numpy.nonzero(Y[i, :])[0].shape[0] > 1: 
                    nptst.assert_array_almost_equal(Y[i, :], Y2[i, :])
        
        
if __name__ == '__main__':
    unittest.main()

