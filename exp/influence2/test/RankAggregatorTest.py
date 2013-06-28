import numpy 
import unittest
import logging
from exp.influence2.RankAggregator import RankAggregator 
import scipy.stats.mstats 
import numpy.testing as nptst 


class  RankAggregatorTest(unittest.TestCase):
    def setUp(self): 
        numpy.random.seed(22) 
    
    
    def testSpearmanFootrule(self): 
        list1 = [5, 4, 3, 2, 1, 0]
        list2 = [5, 4, 3, 2, 1, 0]
        
        dist = RankAggregator.spearmanFootrule(list1, list2)
        
        self.assertEquals(dist, 0)
        
        list2 = [5, 4, 3, 2, 0, 1]
        dist = RankAggregator.spearmanFootrule(list1, list2)
        
        self.assertEquals(dist, 1.0/9)
        
        list2 = [0, 1, 2, 3, 4, 5]
        dist = RankAggregator.spearmanFootrule(list1, list2)
        self.assertEquals(dist, 1.0)
        
    def testBorda(self): 
        list1 = [5, 4, 3, 2, 1, 0]
        list2 = [5, 4, 3, 2, 1, 0]  
        
        outList = RankAggregator.borda(list1, list2)
        
        nptst.assert_array_equal(outList, numpy.array([5,4,3,2,1,0]))
        
        list2 = [4, 3, 2, 5, 1, 0]
        outList = RankAggregator.borda(list1, list2)
        nptst.assert_array_equal(outList, numpy.array([4,5,3,2,1,0]))
        

if __name__ == '__main__':
    unittest.main()

