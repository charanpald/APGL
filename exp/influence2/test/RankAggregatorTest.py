import numpy 
import unittest
import logging
from exp.influence2.RankAggregator import RankAggregator 
import scipy.stats.mstats 
import numpy.testing as nptst 


class  RankAggregatorTest(unittest.TestCase):
    def setUp(self): 
        numpy.random.seed(22) 
        numpy.set_printoptions(suppress=True, precision=4)
    
    
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
        

    def testMC2(self): 
        list1 = [5, 4, 3, 2, 1, 0]
        list2 = [5, 4, 3, 2, 1, 0]  
        
        lists = [list1, list2]
        itemList = [0, 1, 2, 3, 4, 5]
        outList, scores = RankAggregator.MC2(lists, itemList)
        
        self.assertEquals(outList, [5, 4, 3, 2, 1, 0])
        
        list1 = [2, 1, 3, 4, 5, 0]
        list2 = [2, 1, 3, 4, 5, 0]
        lists = [list1, list2]

        outList, scores = RankAggregator.MC2(lists, itemList)
        self.assertEquals(outList, [2, 1, 3, 4, 5, 0])
        
        #Now test weighting 
        alpha = numpy.array([1, 0])
        list1 = [2, 1, 3, 4, 5, 0]
        list2 = [3, 2, 5, 0, 4, 1]
        lists = [list1, list2]
        
        outList, scores = RankAggregator.MC2(lists, itemList, alpha)
        self.assertEquals(outList, list1)
        
        alpha = numpy.array([0, 1.0])
        outList, scores = RankAggregator.MC2(lists, itemList, alpha)
        self.assertEquals(outList, list2)

    def testSupervisedMC22(self): 
        list1 = [2, 1, 3, 4, 5, 0]
        list2 = [3, 2, 5, 0, 4, 1]
        list3 = [3, 4, 2, 5, 1, 0]
        list4 = [5, 0, 3, 4, 1, 2]
        lists = [list1, list2, list3, list4]
        
        itemList = [0, 1, 2, 3, 4, 5]
        topQList = [5, 4, 3, 2, 1, 0]

        
        outputList, scores = RankAggregator.supervisedMC22(lists, itemList, topQList)
        
        
    def testGreedyMC2(self): 
        list1 = [2, 1, 3, 4, 5, 0]
        list2 = [3, 2, 5, 0, 4, 1]
        list3 = [3, 4, 2, 5, 1, 0]
        list4 = [3, 4, 5, 0, 1, 2]
        lists = [list1, list2, list3, list4]
        
        itemList = [0, 1, 2, 3, 4, 5]
        topQList = [3, 4, 5] 
        
        n = 3
        outputInds = RankAggregator.greedyMC2(lists, itemList, topQList, n)
        
        self.assertEquals(outputInds, [3])        
        
        list1 = [2, 1, 3, 4, 5, 0]
        list2 = [3, 2, 5, 4, 0, 1]
        list3 = [4, 3, 2, 5, 1, 0]
        list4 = [3, 4, 0, 5, 1, 2]
        lists = [list1, list2, list3, list4]

        outputInds = RankAggregator.greedyMC2(lists, itemList, topQList, n)
        
        newLists = []
        for ind in outputInds: 
            newLists.append(lists[ind])
            
        outputList, scores = RankAggregator.MC2(newLists, itemList)
        self.assertEquals(outputList[0:3], [3, 4, 5])
        

if __name__ == '__main__':
    unittest.main()

