
import unittest
import numpy
import numpy.testing as nptst 
import sys 
import logging
import scipy.sparse 
from exp.util.IdIndexer import IdIndexer

class IdIndexerTest(unittest.TestCase):
    def setUp(self):
        numpy.set_printoptions(suppress=True, precision=3, linewidth=150)
        numpy.random.seed(21)   
        
        self.indexer = IdIndexer()
        
        self.indexer.append("john")
        self.indexer.append("james")
        self.indexer.append("mark")
        self.indexer.append("james")
        
    def testAppend(self): 
        indexer = IdIndexer()
        
        indexer.append("john")
        indexer.append("james")
        indexer.append("mark")
        indexer.append("james")
        
        nptst.assert_array_equal(indexer.getArray(), numpy.array([0, 1, 2, 1]))
        
    def testTranslate(self): 
        self.assertEquals(self.indexer.translate(["mark"]), [2]) 
        self.assertEquals(self.indexer.translate(["john"]), [0])
        
        self.assertEquals(self.indexer.translate(["john", "james"]), [0, 1])
        
    def testReverseTranslate(self): 
        self.assertEquals(self.indexer.reverseTranslate(0), "john")
        self.assertEquals(self.indexer.reverseTranslate(1), "james")
        self.assertEquals(self.indexer.reverseTranslate(2), "mark")
        
        self.assertEquals(self.indexer.reverseTranslate([2,1,0]), ["mark", "james", "john"])
        
      
if __name__ == '__main__':
    unittest.main()