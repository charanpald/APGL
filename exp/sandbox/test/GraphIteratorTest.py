

import unittest
import numpy
import scipy.sparse 
from exp.sandbox.GraphIterators import DatedPurchasesGraphListIterator


class  DatedPurchasesGraphListIteratorTestCase(unittest.TestCase):
    def setUp(self):
        numpy.random.rand(21)
        numpy.set_printoptions(suppress=True, linewidth=200, precision=3)

    def testGraphCorrectness(self):
        purchases = [ [1, 3, 0, 0], # no relation 
                      [1, 4, 0, 0],
                      [3, 5, 0, 0],
                      [3, 1, 0, 0],
                      
                      [2, 2, 1, 0], # new node without new relation
            
                      [2, 1, 2, 0], # update one node and one new relation and one new final node
            
                      [2, 0, 3, 0], # update one node without relational effect
                      
                      [3, 2, 4, 0], # update one node and update relation
                      
                      [4, 1, 5, 0], # new node with several new relation and one new final node
            
                      [0, 5, 6, 0], # new node with one new relation / one new final node
            
                      [0, 2, 7, 0], # update one node and one new relation (without final) + one updated relation  

                      [5, 7, 8, 0]  # new node without new relation
                    ]
        
        it = DatedPurchasesGraphListIterator(purchases)

        # iteration 0 : steps 0-2
        W = next(it)
        self.assertTrue(scipy.sparse.issparse(W))
        W = W.todense()
        Wtest = scipy.zeros((2,2), dtype='int16')
        Wtest[1,0] = Wtest[0,1] = 1
        self.assertTrue((W==Wtest).all(), "W =\n" + str(W) + str(Wtest))
        
        # iteration 1 : steps 3-4
        W = next(it)
        self.assertTrue(scipy.sparse.issparse(W))
        W = W.todense()
        Wtest = scipy.zeros((2,2), dtype='int16')
        Wtest[1,0] = Wtest[0,1] = 2
        self.assertTrue((W==Wtest).all(), "W = " + str(W))
        
        # iteration 2 : step 5
        W = next(it)
        self.assertTrue(scipy.sparse.issparse(W))
        W = W.todense()
        Wtest = scipy.zeros((3,3), dtype='int16')
        Wtest[1,0] = Wtest[0,1] = 2
        Wtest[2,0] = Wtest[0,2] = Wtest[2,1] = Wtest[1,2] = 1
        self.assertTrue((W==Wtest).all(), "W = " + str(W))
                
        # iteration 3 : step 6
        W = next(it)
        self.assertTrue(scipy.sparse.issparse(W))
        W = W.todense()
        Wtest = scipy.zeros((4,4), dtype='int16')
        Wtest[1,0] = Wtest[0,1] = 2
        Wtest[2,0] = Wtest[0,2] = Wtest[2,1] = Wtest[1,2] = 1
        Wtest[3,1] = Wtest[1,3] = 1
        self.assertTrue((W==Wtest).all(), "W = " + str(W))
                
        # iteration 4 : step 7
        W = next(it)
        self.assertTrue(scipy.sparse.issparse(W))
        W = W.todense()
        Wtest = scipy.zeros((4,4), dtype='int16')
        Wtest[1,0] = Wtest[0,1] = 2
        Wtest[2,0] = Wtest[0,2] = Wtest[2,1] = Wtest[1,2] = 1
        Wtest[3,1] = Wtest[1,3] = 2
        Wtest[3,0] = Wtest[0,3] = 1
        self.assertTrue((W==Wtest).all(), "W = " + str(W))
        
        # iteration 5 : step 8-
        self.assertRaises(StopIteration, next, it)
        
 
if __name__ == '__main__':
    unittest.main()

