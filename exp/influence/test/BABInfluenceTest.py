
import unittest
import numpy
import logging
import sys
from apgl.influence.BABInfluence import BABInfluence 

class  BABInfluenceTestCase(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    def testMaxBudgetedInfluence(self):
        numVertices = 5
        P = numpy.zeros((numVertices, numVertices))
        u = numpy.array([3,4,6,2,1])
        P[0, :] = numpy.array([2, 8, 3, 4, 1])
        P[1, :] = numpy.array([1, 9, 6, 2, 3])
        P[2, :] = numpy.array([12, 8 ,9, 3, 1])
        P[3, :] = numpy.array([0, 2, 1, 6, 9])
        P[4, :] = numpy.array([1, 0, 0, 8, 2])

        L = 7.0

        influence = BABInfluence()
        inds = influence.maxBudgetedInfluence(P, u, L)

        logging.debug(inds)
        self.assertEquals(inds, [2, 4])

        #Jumble up the rows
        newRowInds = numpy.random.permutation(numVertices)
        P2 = P[newRowInds, :]
        u2 = u[newRowInds]
        inds = influence.maxBudgetedInfluence(P2, u2, L)

        self.assertTrue( ((numpy.sort(numpy.array(newRowInds)[inds])) == numpy.array([2, 4])).all() )

if __name__ == '__main__':
    unittest.main()

