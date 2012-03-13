
from apgl.util import *
import numpy
"""
A class which encapsulates common tests for classes than inherit from AbtractMatrixGraph.
"""

class AbstractVertexListTest():
    def initialise(self):
        numpy.set_printoptions(suppress = True)
        numpy.random.seed(21)


    def testSetVertex(self):
        ind = 3;
        value = numpy.array([0.1, 0.5, 0.1])
        self.vList.setVertex(ind, value)

        self.assertEquals((self.vList.getVertex(ind) == value).all(), 1)

    def testClearVertex(self):
        ind = 3;
        value = numpy.array([0.1, 0.5, 0.1])
        self.vList.setVertex(ind, value)

        self.assertEquals((self.vList.getVertex(ind) == value).all(), 1)

        self.vList.clearVertex(3)
        self.assertVertexEquals(self.vList.getVertex(ind), self.emptyVertex)

    def testGetVertices(self):
        vList = self.vList
        vList.setVertex(0, numpy.array([1, 2, 3]))
        vList.setVertex(1, numpy.array([4, 5, 6]))
        vList.setVertex(2, numpy.array([7, 8, 9]))
        vList.setVertex(3, numpy.array([0, 11, 12]))

        self.assertTrue((vList.getVertices([0, 3]) == numpy.array([[1, 2, 3], [0, 11, 12]])).all())
        self.assertTrue((vList.getVertices([3, 0]) == numpy.array([[0, 11, 12], [1, 2, 3]])).all())
        self.assertTrue((vList.getVertices([0, 2]) == numpy.array([[1, 2, 3], [7, 8, 9]])).all())
        self.assertTrue((vList.getVertices([1]) == numpy.array([[4,5,6]])).all())

        #Test returning all vertices
        for i in range(4):
            self.assertVertexEquals(vList.getVertices()[i], vList.V[i])
        for i in range(4, self.numVertices):
            self.assertVertexEquals(vList.getVertices()[i], vList.getVertex(i))
        self.assertVertexEquals(vList.getVertices()[4], self.emptyVertex)

    def testGetCopy(self):
        vList = self.vList
        vList.setVertex(0, numpy.array([1, 2, 3]))
        vList.setVertex(1, numpy.array([4, 5, 6]))
        vList.setVertex(2, numpy.array([7, 8, 9]))

        vList2 = vList.copy()
        for i in range(self.numVertices):
            self.assertVertexEquals(vList2[i], vList[i])

        vList.setVertex(0, numpy.array([10, 20, 30]))
        self.assertVertexEquals(vList2.getVertex(0), numpy.array([1, 2, 3]))

    def testGetItem(self):
        numVertices = 5
        vList = self.vList
        vList.setVertex(0, numpy.array([1, 4, 5]))
        vList.setVertex(1, numpy.array([21, 12.5, 0.5]))

        self.assertVertexEquals(vList[0], numpy.array([1, 4, 5]))
        self.assertVertexEquals(vList[1], numpy.array([21, 12.5, 0.5]))


    def testSetItem(self):
        numVertices = 5
        vList = self.vList
        vList.setVertex(0, numpy.array([1, 4, 5]))
        vList.setVertex(1, numpy.array([21, 12.5, 0.5]))

        self.assertVertexEquals(vList[0], numpy.array([1, 4, 5]))
        vList[0] = numpy.array([2, 3.1, 4.2])
        self.assertVertexEquals(vList[0], numpy.array([2, 3.1, 4.2]))

    def testSubList(self):
        vList = self.vList
        vList.setVertex(0, numpy.array([1, 2, 3]))
        vList.setVertex(1, numpy.array([4, 5, 6]))
        vList.setVertex(2, numpy.array([7, 8, 9]))

        vList2 = vList.subList([0, 2])

        self.assertEquals(vList2.getNumVertices(), 2)
        self.assertVertexEquals(vList.getVertices([0, 2]), vList2.getVertices([0, 1]))

    def assertVertexEquals(self, v1, v2):
        if type(v1) == numpy.ndarray:
            self.assertTrue((v1 == v2).all())
        else:
            self.assertEquals(v1, v2)

