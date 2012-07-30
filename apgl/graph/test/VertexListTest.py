
from apgl.util.PathDefaults import PathDefaults
from apgl.graph.test.AbstractVertexListTest import AbstractVertexListTest
from apgl.graph.VertexList import VertexList
import unittest
import numpy
import logging
import numpy.testing as nptst 

class VertexListTest(unittest.TestCase, AbstractVertexListTest):
    def setUp(self):
        self.numVertices = 10 
        self.numFeatures = 3
        self.vList = VertexList(self.numVertices, self.numFeatures)
        self.emptyVertex = numpy.zeros(self.numFeatures)
        self.initialise()
        
    def testConstructor(self):
        self.assertEquals(self.vList.getNumFeatures(), self.numFeatures)
        self.assertEquals(self.vList.getNumVertices(), self.numVertices)
        
    def testSaveLoad(self):
        try:
            vList = VertexList(self.numVertices, self.numFeatures)
            vList.setVertex(0, numpy.array([1, 2, 3]))
            vList.setVertex(1, numpy.array([4, 5, 6]))
            vList.setVertex(2, numpy.array([7, 8, 9]))

            tempDir = PathDefaults.getTempDir()
            fileName = tempDir + "vList"

            vList.save(fileName)
            vList2 = VertexList.load(fileName)

            self.assertTrue((vList.getVertices() == vList2.getVertices()).all())
        except IOError as e:
            logging.warn(e)
            pass 

    def testGetItem2(self): 
        V = numpy.random.rand(self.numVertices, self.numFeatures)
        self.vList.setVertices(V)
        
        for i in range(self.numVertices): 
            nptst.assert_array_equal(self.vList[i, :], V[i, :])
            
    def testSetItem2(self):         
        V = numpy.random.rand(self.numVertices, self.numFeatures)        
        
        for i in range(self.numVertices): 
            self.vList[i, :] = V[i, :]
            nptst.assert_array_equal(self.vList[i, :], V[i, :])

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(VertexListTest)
    unittest.TextTestRunner(verbosity=2).run(suite)
    