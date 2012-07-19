
import apgl
import numpy 
import unittest
import pickle 
import numpy.testing as nptst 

from exp.viroscopy.model.HIVGraph import HIVGraph
from exp.viroscopy.model.HIVVertices import HIVVertices

@apgl.skipIf(not apgl.checkImport('pysparse'), 'No module pysparse')
class  HIVGraphTest(unittest.TestCase):
    def setup(self):
        pass

    def testContructor(self):
        numVertices = 10
        graph = HIVGraph(numVertices)

        
        self.assertEquals(numVertices, graph.getNumVertices())
        self.assertEquals(8, graph.getVertexList().getNumFeatures())
        self.assertTrue(graph.isUndirected() == True)

    def testGetSusceptibleSet(self):
        numVertices = 10
        graph = HIVGraph(numVertices)

        self.assertTrue(graph.getSusceptibleSet() == set(range(numVertices)))

        for i in range(9):
            graph.getVertexList().setInfected(i, 0.0)

        self.assertTrue(graph.getSusceptibleSet() == set([9]))

    def testGetInfectedSet(self):
        numVertices = 10
        graph = HIVGraph(numVertices)

        self.assertTrue(graph.getInfectedSet() == set([]))

        graph.getVertexList().setInfected(1, 0.0)
        graph.getVertexList().setInfected(3, 0.0)
        graph.getVertexList().setInfected(7, 0.0)

        self.assertTrue(graph.getInfectedSet() == set([1, 3, 7]))

    def testGetRemovedSet(self):
        numVertices = 10
        graph = HIVGraph(numVertices)

        self.assertTrue(graph.getRemovedSet() == set([]))

        graph.getVertexList().setInfected(1, 0.0)
        graph.getVertexList().setInfected(2, 0.0)
        graph.getVertexList().setInfected(7, 0.0)

        graph.getVertexList().setDetected(1, 0.0, HIVVertices.randomDetect)
        graph.getVertexList().setDetected(2, 0.0, HIVVertices.randomDetect)
        graph.getVertexList().setDetected(7, 0.0, HIVVertices.randomDetect)

        self.assertTrue(graph.getRemovedSet() == set([1, 2, 7]))


    def testDetectedNeighbours(self):
        numVertices = 10
        graph = HIVGraph(numVertices)

        self.assertTrue(graph.getRemovedSet() == set([]))

        graph.getVertexList().setInfected(1, 0.0)
        graph.getVertexList().setInfected(2, 0.0)
        graph.getVertexList().setInfected(7, 0.0)

        graph.getVertexList().setDetected(1, 0.0, HIVVertices.randomDetect)
        graph.getVertexList().setDetected(2, 0.0, HIVVertices.randomDetect)
        graph.getVertexList().setDetected(7, 0.0, HIVVertices.randomDetect)

        graph[0, 1] = 1
        graph[0, 2] = 1
        graph[0, 3] = 1
        graph[0, 4] = 1

        self.assertTrue((graph.detectedNeighbours(0) == numpy.array([1, 2])).all())

        graph.getVertexList().setInfected(3, 0.0)
        graph.getVertexList().setDetected(3, 0.0, HIVVertices.randomDetect)
        self.assertTrue((graph.detectedNeighbours(0) == numpy.array([1, 2, 3])).all())
        
    def testInfectedIndsAt(self): 
        numVertices = 10
        graph = HIVGraph(numVertices)

        self.assertTrue(graph.getRemovedSet() == set([]))

        graph.getVertexList().setInfected(1, 0.0)
        graph.getVertexList().setInfected(2, 2.0)
        graph.getVertexList().setInfected(7, 3.0)
        
        
        inds = graph.infectedIndsAt(10)
        nptst.assert_array_equal(inds, numpy.array([1, 2, 7]))
        
        graph.getVertexList().setInfected(5, 12.0)
        nptst.assert_array_equal(inds, numpy.array([1, 2, 7]))
                
    def testPickle(self): 
        numVertices = 10
        graph = HIVGraph(numVertices)  
        graph[0, 0] = 1
        graph[3, 5] = 0.1
        
        output = pickle.dumps(graph)
        newGraph = pickle.loads(output)
        
        graph[2, 2] = 1
        
        self.assertEquals(newGraph[0, 0], 1)
        self.assertEquals(newGraph[3, 5], 0.1)
        self.assertEquals(newGraph[2, 2], 0.0)
        self.assertEquals(newGraph.getNumEdges(), 2)
        self.assertEquals(newGraph.getNumVertices(), numVertices)
        self.assertEquals(newGraph.isUndirected(), True)
        
        self.assertEquals(graph[0, 0], 1)
        self.assertEquals(graph[3, 5], 0.1)
        self.assertEquals(graph[2, 2], 1)
        self.assertEquals(graph.getNumEdges(), 3)
        self.assertEquals(graph.getNumVertices(), numVertices)
        self.assertEquals(graph.isUndirected(), True)        
        
        for i in range(numVertices): 
            nptst.assert_array_equal(graph.getVertex(i), newGraph.getVertex(i))
        
if __name__ == '__main__':
    unittest.main()

