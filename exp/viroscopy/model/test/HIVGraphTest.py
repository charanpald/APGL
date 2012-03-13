
import apgl
import numpy 
import unittest
try:
    from apgl.viroscopy.model.HIVGraph import HIVGraph
except ImportError:
    pass
from apgl.viroscopy.model.HIVVertices import HIVVertices

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
if __name__ == '__main__':
    unittest.main()

