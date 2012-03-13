
import unittest
import logging
from apgl.generator.BarabasiAlbertGenerator import BarabasiAlbertGenerator
from apgl.graph.VertexList import VertexList
from apgl.graph.SparseGraph import SparseGraph

class  BarabasiAlbertGeneratorTest(unittest.TestCase):
    def testGenerate(self):
        numFeatures = 1
        numVertices = 20

        vList = VertexList(numVertices, numFeatures)
        graph = SparseGraph(vList)

        ell = 2
        m = 0
        generator = BarabasiAlbertGenerator(ell, m)

        graph = generator.generate(graph)
        self.assertEquals(graph.getNumEdges(), 0)

        ell = 5
        graph.removeAllEdges()
        generator.setEll(ell)
        graph = generator.generate(graph)
        self.assertEquals(graph.getNumEdges(), 0)

        #Now test case where we m != 0
        ell = 2
        m = 1
        graph.removeAllEdges()
        generator.setEll(ell)
        generator.setM(m)
        graph = generator.generate(graph)
        self.assertEquals(graph.getNumEdges(), (numVertices-ell)*m)

        m = 2
        graph.removeAllEdges()
        generator.setM(m)
        graph = generator.generate(graph)
        self.assertEquals(graph.getNumEdges(), (numVertices-ell)*m)


    def testGraphDisplay(self):
        try:
            import networkx
            import matplotlib
        except ImportError as error:
            logging.debug(error)
            return 

        #Show
        numFeatures = 1
        numVertices = 20

        vList = VertexList(numVertices, numFeatures)
        graph = SparseGraph(vList)

        ell = 2
        m = 2
        generator = BarabasiAlbertGenerator(ell, m)

        graph = generator.generate(graph)

        logging.debug((graph.degreeDistribution()))

        nxGraph = graph.toNetworkXGraph()
        nodePositions = networkx.spring_layout(nxGraph)
        nodesAndEdges = networkx.draw_networkx(nxGraph, pos=nodePositions)
        #matplotlib.pyplot.show()

if __name__ == '__main__':
    unittest.main()

