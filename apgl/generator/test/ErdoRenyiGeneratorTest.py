'''
Created on 3 Jul 2009

@author: charanpal
'''
from apgl.graph.DenseGraph import DenseGraph
from apgl.graph.SparseGraph import SparseGraph
from apgl.graph.VertexList import VertexList
from apgl.graph.GeneralVertexList import GeneralVertexList
from apgl.generator.ErdosRenyiGenerator import ErdosRenyiGenerator
import unittest
import logging

class ErdoRenyiGeneratorTest(unittest.TestCase):
    def setUp(self):    
        self.numVertices = 10; 
        self.numFeatures = 2; 
        
        self.vList = VertexList(self.numVertices, self.numFeatures)
        self.graph = SparseGraph(self.vList)
        self.p = 0.1 
        self.erg = ErdosRenyiGenerator(self.p)

        
    def testGenerate(self):
        #undirected = True 
        p = 0.0
             
        self.graph.removeAllEdges()
        self.erg.setP(p)
        graph = self.erg.generate(self.graph)
        self.assertEquals(graph.getNumEdges(), 0)
        
        undirected = False
        self.graph = SparseGraph(self.vList, undirected)
        self.graph.removeAllEdges()
        self.erg = ErdosRenyiGenerator(p)
        graph = self.erg.generate(self.graph)
        self.assertEquals(graph.getNumEdges(), 0)
        
        p = 1.0
        undirected = True
        self.graph = SparseGraph(self.vList, undirected)
        self.graph.removeAllEdges()
        self.erg = ErdosRenyiGenerator(p)
        graph = self.erg.generate(self.graph)
        self.assertEquals(graph.getNumEdges(), (self.numVertices*self.numVertices-self.numVertices)/2)
        
        p = 1.0
        undirected = False
        self.graph = SparseGraph(self.vList, undirected)
        self.graph.removeAllEdges()
        self.erg = ErdosRenyiGenerator(p)
        graph = self.erg.generate(self.graph)
        self.assertEquals(graph.getNumEdges(), self.numVertices*self.numVertices-self.numVertices)
        
        self.assertEquals(graph.getEdge(1, 2), 1)
        self.assertEquals(graph.getEdge(1, 1), None)

        p = 0.5
        numVertices = 1000
        numFeatures = 0

        vList = VertexList(numVertices, numFeatures)
        undirected = False
        self.graph = SparseGraph(vList, undirected)
        self.erg = ErdosRenyiGenerator(p)
        graph = self.erg.generate(self.graph)

        self.assertAlmostEquals(graph.getNumEdges()/float(numVertices**2 - numVertices), p, places=2)

        p = 0.1
        self.graph = SparseGraph(vList, undirected)
        self.erg = ErdosRenyiGenerator(p)
        graph = self.erg.generate(self.graph)

        self.assertAlmostEquals(graph.getNumEdges()/float(numVertices**2 - numVertices), p, places=2)

        #Test the case in which we have a graph with edges
        p = 0.5
        numVertices = 10 
        vList = VertexList(numVertices, numFeatures)
        graph = SparseGraph(vList, undirected)
        graph.addEdge(0, 1, 5)
        graph.addEdge(0, 2)
        graph.addEdge(0, 5, 0.7)
        graph.addEdge(1, 8)
        graph.addEdge(2, 9)
        numEdges = graph.getNumEdges()

        graph = self.erg.generate(graph, False)
        self.assertTrue(graph.getNumEdges() > numEdges)


    @unittest.skip("")
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

        p = 0.2
        generator = ErdosRenyiGenerator(p)

        graph = generator.generate(graph)
        logging.debug((graph.getNumEdges()))

        nxGraph = graph.toNetworkXGraph()
        nodePositions = networkx.spring_layout(nxGraph)
        nodesAndEdges = networkx.draw_networkx(nxGraph, pos=nodePositions)
        ax = matplotlib.pyplot.axes()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        #matplotlib.pyplot.show()


    def testErdosRenyiGenerations(self):
        numVertices = 20
        graph = DenseGraph(GeneralVertexList(numVertices))
        p = 0.2
        generator = ErdosRenyiGenerator(p)
        graph = generator.generate(graph)
    
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(ErdoRenyiGeneratorTest)
    unittest.TextTestRunner(verbosity=2).run(suite)