'''
Created on 6 Jul 2009

@author: charanpal
'''
from apgl.io.PajekWriter import PajekWriter
from apgl.graph.DenseGraph import DenseGraph
from apgl.graph.DictGraph import DictGraph
from apgl.graph.SparseGraph import SparseGraph
from apgl.graph.VertexList import VertexList
from apgl.generator.ErdosRenyiGenerator import ErdosRenyiGenerator
from apgl.generator.SmallWorldGenerator import SmallWorldGenerator
from apgl.util.PathDefaults import PathDefaults
import unittest
import os

class PajekWriterTest(unittest.TestCase):
    def setUp(self):
        #Let's set up a very simple graph 
        numVertices = 5    
        numFeatures = 1    
        edges = []

        vList = VertexList(numVertices, numFeatures)
        
        #An undirected dense graph 
        self.dGraph1 = DenseGraph(vList, True)
        self.dGraph1.addEdge(0, 1, 1)
        self.dGraph1.addEdge(0, 2, 1)
        self.dGraph1.addEdge(2, 4, 1)
        self.dGraph1.addEdge(2, 3, 1)
        self.dGraph1.addEdge(3, 4, 1)
        
        #A directed sparse graph 
        self.dGraph2 = DenseGraph(vList, False)
        self.dGraph2.addEdge(0, 1, 1)
        self.dGraph2.addEdge(0, 2, 1)
        self.dGraph2.addEdge(2, 4, 1)
        self.dGraph2.addEdge(2, 3, 1)
        self.dGraph2.addEdge(3, 4, 1)
        
        #Now try sparse graphs 
        vList = VertexList(numVertices, numFeatures)
        self.sGraph1 = SparseGraph(vList, True)
        self.sGraph1.addEdge(0, 1, 1)
        self.sGraph1.addEdge(0, 2, 1)
        self.sGraph1.addEdge(2, 4, 1)
        self.sGraph1.addEdge(2, 3, 1)
        self.sGraph1.addEdge(3, 4, 1)
        
        self.sGraph2 = SparseGraph(vList, False)
        self.sGraph2.addEdge(0, 1, 1)
        self.sGraph2.addEdge(0, 2, 1)
        self.sGraph2.addEdge(2, 4, 1)
        self.sGraph2.addEdge(2, 3, 1)
        self.sGraph2.addEdge(3, 4, 1)

        #Finally, try DictGraphs
        self.dctGraph1 = DictGraph(True)
        self.dctGraph1.addEdge(0, 1, 1)
        self.dctGraph1.addEdge(0, 2, 2)
        self.dctGraph1.addEdge(2, 4, 8)
        self.dctGraph1.addEdge(2, 3, 1)
        self.dctGraph1.addEdge(12, 4, 1)

        self.dctGraph2 = DictGraph(False)
        self.dctGraph2.addEdge(0, 1, 1)
        self.dctGraph2.addEdge(0, 2, 1)
        self.dctGraph2.addEdge(2, 4, 1)
        self.dctGraph2.addEdge(2, 3, 1)
        self.dctGraph2.addEdge(12, 4, 1)

        
    def tearDown(self):
        pass

    def testInit(self):
        pass        

    def testWriteToFile(self):
        pw = PajekWriter()
        directory = PathDefaults.getOutputDir() + "test/"
        
        #Have to check the files
        fileName1 = directory + "denseTestUndirected"
        pw.writeToFile(fileName1, self.dGraph1)
        
        fileName2 = directory + "denseTestDirected"
        pw.writeToFile(fileName2, self.dGraph2)
        
        fileName3 = directory + "sparseTestUndirected"
        pw.writeToFile(fileName3, self.sGraph1)
        
        fileName4 = directory + "sparseTestDirected"
        pw.writeToFile(fileName4, self.sGraph2)

        fileName5 = directory + "dictTestUndirected"
        pw.writeToFile(fileName5, self.dctGraph1)

        fileName6 = directory + "dictTestDirected"
        pw.writeToFile(fileName6, self.dctGraph2)

    def testWriteToFile2(self):
        pw = PajekWriter()
        directory = PathDefaults.getOutputDir() + "test/"

        def setVertexColour(vertexIndex, graph):
            colours = ["grey05", "grey10", "grey15", "grey20", "grey25"]
            return colours[vertexIndex]

        def setVertexSize(vertexIndex, graph):
            return vertexIndex

        def setEdgeColour(vertexIndex1, vertexIndex2, graph):
            colours = ["grey05", "grey10", "grey15", "grey20", "grey25"]
            return colours[vertexIndex1]

        def setEdgeSize(vertexIndex1, vertexIndex2, graph):
            return vertexIndex1+vertexIndex2

        pw.setVertexColourFunction(setVertexColour)
        fileName1 = directory + "vertexColourTest"
        pw.writeToFile(fileName1, self.dGraph1)
        pw.setVertexColourFunction(None)

        pw.setVertexSizeFunction(setVertexSize)
        fileName1 = directory + "vertexSizeTest"
        pw.writeToFile(fileName1, self.dGraph1)
        pw.setVertexSizeFunction(None)

        pw.setEdgeColourFunction(setEdgeColour)
        fileName1 = directory + "edgeColourTest"
        pw.writeToFile(fileName1, self.dGraph1)
        pw.setEdgeColourFunction(None)

        pw.setEdgeSizeFunction(setEdgeSize)
        fileName1 = directory + "edgeSizeTest"
        pw.writeToFile(fileName1, self.dGraph1)
        pw.setEdgeColourFunction(None)

    def testWriteToFile3(self):
        """
        We will test out writing out some random graphs to Pajek
        """
        numVertices = 20
        numFeatures = 0 
        vList = VertexList(numVertices, numFeatures)
        graph = SparseGraph(vList)

        p = 0.1
        generator = ErdosRenyiGenerator(p)
        graph = generator.generate(graph)

        pw = PajekWriter()
        directory = PathDefaults.getOutputDir() + "test/"
        pw.writeToFile(directory + "erdosRenyi20", graph)

        #Now write a small world graph
        p = 0.2
        k = 3

        graph.removeAllEdges()
        generator = SmallWorldGenerator(p, k)
        graph = generator.generate(graph)

        pw.writeToFile(directory + "smallWorld20", graph)
        

 

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(PajekWriterTest)
    unittest.TextTestRunner(verbosity=2).run(suite)