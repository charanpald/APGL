'''
Created on 3 Jul 2009

@author: charanpal
'''
from apgl.graph.VertexList import VertexList
from apgl.graph.SparseGraph import SparseGraph
from apgl.generator.SmallWorldGenerator import SmallWorldGenerator
from apgl.io.PajekWriter import PajekWriter
import unittest
import logging

class SmallWorldGeneratorTest(unittest.TestCase):
    def setUp(self):    
        self.numVertices = 100; 
        self.numFeatures = 2; 

        p = 0.1
        k = 10 

        self.vList = VertexList(self.numVertices, self.numFeatures)
        self.graph = SparseGraph(self.vList)
        self.swg = SmallWorldGenerator(p, k)
        
    def testgenerate(self):
        p = 0.0
        k = 1

        self.swg.setP(p)
        self.swg.setK(k)
        sGraph = self.swg.generate(self.graph)
        self.assertEquals(sGraph.getNumEdges(), sGraph.getNumVertices())
        
        for i in range(self.numVertices): 
            for j in range(k):
                self.assertEquals(sGraph.getEdge(i, (i+j+1)%self.numVertices), 1)
        
        k = 3
        sGraph.removeAllEdges()
        self.swg.setP(p)
        self.swg.setK(k)
        sGraph = self.swg.generate(self.graph)
        
        self.assertEquals(sGraph.getNumEdges(), sGraph.getNumVertices()*k)
        
        for i in range(self.numVertices): 
            for j in range(k):
                self.assertEquals(sGraph.getEdge(i, (i + j + 1) % self.numVertices), 1)
           
        p = 0.5 
        k = 1    
        sGraph.removeAllEdges()
        self.swg.setP(p)
        self.swg.setK(k)
        sGraph = self.swg.generate(self.graph)
        self.assertEquals(sGraph.getNumEdges(), sGraph.getNumVertices())
        
        p = 0.1 
        k = 2    
        sGraph.removeAllEdges()
        self.swg.setP(p)
        self.swg.setK(k)
        sGraph = self.swg.generate(self.graph)
        self.assertEquals(sGraph.getNumEdges(), sGraph.getNumVertices()*k)

    def tearDown(self):
        pass

    def testInit(self):
        pass     

    def testGetClusteringCoefficient(self):
        p = 0.0
        k = 10

        self.swg.setP(p)
        self.swg.setK(k)

        cc = 3*(k-1)*(1-p)**3/(2*(2*k-1))
        self.assertEquals(self.swg.clusteringCoefficient(), cc)

        p = 0.5
        self.swg.setP(p)
        cc = 3*(k-1)*(1-p)**3/(2*(2*k-1))
        self.assertEquals(self.swg.clusteringCoefficient(), cc)

        k = 1
        self.swg.setK(k)
        self.assertEquals(self.swg.clusteringCoefficient(), 0)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(SmallWorldGeneratorTest)
    unittest.TextTestRunner(verbosity=2).run(suite)