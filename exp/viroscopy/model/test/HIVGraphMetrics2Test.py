import apgl
import numpy 
import unittest
import numpy.testing as nptst 

from exp.viroscopy.model.HIVGraph import HIVGraph
from exp.viroscopy.model.HIVGraphMetrics2 import HIVGraphMetrics2
from exp.sandbox.GraphMatch import GraphMatch

class  HIVGraphMetrics2Test(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(21)
        numpy.set_printoptions(linewidth=100, suppress=True, precision=3)
        
        
        numVertices = 10
        self.graph = HIVGraph(numVertices)

        self.graph.vlist.setInfected(1, 0.0)
        self.graph.vlist.setDetected(1, 0.1, 0)
        self.graph.vlist.setInfected(2, 2.0)
        self.graph.vlist.setDetected(2, 2.0, 0)
        self.graph.vlist.setInfected(7, 3.0)
        self.graph.vlist.setDetected(7, 3.0, 0)

    def testAddGraph(self): 
        epsilon = 0.12
        metrics = HIVGraphMetrics2(self.graph, epsilon)
        
        metrics.addGraph(self.graph)
      
        self.assertEquals(metrics.dists[0], 0.0)
        self.assertEquals(metrics.meanDistance(), 0.0)
        
        #Start a new graph 
        #Compute distances directly 
        matcher = GraphMatch("U")
        graph =  HIVGraph(self.graph.size)
        dists = [] 
        metrics = HIVGraphMetrics2(self.graph, epsilon)
        
        graph.vlist.setInfected(1, 0.0)
        graph.vlist.setDetected(1, 0.1, 0)
        metrics.addGraph(graph)
        
        t = graph.endTime()
        subgraph1 = graph.subgraph(graph.removedIndsAt(t))
        subgraph2 = self.graph.subgraph(graph.removedIndsAt(t)) 
        permutation, distance, time = matcher.match(subgraph1, subgraph2)
        lastDist = matcher.distance(subgraph1, subgraph2, permutation, True, True) 
        self.assertEquals(metrics.dists[-1], lastDist)
        self.assertTrue(metrics.shouldBreak())
        
        graph.vlist.setInfected(2, 2.0)
        graph.vlist.setDetected(2, 2.0, 0)
        metrics.addGraph(graph)
        
        t = graph.endTime()
        subgraph1 = graph.subgraph(graph.removedIndsAt(t))
        subgraph2 = self.graph.subgraph(graph.removedIndsAt(t)) 
        permutation, distance, time = matcher.match(subgraph1, subgraph2)
        lastDist = matcher.distance(subgraph1, subgraph2, permutation, True, True) 
        self.assertEquals(metrics.dists[-1], lastDist)   
        self.assertTrue(metrics.shouldBreak())
        
        graph.vlist.setInfected(7, 3.0)
        graph.vlist.setDetected(7, 3.0, 0)
        metrics.addGraph(graph)
        
        t = graph.endTime()
        subgraph1 = graph.subgraph(graph.removedIndsAt(t))
        subgraph2 = self.graph.subgraph(graph.removedIndsAt(t)) 
        permutation, distance, time = matcher.match(subgraph1, subgraph2)
        lastDist = matcher.distance(subgraph1, subgraph2, permutation, True, True) 
        self.assertEquals(metrics.dists[-1], lastDist) 
        self.assertFalse(metrics.shouldBreak())
        
        

if __name__ == '__main__':
    unittest.main()
