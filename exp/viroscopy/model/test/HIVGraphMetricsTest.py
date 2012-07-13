
import apgl
import numpy 
import unittest
import numpy.testing as nptst 

from exp.viroscopy.model.HIVGraph import HIVGraph
from exp.viroscopy.model.HIVVertices import HIVVertices
from exp.viroscopy.model.HIVGraphMetrics import HIVGraphMetrics, HIVGraphMetrics2
from exp.sandbox.GraphMatch import GraphMatch

class  HIVGraphMetricsTest(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(21)

    def testSummary(self): 
        numVertices = 10
        graph = HIVGraph(numVertices)

        graph.getVertexList().setInfected(1, 0.0)
        graph.getVertexList().setInfected(2, 2.0)
        graph.getVertexList().setInfected(7, 3.0)
        
        times = numpy.array([0, 1.0, 3.0, 4.0])
        
        metrics = HIVGraphMetrics(times)
        summary = metrics.summary(graph)
        
        summaryReal = numpy.array([[1,0], [1,0], [3, 0], [3,0]])
        nptst.assert_array_equal(summaryReal, summary)
    
        
    def testDistance(self): 
        summary1 = numpy.array([[1,0], [1,0], [3, 0], [3,0]])
        summary2 = numpy.array([[1,0], [1,0], [3, 0], [4,0]])
        
        times = numpy.array([0, 1.0, 3.0, 4.0])        
        
        self.assertEquals(HIVGraphMetrics(times).distance(summary1, summary2), numpy.linalg.norm(summary1 - summary2)) 
        
    def testShouldBreak(self): 
        numVertices = 10
        graph = HIVGraph(numVertices)

        graph.getVertexList().setInfected(1, 0.0)
        graph.getVertexList().setInfected(2, 2.0)
        graph.getVertexList().setInfected(7, 3.0)        
        
        summary1 = numpy.array([[1,0], [1,0], [3, 0], [3,0]])
        summary2 = numpy.array([[1,0], [2,0], [3, 0], [3,0]])
        
        times = numpy.array([0, 1.0, 3.0, 4.0])      
        epsilon = 1
        
        currentTime = 5
        self.assertTrue(HIVGraphMetrics(times).shouldBreak(summary2, graph, epsilon, currentTime))

        currentTime = 1        
        self.assertTrue(HIVGraphMetrics(times).shouldBreak(summary2, graph, epsilon, currentTime))
        
        currentTime = 0.9        
        self.assertFalse(HIVGraphMetrics(times).shouldBreak(summary2, graph, epsilon, currentTime))

    
    def testSummary2(self): 
        numVertices = 10
        graph = HIVGraph(numVertices)

        graph.getVertexList().setInfected(1, 0.0)
        graph.getVertexList().setDetected(1, 0.1, 0)
        graph.getVertexList().setInfected(2, 2.0)
        graph.getVertexList().setDetected(2, 2.0, 0)
        graph.getVertexList().setInfected(7, 3.0)
        graph.getVertexList().setDetected(7, 3.0, 0)
        
        times = numpy.array([0, 1.0, 2.9, 4.0, 10.0])
        
        metrics = HIVGraphMetrics2(times)
        summary = metrics.summary(graph)
        
        self.assertEquals(summary[0].size, 0)   
        self.assertEquals(summary[1].size, 1)
        self.assertEquals(summary[2].size, 2)
        self.assertEquals(summary[3].size, 3)

    def testShouldBreak2(self): 
        numVertices = 10
        graph1 = HIVGraph(numVertices)
        graph1.getVertexList().setInfected(1, 0.0)
        graph1.getVertexList().setDetected(1, 0.0, 0)
        graph1.getVertexList().setInfected(2, 2.0)
        graph1.getVertexList().setDetected(2, 2.0, 0)
        graph1.getVertexList().setInfected(7, 3.0) 
        graph1.getVertexList().setDetected(7, 3.0, 0)
        
        graph2 = HIVGraph(numVertices)
        graph2.getVertexList().setInfected(2, 0.0)
        graph2.getVertexList().setDetected(2, 0.0, 0)
        graph2.getVertexList().setInfected(3, 2.0)
        graph2.getVertexList().setDetected(3, 2.0, 0)
        graph2.getVertexList().setInfected(8, 3.0)
        graph2.getVertexList().setDetected(8, 3.0, 0)
        
        times = numpy.array([0, 1.0, 3.0, 4.0])
        metrics = HIVGraphMetrics2(times, GraphMatch(alpha=0.7))
        summary1 = metrics.summary(graph1)
        summary2 = metrics.summary(graph2)
        
        metrics.distance(summary1, summary2)
 
        times = numpy.array([0, 1.0, 3.0, 4.0])      
        epsilon = 0.05
        
        currentTime = 1
        self.assertFalse(metrics.shouldBreak(summary2, graph1, epsilon, currentTime))

        currentTime = 2        
        self.assertFalse(metrics.shouldBreak(summary2, graph1, epsilon, currentTime))
        
        currentTime = 3        
        self.assertFalse(metrics.shouldBreak(summary2, graph1, epsilon, currentTime))
        
        currentTime = 4        
        self.assertTrue(metrics.shouldBreak(summary2, graph1, epsilon, currentTime))

if __name__ == '__main__':
    unittest.main()
