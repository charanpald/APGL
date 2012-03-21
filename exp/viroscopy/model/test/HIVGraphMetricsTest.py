
import apgl
import numpy 
import unittest
import numpy.testing as nptst 

from exp.viroscopy.model.HIVGraph import HIVGraph
from exp.viroscopy.model.HIVVertices import HIVVertices
from exp.viroscopy.model.HIVGraphMetrics import HIVGraphMetrics

class  HIVGraphMetricsTest(unittest.TestCase):
    def setup(self):
        pass

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
        

if __name__ == '__main__':
    unittest.main()
