import apgl
import numpy 
import unittest
import numpy.testing as nptst 

from exp.viroscopy.model.HIVGraph import HIVGraph
from exp.viroscopy.model.HIVVertices import HIVVertices
from exp.viroscopy.model.HIVGraphMetrics import HIVGraphMetrics
from exp.sandbox.GraphMatch import GraphMatch

class  HIVGraphMetrics2Test(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(21)
        numpy.set_printoptions(linewidth=100, suppress=True, precision=3)
        
        
        numVertices = 10
        self.graph = HIVGraph(numVertices)

        self.graph.getVertexList().setInfected(1, 0.0)
        self.graph.getVertexList().setDetected(1, 0.1, 0)
        self.graph.getVertexList().setInfected(2, 2.0)
        self.graph.getVertexList().setDetected(2, 2.0, 0)
        self.graph.getVertexList().setInfected(7, 3.0)
        self.graph.getVertexList().setDetected(7, 3.0, 0)

    def testAddGraph(self): 
        epsilon = 0.7 
        metrics = HIVGraphMetrics2(self.graph, epsilon)
      
        

if __name__ == '__main__':
    unittest.main()
