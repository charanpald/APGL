
import apgl
import numpy 
import unittest
import pickle 
import numpy.testing as nptst 

from exp.viroscopy.model.HIVVertices import HIVVertices
from exp.viroscopy.HIVGraphReader import HIVGraphReader

class  HIVGraphReaderTest(unittest.TestCase):
    def setUp(self):
        numpy.set_printoptions(suppress=True, precision=3)
    
    def testreadSimulationHIVGraph(self): 
        
        hivReader = HIVGraphReader()
        graph = hivReader.readSimulationHIVGraph()
        
        
        self.assertEquals(graph.getNumVertices(), 5389)
        self.assertEquals(graph.getNumEdges(), 4097)
        #Check the edges of the first few vertices 
        nptst.assert_array_equal(graph.neighbours(0), numpy.array([50]))
        nptst.assert_array_equal(graph.neighbours(1), numpy.array([1380]))
        nptst.assert_array_equal(graph.neighbours(2), numpy.array([]))
        nptst.assert_array_equal(graph.neighbours(3), numpy.array([]))
        nptst.assert_array_equal(graph.neighbours(30), numpy.array([43, 51, 57, 151, 304, 386,913]))
        
        #Check vertex values 
        v = graph.getVertex(0)
        self.assertEquals(v[HIVVertices.dobIndex], 19602)
        self.assertEquals(v[HIVVertices.genderIndex], HIVVertices.male)
        self.assertEquals(v[HIVVertices.orientationIndex], HIVVertices.hetero)
        self.assertEquals(v[HIVVertices.stateIndex], HIVVertices.removed)
        self.assertEquals(v[HIVVertices.infectionTimeIndex], -1)
        self.assertEquals(v[HIVVertices.detectionTimeIndex], 31563)
        self.assertEquals(v[HIVVertices.detectionTypeIndex], HIVVertices.randomDetect)
        self.assertEquals(v[HIVVertices.hiddenDegreeIndex], 3)
        
        v = graph.getVertex(1)
        self.assertEquals(v[HIVVertices.dobIndex], 21508)
        self.assertEquals(v[HIVVertices.genderIndex], HIVVertices.male)
        self.assertEquals(v[HIVVertices.orientationIndex], HIVVertices.hetero)
        self.assertEquals(v[HIVVertices.stateIndex], HIVVertices.removed)
        self.assertEquals(v[HIVVertices.infectionTimeIndex], -1)
        self.assertEquals(v[HIVVertices.detectionTimeIndex], 31563)
        self.assertEquals(v[HIVVertices.detectionTypeIndex], HIVVertices.randomDetect)
        self.assertEquals(v[HIVVertices.hiddenDegreeIndex], 4)
        
        v = graph.getVertex(6)
        self.assertEquals(v[HIVVertices.dobIndex], 21676)
        self.assertEquals(v[HIVVertices.genderIndex], HIVVertices.female)
        self.assertEquals(v[HIVVertices.orientationIndex], HIVVertices.hetero)
        self.assertEquals(v[HIVVertices.stateIndex], HIVVertices.removed)
        self.assertEquals(v[HIVVertices.infectionTimeIndex], -1)
        self.assertEquals(v[HIVVertices.detectionTimeIndex], 31472)
        self.assertEquals(v[HIVVertices.detectionTypeIndex], HIVVertices.contactTrace)
        self.assertEquals(v[HIVVertices.hiddenDegreeIndex], 1)
        
        v = graph.getVertex(5381)
        self.assertEquals(v[HIVVertices.dobIndex], 25307)
        self.assertEquals(v[HIVVertices.genderIndex], HIVVertices.male)
        self.assertEquals(v[HIVVertices.orientationIndex], HIVVertices.bi)
        self.assertEquals(v[HIVVertices.stateIndex], HIVVertices.removed)
        self.assertEquals(v[HIVVertices.infectionTimeIndex], -1)
        self.assertEquals(v[HIVVertices.detectionTimeIndex], 38231)
        self.assertEquals(v[HIVVertices.detectionTypeIndex], HIVVertices.randomDetect)
        self.assertEquals(v[HIVVertices.hiddenDegreeIndex], 0)


if __name__ == '__main__':
    unittest.main()

