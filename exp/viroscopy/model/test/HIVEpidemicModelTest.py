
import apgl
import unittest
import logging
import sys
import numpy
import scipy.stats 

from exp.viroscopy.model.HIVEpidemicModel import HIVEpidemicModel
from exp.viroscopy.model.HIVRates import HIVRates
from exp.viroscopy.model.HIVGraph import HIVGraph
from exp.viroscopy.model.HIVVertices import HIVVertices
from apgl.graph import * 

@apgl.skipIf(not apgl.checkImport('pysparse'), 'No module pysparse')
class  HIVEpidemicModelTest(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(21)
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)

        M = 100
        undirected = True
        self.graph = HIVGraph(M, undirected)
        s = 3
        self.gen = scipy.stats.zipf(s)
        hiddenDegSeq = self.gen.rvs(size=self.graph.getNumVertices())
        rates = HIVRates(self.graph, hiddenDegSeq)
        self.model = HIVEpidemicModel(self.graph, rates)
        
    def testSimulate(self):
        T = 1.0

        self.graph.getVertexList().setInfected(0, 0.0)
        self.model.setT(T)

        times, infectedIndices, removedIndices, graph = self.model.simulate()

        numInfects = 0
        for i in range(graph.getNumVertices()):
            if graph.getVertex(i)[HIVVertices.stateIndex] == HIVVertices==infected:
                numInfects += 1

        self.assertTrue(numInfects == 0 or times[len(times)-1] >= T)

        #Test with a larger population as there seems to be an error when the
        #number of infectives becomes zero.
        M = 100
        undirected = True
        graph = HIVGraph(M, undirected)
        graph.setRandomInfected(10)

        T = 21.0
        hiddenDegSeq = self.gen.rvs(size=self.graph.getNumVertices())
        rates = HIVRates(self.graph, hiddenDegSeq)
        model = HIVEpidemicModel(graph, rates)
        model.setRecordStep(10)
        model.setT(T)

        times, infectedIndices, removedIndices, graph = model.simulate()
        self.assertTrue((times == numpy.array([0, 10, 20], numpy.int)).all())
        self.assertEquals(len(infectedIndices), 3)
        self.assertEquals(len(removedIndices), 3)


        #TODO: Much better testing 

    def testFindStandardResults(self):
        times = [3, 12, 22, 25, 40, 50]
        infectedIndices = [[1], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2]]
        removedIndices = [[1], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2]]

        self.model.setT(51.0)
        self.model.setRecordStep(10)

        times, infectedIndices, removedIndices = self.model.findStandardResults(times, infectedIndices, removedIndices)

        self.assertTrue((numpy.array(times)==numpy.arange(0, 60, 10)).all())


        #Now try case where simulation is slightly longer than T and infections = 0
        numpy.random.seed(21)
        times = [3, 12, 22, 25, 40, 50]
        infectedIndices = [[1], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2]]
        removedIndices = [[1], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2]]

        self.model.setT(51.0)
        self.model.setRecordStep(10)

        times, infectedIndices, removedIndices = self.model.findStandardResults(times, infectedIndices, removedIndices)
        logging.debug(times)


if __name__ == '__main__':
    unittest.main()

