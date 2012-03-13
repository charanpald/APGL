

import unittest
import numpy
import logging
from apgl.graph import *
from apgl.predictors.edge import *
from apgl.generator import * 

class SpectralEdgePredictorTest(unittest.TestCase):
    def setUp(self):
        numpy.seterr(all='ignore')


    def testEdgePredict(self):
        numVertices = 10
        numFeatures = 1

        vList = VertexList(numVertices, numFeatures)
        graph1 = SparseGraph(vList)
        graph2 = SparseGraph(vList)

        p = 0.5
        generator = ErdosRenyiGenerator(p)
        generator.generate(graph1)
        generator.generate(graph2)

        logging.debug(graph1)
        logging.debug(graph2)

        degree = 2
        predictor = SpectralEdgePredictor(degree)

        i = predictor.predictEdge(graph1, graph2)

        logging.debug(i)
        
    
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    