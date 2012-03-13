
from apgl.predictors import * 
from apgl.data import * 
from apgl.egograph.EgoGenerator import EgoGenerator
from apgl.egograph.EgoSimulator2 import EgoSimulator2
from apgl.egograph.EgoUtils import EgoUtils
from apgl.graph import *
from apgl.generator import *
from apgl.util import *
import numpy
import numpy.random as rand
import unittest
import logging


class EgoSimulator2Test(unittest.TestCase):
    def setUp(self):
        numVertices = 500
        numFeatures = 49

        self.means = rand.randn(numFeatures)
        self.vars = rand.randn(numFeatures, numFeatures)
        self.vars = self.vars + self.vars.T #Make vars symmetric
        p1 = 0.1

        self.egoGenerator = EgoGenerator()
        vList = self.egoGenerator.generateIndicatorVertices(numVertices, self.means, self.vars, p1)
        sGraph = SparseGraph(vList)

        p2 = 0.1
        k = 5

        #Create the graph edges according to the small world model
        graphGen = SmallWorldGenerator(p2, k)
        self.sGraph = graphGen.generate(sGraph)

        edges = sGraph.getAllEdges()

        for i in range(edges.shape[0]):
            sGraph.addEdge(edges[i, 0], edges[i, 1], numpy.random.rand())

        #Now learn using an edge label predictor
        lmbda = 0.01
        alterRegressor = PrimalRidgeRegression(lmbda)
        egoRegressor = PrimalRidgeRegression(lmbda)
        self.predictor = EgoEdgeLabelPredictor(alterRegressor, egoRegressor)
        self.predictor.learnModel(sGraph)

        self.egoSimulator = EgoSimulator2(self.sGraph, self.predictor)

        #Define a classifier which predicts transfer if gender of ego is female
        class DummyClassifier(object):
            def __init(self):
                pass

            def learnModel(self, graph):
                pass 

            def predictEdges(self, graph, edges):
                y = numpy.zeros((edges.shape[0]))

                for i in range(edges.shape[0]):
                    V = graph.getVertexList().getVertices(list(range(0, graph.getNumVertices())))
                    if V[edges[i, 0], 0] == 0:
                        y[i] = 1
                    else:
                        y[i] = -1
                return y

        self.dc = DummyClassifier()


    def tearDown(self):
        pass


    #Don't really care about this function 
    def testAdvanceGraph(self):
        totalInfo = EgoUtils.getTotalInformation(self.sGraph)

        self.sGraph = self.egoSimulator.advanceGraph()
        totalInfo2 = EgoUtils.getTotalInformation(self.sGraph)

        #Test that the number of people who know information is the same or more
        logging.debug(totalInfo)
        logging.debug(totalInfo2)
        self.assertTrue(totalInfo2 >= totalInfo)

    def testFullTransGraph(self):
        #Create a simple graph and deterministic classifier
        numExamples = 10
        numFeatures = 3

        #Here, the first element is gender (say) with female = 0, male = 1
        vList = VertexList(numExamples, numFeatures)
        vList.setVertex(0, numpy.array([0,0,1]))
        vList.setVertex(1, numpy.array([1,0,0]))
        vList.setVertex(2, numpy.array([1,0,0]))
        vList.setVertex(3, numpy.array([1,0,0]))
        vList.setVertex(4, numpy.array([0,0,1]))
        vList.setVertex(5, numpy.array([0,0,1]))
        vList.setVertex(6, numpy.array([0,0,0]))
        vList.setVertex(7, numpy.array([1,0,0]))
        vList.setVertex(8, numpy.array([0,0,1]))
        vList.setVertex(9, numpy.array([1,0,0]))

        sGraph = SparseGraph(vList)
        sGraph.addEdge(0, 1, 1)
        sGraph.addEdge(0, 2, 1)
        sGraph.addEdge(0, 3, 1)
        sGraph.addEdge(4, 5, 1)
        sGraph.addEdge(4, 6, 1)
        sGraph.addEdge(6, 7, 1)
        sGraph.addEdge(6, 8, 1)
        sGraph.addEdge(6, 9, 1)

        simulator = EgoSimulator2(sGraph, self.dc)
        logging.debug("Writing out full transmission graph")
        transGraph = simulator.fullTransGraph()

        self.assertEquals(transGraph.isUndirected(), False)
        self.assertEquals(transGraph.getNumEdges(), 11)
        self.assertEquals(transGraph.getEdge(0,1), 1)
        self.assertEquals(transGraph.getEdge(0,2), 1)
        self.assertEquals(transGraph.getEdge(0,3), 1)
        self.assertEquals(transGraph.getEdge(4,5), 1)
        self.assertEquals(transGraph.getEdge(4,6), 1)
        self.assertEquals(transGraph.getEdge(5,4), 1)
        self.assertEquals(transGraph.getEdge(6,4), 1)
        self.assertEquals(transGraph.getEdge(6,7), 1)
        self.assertEquals(transGraph.getEdge(6,8), 1)
        self.assertEquals(transGraph.getEdge(6,9), 1)
        self.assertEquals(transGraph.getEdge(8,6), 1)

        #self.assertEquals(transGraph.getVertexList(), vList)