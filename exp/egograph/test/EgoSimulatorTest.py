'''
Created on 23 Jul 2009

@author: charanpal
'''
from apgl.predictors.NaiveBayes import NaiveBayes
from apgl.predictors.MlpySVM import MlpySVM
from apgl.data.ExamplesList import ExamplesList
from apgl.data.Standardiser import Standardiser
from apgl.egograph.EgoGenerator import EgoGenerator
from apgl.egograph.EgoSimulator import EgoSimulator
from apgl.egograph.EgoUtils import EgoUtils
from apgl.generator.SmallWorldGenerator import SmallWorldGenerator
from apgl.graph.SparseGraph import SparseGraph
from apgl.graph.VertexList import VertexList
from apgl.util import * 
import numpy
import numpy.random as rand
import unittest
import apgl
import logging

try:
    import mlpy
except ImportError:
    pass

@apgl.skipIf(not apgl.checkImport('mlpy'), 'No module mlpy')
class EgoSimulatorTest(unittest.TestCase):
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

        dataDir = PathDefaults.getDataDir() + "infoDiffusion/"
        matFileName = dataDir + "EgoAlterTransmissions1000.mat"
        sampleSize = 100
        egoAlterExamplesList = ExamplesList.readFromMatFile(matFileName)
        egoAlterExamplesList.setDefaultExamplesName("X")
        egoAlterExamplesList.setLabelsName("y")
        egoAlterExamplesList.randomSubData(sampleSize)

        X = egoAlterExamplesList.getDataField("X")
        y = egoAlterExamplesList.getDataField("y")

        #Now learn using NaiveBayes
        self.nb = NaiveBayes()
        self.nb.learnModel(X, y)
    
        self.egoSimulator = EgoSimulator(self.sGraph, self.nb)

        #Define a classifier which predicts transfer if gender is female
        class DummyClassifier(object):
            def __init(self):
                pass

            def classify(self, X):
                y = numpy.zeros((X.shape[0]))

                for i in range(X.shape[0]):
                    if X[i, 0] == 0:
                        y[i] = 1
                    else:
                        y[i] = -1
                return y

        self.dc = DummyClassifier()
        
    def tearDown(self):
        pass


    def testAdvanceGraph(self):    
        totalInfo = EgoUtils.getTotalInformation(self.sGraph)
                
        self.sGraph = self.egoSimulator.advanceGraph()     
        totalInfo2 = EgoUtils.getTotalInformation(self.sGraph)
        
        #Test that the number of people who know information is the same or more 
        self.assertTrue(totalInfo2 >= totalInfo)
        
        
    def testAdvanceGraph2(self):
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
        

            
        
        
        
        simulator = EgoSimulator(sGraph, self.dc)
        simulator.advanceGraph()
        
        
        
        self.assertEquals(simulator.getNumIterations(), 1)
        
        self.assertEquals(sGraph.getVertex(0)[numFeatures-1], 1)
        self.assertEquals(sGraph.getVertex(1)[numFeatures-1], 1)
        self.assertEquals(sGraph.getVertex(2)[numFeatures-1], 1)
        self.assertEquals(sGraph.getVertex(3)[numFeatures-1], 1)
        self.assertEquals(sGraph.getVertex(4)[numFeatures-1], 1)
        self.assertEquals(sGraph.getVertex(5)[numFeatures-1], 1)
        self.assertEquals(sGraph.getVertex(6)[numFeatures-1], 1)
        self.assertEquals(sGraph.getVertex(7)[numFeatures-1], 0)
        self.assertEquals(sGraph.getVertex(8)[numFeatures-1], 1)
        self.assertEquals(sGraph.getVertex(9)[numFeatures-1], 0)
        
        
        #Advance again and all egos have information 
        simulator.advanceGraph()
        
        self.assertEquals(simulator.getNumIterations(), 2)

        self.assertEquals(sGraph.getVertex(0)[numFeatures-1], 1)
        self.assertEquals(sGraph.getVertex(1)[numFeatures-1], 1)
        self.assertEquals(sGraph.getVertex(2)[numFeatures-1], 1)
        self.assertEquals(sGraph.getVertex(3)[numFeatures-1], 1)
        self.assertEquals(sGraph.getVertex(4)[numFeatures-1], 1)
        self.assertEquals(sGraph.getVertex(5)[numFeatures-1], 1)
        self.assertEquals(sGraph.getVertex(6)[numFeatures-1], 1)
        self.assertEquals(sGraph.getVertex(7)[numFeatures-1], 1)
        self.assertEquals(sGraph.getVertex(8)[numFeatures-1], 1)
        self.assertEquals(sGraph.getVertex(9)[numFeatures-1], 1)
        
        #Should be no change 
        simulator.advanceGraph()
        
        self.assertEquals(simulator.getNumIterations(), 3)
        
        #Check the correct alters are added at each step 
        self.assertTrue((simulator.getAlters(0) == numpy.array([1,2,3,6])).all())
        self.assertTrue((simulator.getAlters(1) == numpy.array([7,9])).all())
        self.assertTrue((simulator.getAlters(2) == numpy.array([])).all())

        #Check that the transmission graph is okay
        transGraph = simulator.getTransmissionGraph()
        self.assertEquals(transGraph.getNumVertices(), 9)
        self.assertEquals(transGraph.getNumEdges(), 7)
        self.assertEquals(transGraph.getAllVertexIds(), [0, 1, 2, 3, 4, 6, 7, 8, 9])

        for i in transGraph.getAllVertexIds():
            self.assertTrue((transGraph.getVertex(i) == sGraph.getVertex(i)).all())
        
    def testFullTransGraph(self):
        transGraph = self.egoSimulator.fullTransGraph()

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

        simulator = EgoSimulator(sGraph, self.dc)
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

        self.assertEquals(transGraph.getVertexList(), vList)
      
    def testAdvanceGraph3(self):
        """ 
        This test will learn from a set of ego and alter pairs, then we will make predictions on 
        the pairs and see the results. The we test if the same results are present in a simulation.  
        """
        dataDir = PathDefaults.getDataDir() + "infoDiffusion/"
        matFileName = dataDir +  "EgoAlterTransmissions1000.mat"
        examplesList = ExamplesList.readFromMatFile(matFileName)
        examplesList.setDefaultExamplesName("X")
        examplesList.setLabelsName("y")
        
        logging.debug(("Number of y = +1: " + str(sum(examplesList.getSampledDataField("y") == 1))))
        logging.debug(("Number of y = -1: " + str(sum(examplesList.getSampledDataField("y") == -1))))
        
        #Standardise the examples 
        preprocessor = Standardiser()
        X = examplesList.getDataField(examplesList.getDefaultExamplesName())
        X = preprocessor.standardiseArray(X)
        examplesList.overwriteDataField(examplesList.getDefaultExamplesName(), X)
        
        classifier = MlpySVM(kernel='linear', kp=1, C=32.0)

        y = examplesList.getDataField("y")
        classifier.learnModel(X, y)
        predY = classifier.classify(X)
        logging.debug(("Number of y = +1: " + str(sum(examplesList.getSampledDataField("y") == 1))))
        logging.debug(("Number of y = -1: " + str(sum(examplesList.getSampledDataField("y") == -1))))

        sampledY = examplesList.getSampledDataField(examplesList.getLabelsName()).ravel()

        error = mlpy.err(sampledY, predY)
        sensitivity = mlpy.sens(sampledY, predY)
        specificity = mlpy.spec(sampledY, predY)
        errorP = mlpy.errp(sampledY, predY)
        errorN = mlpy.errn(sampledY, predY)
        
        logging.debug("--- Classification evaluation ---")
        logging.debug(("Error on " + str(examplesList.getNumExamples()) + " examples is " + str(error)))
        logging.debug(("Sensitivity (recall = TP/(TP+FN)): " + str(sensitivity)))
        logging.debug(("Specificity (TN/TN+FP): "  + str(specificity)))
        logging.debug(("Error on positives: "  + str(errorP)))
        logging.debug(("Error on negatives: "  + str(errorN)))
        
        sGraph = EgoUtils.graphFromMatFile(matFileName)

        #Notice that the data is preprocessed in the same way as the survey data 
        egoSimulator = EgoSimulator(sGraph, classifier, preprocessor)
        
        totalInfo = EgoUtils.getTotalInformation(sGraph)
        logging.debug(("Total number of people with information: " + str(totalInfo)))
        self.assertEquals(totalInfo, 1000)
        
        sGraph = egoSimulator.advanceGraph()
        
        totalInfo = EgoUtils.getTotalInformation(sGraph)
        logging.debug(("Total number of people with information: " + str(totalInfo)))
        self.assertEquals(totalInfo, 1000 + sum(predY == 1))
        
        altersList = egoSimulator.getAlters(0)
        predictedAlters = numpy.nonzero(predY == 1)[0]
        
        self.assertTrue((altersList == predictedAlters*2+1).all())



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    