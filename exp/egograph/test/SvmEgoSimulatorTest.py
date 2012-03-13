
from apgl.util.PathDefaults import PathDefaults

import unittest
import logging
import sys
import apgl
from apgl.egograph.SvmEgoSimulator import SvmEgoSimulator
from apgl.generator import *
from apgl.graph import * 

@apgl.skipIf(not apgl.checkImport('svm'), 'No module svm')
class SvmEgoSimulatorTest(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        dataDir = PathDefaults.getDataDir() + "infoDiffusion/"
        #dataDir = "/home/charanpal/Private/Postdoc/Code/APGL/data/"
        matFileName = dataDir + "EgoAlterTransmissions.mat"
        self.svmEgoSimulator = SvmEgoSimulator(matFileName)

    def testSampleExamples(self):
        self.svmEgoSimulator.sampleExamples(100)

        self.assertEquals(self.svmEgoSimulator.examplesList.getNumSampledExamples(), 100)

    def testModelSelection(self):
        Cs = [1.0, 2.0]
        kernel = "linear"
        kernelParams = [0.0]
        errorCosts = [0.1, 0.2]
        folds = 5
        sampleSize = 1000
        
        CVal, kernelParamVal, errorCost, error = self.svmEgoSimulator.modelSelection(Cs, kernel, kernelParams, errorCosts, folds, sampleSize)

        self.assertTrue(CVal in Cs)
        self.assertTrue(kernelParamVal in kernelParams)
        self.assertTrue(errorCost in errorCosts)
        self.assertTrue(error >= 0.0 and error < 1.0)

    def testEvaluateClassifier(self):
        CVal = 1.0
        kernel = "linear"
        kernelParamVal = 0.0
        errorCost = 0.5
        folds = 6
        sampleSize = 1000
        invert = False

        (means, vars) = self.svmEgoSimulator.evaluateClassifier(CVal, kernel, kernelParamVal, errorCost, folds, sampleSize, invert)


    def testTrainClassifier(self):
        CVal = 1.0
        kernel = "linear"
        kernelParamVal = 0.0
        errorCost = 0.5
        folds = 6
        sampleSize = 1000

        self.svmEgoSimulator.trainClassifier(CVal, kernel, kernelParamVal, errorCost, sampleSize)

    def testGenerateRandomGraph(self):
        egoFileName = PathDefaults.getDataDir() + "infoDiffusion/EgoData.csv"
        alterFileName = PathDefaults.getDataDir()  + "infoDiffusion/AlterData.csv"
        numVertices = 1000
        infoProb = 0.1

        
        p = 0.1
        neighbours = 10
        generator = SmallWorldGenerator(p, neighbours)
        graph = SparseGraph(VertexList(numVertices, 0))
        graph = generator.generate(graph)

        self.svmEgoSimulator.generateRandomGraph(egoFileName, alterFileName, infoProb, graph)

    def testRunSimulation(self):
        egoFileName = PathDefaults.getDataDir() + "infoDiffusion/EgoData.csv"
        alterFileName = PathDefaults.getDataDir()  + "infoDiffusion/AlterData.csv"
        numVertices = 1000
        infoProb = 0.1
        p = 0.1
        neighbours = 10

        generator = SmallWorldGenerator(p, neighbours)
        graph = SparseGraph(VertexList(numVertices, 0))
        graph = generator.generate(graph)
        
        CVal = 1.0
        kernel = "linear"
        kernelParamVal = 0.0
        errorCost = 0.5
        folds = 6
        sampleSize = 1000

        maxIterations = 5

        self.svmEgoSimulator.trainClassifier(CVal, kernel, kernelParamVal, errorCost, sampleSize)
        self.svmEgoSimulator.generateRandomGraph(egoFileName, alterFileName, infoProb, graph)
        self.svmEgoSimulator.runSimulation(maxIterations)


if __name__ == '__main__':
    unittest.main()

