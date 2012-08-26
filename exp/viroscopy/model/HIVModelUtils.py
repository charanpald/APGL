

"""
Keep some default parameters for the epidemic model. 
"""
import numpy 
import logging 
from apgl.util import Util 
from apgl.util import PathDefaults 
from apgl.graph.GraphStatistics import GraphStatistics
from exp.viroscopy.model.HIVGraph import HIVGraph
from exp.viroscopy.model.HIVEpidemicModel import HIVEpidemicModel
from exp.viroscopy.model.HIVRates import HIVRates
from exp.viroscopy.model.HIVVertices import HIVVertices
from exp.viroscopy.HIVGraphReader import HIVGraphReader, CsvConverters

class HIVModelUtils(object):
    def __init__(self): 
        pass 
    
    @staticmethod
    def estimatedRealTheta():
        """
        This is taken from simulated runs using the real data 
        """
        theta = numpy.array([ 150,  0.5, 0.4, 0.3, 0.001, 0.05, 0.2, 0.02, 0.0038, 0.003, 0.017])
        sigmaTheta = numpy.array([100, 0.3, 0.1, 0.1, 0.001, 0.1, 0.1, 0.02, 0.02, 0.02, 0.02])
        return theta, sigmaTheta 
    
    @staticmethod
    def toyTheta(): 
        theta = numpy.array([50, 0.5, 1.0, 0.5, 1.0/800, 0.01, 0.05, 0.1, 38.0/1000, 30.0/1000, 170.0/1000])
        sigmaTheta = theta/2
        return theta, sigmaTheta 
        
    @staticmethod 
    def toySimulationParams(): 

        resultsDir = PathDefaults.getOutputDir() + "viroscopy/toy/" 
        graphFile = resultsDir + "ToyEpidemicGraph0"
        targetGraph = HIVGraph.load(graphFile)        
        
        startDate = 0.0        
        endDate = 1000.0
        recordStep = 90
        printStep = 500
        M = 2000
        
        return startDate, endDate, recordStep, printStep, M, targetGraph
        
    @staticmethod 
    def realSimulationParams(): 
        hivReader = HIVGraphReader()
        targetGraph = hivReader.readSimulationHIVGraph()
        
        recordStep = 100 
        printStep = 100
        #Note that 5% of the population is bi 
        M = targetGraph.size * 4
        #This needs to be from 1986 to 2004 
        startDate = CsvConverters.dateConv("01/01/1986")
        endDate = CsvConverters.dateConv("01/01/1989")
        #endDate = CsvConverters.dateConv("31/12/2004")
        
        return float(startDate), float(endDate), recordStep, printStep, M, targetGraph
    
    @staticmethod     
    def simulate(theta, startDate, endDate, recordStep, printStep, M, graphMetrics=None): 
        undirected = True
        graph = HIVGraph(M, undirected)
        logging.debug("Created graph: " + str(graph))
    
        alpha = 2
        zeroVal = 0.9
        p = Util.powerLawProbs(alpha, zeroVal)
        hiddenDegSeq = Util.randomChoice(p, graph.getNumVertices())
    
        rates = HIVRates(graph, hiddenDegSeq)
        model = HIVEpidemicModel(graph, rates, endDate, startDate, metrics=graphMetrics)
        model.setRecordStep(recordStep)
        model.setPrintStep(printStep)
        model.setParams(theta)
        
        logging.debug("Theta = " + str(theta))
        
        return model.simulate(True)
        
    @staticmethod 
    def generateStatistics(graph, startDate, endDate, recordStep): 
        """
        For a given theta, simulate the epidemic, and then return a number of 
        relevant statistics. 
        """
        times = [] 
        removedIndices = []
        
        for t in numpy.arange(startDate, endDate, recordStep): 
            times.append(t)
            removedIndices.append(graph.removedIndsAt(t))

        V = graph.getVertexList().getVertices()
        
        removedArray  = numpy.array([len(x) for x in removedIndices])
        maleArray  = numpy.array([numpy.sum(V[x, HIVVertices.genderIndex]==HIVVertices.male) for x in removedIndices])
        femaleArray = numpy.array([numpy.sum(V[x, HIVVertices.genderIndex]==HIVVertices.female) for x in removedIndices])
        heteroArray = numpy.array([numpy.sum(V[x, HIVVertices.orientationIndex]==HIVVertices.hetero) for x in removedIndices])
        biArray = numpy.array([numpy.sum(V[x, HIVVertices.orientationIndex]==HIVVertices.bi) for x in removedIndices])
        randDetectArray = numpy.array([numpy.sum(V[x, HIVVertices.detectionTypeIndex]==HIVVertices.randomDetect) for x in removedIndices])
        conDetectArray = numpy.array([numpy.sum(V[x, HIVVertices.detectionTypeIndex]==HIVVertices.contactTrace) for x in removedIndices])
        
        vertexArray = numpy.c_[removedArray, maleArray, femaleArray, heteroArray, biArray, randDetectArray, conDetectArray]
        
        graphStats = GraphStatistics()
        removedGraphStats = graphStats.sequenceScalarStats(graph, removedIndices, slowStats=False)
        
        return times, vertexArray, removedGraphStats
