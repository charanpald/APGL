

"""
Keep some default parameters for the epidemic model. 
"""
import numpy 
import logging 
from apgl.util import Util 
from apgl.graph.GraphStatistics import GraphStatistics
from exp.viroscopy.model.HIVGraph import HIVGraph
from exp.viroscopy.model.HIVEpidemicModel import HIVEpidemicModel
from exp.viroscopy.model.HIVRates import HIVRates
from exp.viroscopy.model.HIVVertices import HIVVertices

class HIVModelUtils(object):
    def __init__(self): 
        pass 
    
    @staticmethod
    def defaultTheta(): 
        theta = numpy.array([50, 1.0, 0.5, 1.0/800, 0.01, 0.05, 0.1, 38.0/1000, 30.0/1000, 170.0/1000])
        return theta 
        
    @staticmethod 
    def defaultSimulationParams(): 
        T = 1000.0
        recordStep = 90
        printStep = 500
        M = 2000
        
        return T, recordStep, printStep, M 
    
    @staticmethod     
    def defaultSimulate(theta): 
        T, recordStep, printStep, M = HIVModelUtils.defaultSimulationParams()
        undirected = True
        graph = HIVGraph(M, undirected)
        logging.debug("Created graph: " + str(graph))
    
        alpha = 2
        zeroVal = 0.9
        p = Util.powerLawProbs(alpha, zeroVal)
        hiddenDegSeq = Util.randomChoice(p, graph.getNumVertices())
    
        rates = HIVRates(graph, hiddenDegSeq)
        model = HIVEpidemicModel(graph, rates)
        model.setT(T)
        model.setRecordStep(recordStep)
        model.setPrintStep(printStep)
        model.setParams(theta)
        
        logging.debug("Theta = " + str(theta))
        
        return model.simulate(True)
        
    @staticmethod 
    def generateStatistics(theta): 
        """
        For a given theta, simulate the epidemic, and then return a number of 
        relevant statistics. 
        """
        T, recordStep, printStep, M = HIVModelUtils.defaultSimulationParams()

        times, infectedIndices, removedIndices, graph = HIVModelUtils.defaultSimulate(theta)
        V = graph.getVertexList().getVertices()
        
        infectedArray = numpy.array([len(x) for x in infectedIndices])
        removedArray  = numpy.array([len(x) for x in removedIndices])
        maleArray  = numpy.array([numpy.sum(V[x, HIVVertices.genderIndex]==HIVVertices.male) for x in removedIndices])
        femaleArray = numpy.array([numpy.sum(V[x, HIVVertices.genderIndex]==HIVVertices.female) for x in removedIndices])
        heteroArray = numpy.array([numpy.sum(V[x, HIVVertices.orientationIndex]==HIVVertices.hetero) for x in removedIndices])
        biArray = numpy.array([numpy.sum(V[x, HIVVertices.orientationIndex]==HIVVertices.bi) for x in removedIndices])
        
        vertexArray = numpy.c_[infectedArray, removedArray, maleArray, femaleArray, heteroArray, biArray]
        
        graphStats = GraphStatistics()
        infectedGraphStats = graphStats.sequenceScalarStats(graph, infectedIndices, slowStats=False)
        removedGraphStats = graphStats.sequenceScalarStats(graph, removedIndices, slowStats=False)
        
        return times, vertexArray, infectedGraphStats, removedGraphStats

        
            
    
            
