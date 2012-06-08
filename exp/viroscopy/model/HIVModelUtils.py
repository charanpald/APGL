

"""
Keep some default parameters for the epidemic model. 
"""
import numpy 
import logging 
from apgl.util import Util 
from exp.viroscopy.model.HIVGraph import HIVGraph
from exp.viroscopy.model.HIVEpidemicModel import HIVEpidemicModel
from exp.viroscopy.model.HIVRates import HIVRates

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
            
    
            
