
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
from exp.viroscopy.model.HIVModelUtils import HIVModelUtils
from apgl.graph import * 
from apgl.util import Util 

def runModel(theta, endDate=100.0, M=1000): 
    numpy.random.seed(21)
    undirected= True
    recordStep = 10 
    startDate = 0
    alpha = 2
    zeroVal = 0.9
    p = Util.powerLawProbs(alpha, zeroVal)
    graph = HIVGraph(M, undirected)
    hiddenDegSeq = Util.randomChoice(p, graph.getNumVertices())
    logging.debug("MeanTheta=" + str(theta))
    
    rates = HIVRates(graph, hiddenDegSeq)
    model = HIVEpidemicModel(graph, rates, endDate, startDate)
    model.setRecordStep(recordStep)
    model.setParams(theta)
    
    times, infectedIndices, removedIndices, graph = model.simulate(True)            
    
    return times, infectedIndices, removedIndices, graph, model  

@apgl.skipIf(not apgl.checkImport('sppy'), 'No module pysparse')
class  HIVEpidemicModelTest(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(21)
        numpy.set_printoptions(suppress=True, precision=4)
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

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

        times, infectedIndices, removedIndices, graph = self.model.simulate(verboseOut=True)

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

        self.graph.removeAllEdges()

        T = 21.0
        hiddenDegSeq = self.gen.rvs(size=self.graph.getNumVertices())
        rates = HIVRates(self.graph, hiddenDegSeq)
        model = HIVEpidemicModel(self.graph, rates)
        model.setRecordStep(10)
        model.setT(T)

        #Test detection rates
        print("Starting test")

        T = 1000.0
        graph = HIVGraph(M, undirected)
        graph.setRandomInfected(10)
        rates = HIVRates(graph, hiddenDegSeq)
        rates.contactRate = 0
        rates.randDetectRate = 0.1
        model = HIVEpidemicModel(graph, rates)
        model.setT(T)
        times, infectedIndices, removedIndices, graph = model.simulate(verboseOut=True)
        print(times)
        self.assertEquals(len(infectedIndices[0]), 10)
        self.assertEquals(len(removedIndices[0]), 0)
        
        T = 10.0
        graph.removeAllEdges()
        graph = HIVGraph(M, undirected)
        graph.setRandomInfected(10)
        rates = HIVRates(graph, hiddenDegSeq)  
        rates.randDetectRate = 0.0
        model = HIVEpidemicModel(graph, rates)
        model.setT(T)
        times, infectedIndices, removedIndices, graph = model.simulate(verboseOut=True)
        self.assertEquals(len(removedIndices[-1]), 0)
        
        T = 100.0
        graph.removeAllEdges()
        graph = HIVGraph(M, undirected)
        graph.setRandomInfected(10)
        rates = HIVRates(graph, hiddenDegSeq)  
        rates.randDetectRate = 10.0
        model = HIVEpidemicModel(graph, rates)
        model.setT(T)
        times, infectedIndices, removedIndices, graph = model.simulate(verboseOut=True)
        self.assertEquals(len(removedIndices[-1]), 10)
        
        #Test contact tracing 
        T = 1000.0
        graph = HIVGraph(M, undirected)
        graph.setRandomInfected(10)
        rates = HIVRates(graph, hiddenDegSeq)  
        rates.randDetectRate = 0.01
        rates.ctRatePerPerson = 0.5 
        model = HIVEpidemicModel(graph, rates)
        model.setT(T)
        times, infectedIndices, removedIndices, graph = model.simulate(verboseOut=True)
        self.assertTrue((graph.vlist.V[:, HIVVertices.detectionTypeIndex] == HIVVertices.contactTrace).sum() > 0) 
        
        #Test contact rate 
        print("Testing contact rate")
        contactRates = [0.5, 1, 2, 4]     
        numContacts = numpy.zeros(len(contactRates))
        
        for i, contactRate in enumerate(contactRates): 
            T = 100.0
            graph = HIVGraph(M, undirected)
            graph.setRandomInfected(1)
            print(i, graph.vlist.V[graph.getInfectedSet().pop(), :]) 
            rates = HIVRates(graph, hiddenDegSeq)  
            rates.contactRate = contactRate 
            rates.infectProb = 0.0
            model = HIVEpidemicModel(graph, rates)
            model.setT(T)
            times, infectedIndices, removedIndices, graph = model.simulate(verboseOut=True)
            numContacts[i] = model.numContacts

    
        lastN = -1
        
        for i, n in enumerate(numContacts):
            #This is an odd case in which we have a bisexual woman, there are no contacts 
            #since they are not modelled 
            if n != 0: 
                self.assertTrue(n > lastN)    
                
                
        #Test infection rate 
        print("Testing infection probability")
        infectProbs = [0.01, 0.1, 0.2, 0.5]     
        numInfects = numpy.zeros(len(contactRates))
        
        for i, infectProb in enumerate(infectProbs): 
            T = 100.0
            graph = HIVGraph(M, undirected)
            graph.setRandomInfected(10)
            rates = HIVRates(graph, hiddenDegSeq)  
            rates.contactRate = 0.5
            rates.infectProb = infectProb 
            model = HIVEpidemicModel(graph, rates)
            model.setT(T)
            times, infectedIndices, removedIndices, graph = model.simulate(verboseOut=True)
            numInfects[i] = len(graph.getInfectedSet())
        
        for n in numInfects:
            self.assertTrue(n > lastN)   
        
        
        print("Testing contact paramters")
        alphas = [0.01, 0.1, 0.2, 0.5, 1.0]     
        edges = numpy.zeros(len(alphas))
        
        for i, alpha in enumerate(alphas): 
            T = 100.0
            graph = HIVGraph(M, undirected)
            graph.setRandomInfected(1)
            rates = HIVRates(graph, hiddenDegSeq)  
            rates.setAlpha(alpha)
            rates.infectProb = 0 
            model = HIVEpidemicModel(graph, rates)
            model.setT(T)
            times, infectedIndices, removedIndices, graph = model.simulate(verboseOut=True)
            edges[i] = graph.getNumEdges()
            
        print(edges)

    @unittest.skip("")
    def testSimulate2(self):    
        startDate = 0.0 
        endDate = 100.0 
        M = 1000 
        meanTheta, sigmaTheta = HIVModelUtils.estimatedRealTheta()
        
        undirected = True
        graph = HIVGraph(M, undirected)
        
        alpha = 2
        zeroVal = 0.9
        p = Util.powerLawProbs(alpha, zeroVal)
        hiddenDegSeq = Util.randomChoice(p, graph.getNumVertices())
        
        meanTheta[4] = 0.1        
        
        recordStep = 10 
        rates = HIVRates(graph, hiddenDegSeq)
        model = HIVEpidemicModel(graph, rates, endDate, startDate)
        model.setRecordStep(recordStep)
        model.setParams(meanTheta)
        
        initialInfected = graph.getInfectedSet()
        
        times, infectedIndices, removedIndices, graph = model.simulate(True)
        
        #Now test the final graph 
        edges = graph.getAllEdges()
        
        for i, j in edges:
            if graph.vlist.V[i, HIVVertices.genderIndex] == graph.vlist.V[j, HIVVertices.genderIndex] and (graph.vlist.V[i, HIVVertices.orientationIndex] != HIVVertices.bi or graph.vlist.V[j, HIVVertices.orientationIndex] != HIVVertices.bi): 
                self.fail()
                      
        finalRemoved = graph.getRemovedSet()
        
        self.assertEquals(numpy.intersect1d(initialInfected, finalRemoved).shape[0], len(initialInfected))
        
        #Test case where there is no contact  
        meanTheta = numpy.array([100, 0.95, 1, 1, 0, 0, 0, 0, 0, 0, 0], numpy.float)
        
        times, infectedIndices, removedIndices, graph, model = runModel(meanTheta)

        self.assertEquals(len(graph.getInfectedSet()), 100)
        self.assertEquals(len(graph.getRemovedSet()), 0)
        self.assertEquals(graph.getNumEdges(), 0)
        
        heteroContactRate = 0.1
        meanTheta = numpy.array([100, 0.95, 1, 1, 0, 0, heteroContactRate, 0, 0, 0, 0], numpy.float)
        times, infectedIndices, removedIndices, graph, model = runModel(meanTheta)
        
        self.assertEquals(len(graph.getInfectedSet()), 100)
        self.assertEquals(len(graph.getRemovedSet()), 0)
        
        edges = graph.getAllEdges()
        
        for i, j in edges:
            self.assertNotEqual(graph.vlist.V[i, HIVVertices.genderIndex], graph.vlist.V[j, HIVVertices.genderIndex]) 
            
        #Number of conacts = rate*people*time
        infectedSet = graph.getInfectedSet()
        numHetero = (graph.vlist.V[list(infectedSet), HIVVertices.orientationIndex] == HIVVertices.hetero).sum()
        self.assertTrue(abs(numHetero*endDate*heteroContactRate- model.getNumContacts()) < 100)
        
        heteroContactRate = 0.01
        meanTheta = numpy.array([100, 0.95, 1, 1, 0, 0, heteroContactRate, 0, 0, 0, 0], numpy.float)
        times, infectedIndices, removedIndices, graph, model = runModel(meanTheta)
        infectedSet = graph.getInfectedSet()
        numHetero = (graph.vlist.V[list(infectedSet), HIVVertices.orientationIndex] == HIVVertices.hetero).sum()
        self.assertAlmostEqual(numHetero*endDate*heteroContactRate/100, model.getNumContacts()/100.0, 0)      
        
  

    @unittest.skip("")
    def testSimulateInfects(self): 
        #Test varying infection probabilities 
        
        heteroContactRate = 0.1
        manWomanInfectProb = 1.0 
        meanTheta = numpy.array([100, 1, 1, 0, 0, heteroContactRate, manWomanInfectProb], numpy.float)
        times, infectedIndices, removedIndices, graph, model = runModel(meanTheta)
        
        newInfects = numpy.setdiff1d(graph.getInfectedSet(), numpy.array(infectedIndices[0]))
        
        self.assertTrue((graph.vlist.V[newInfects, HIVVertices.genderIndex] == HIVVertices.female).all())
        
        manWomanInfectProb = 0.1
        meanTheta = numpy.array([100, 0.95, 1, 1, 0, 0, heteroContactRate, 0, 0, manWomanInfectProb, 0], numpy.float)
        times, infectedIndices, removedIndices, graph, model = runModel(meanTheta)
        newInfects2 = numpy.setdiff1d(graph.getInfectedSet(), numpy.array(infectedIndices[0]))
        
        self.assertTrue((graph.vlist.V[newInfects2, HIVVertices.genderIndex] == HIVVertices.female).all())
        self.assertTrue(newInfects.shape[0] > newInfects2.shape[0])
        
        
        #Now only women infect 
        heteroContactRate = 0.1
        womanManInfectProb = 1.0 
        meanTheta = numpy.array([100, 0.95, 1, 1, 0, 0, heteroContactRate, 0, womanManInfectProb, 0, 0], numpy.float)
        times, infectedIndices, removedIndices, graph, model = runModel(meanTheta)
        
        newInfects = numpy.setdiff1d(graph.getInfectedSet(), numpy.array(infectedIndices[0]))
        
        self.assertTrue((graph.vlist.V[newInfects, HIVVertices.genderIndex] == HIVVertices.male).all())
        
        womanManInfectProb = 0.1
        meanTheta = numpy.array([100, 0.95, 1, 1, 0, 0, heteroContactRate, 0, womanManInfectProb, 0, 0], numpy.float)
        times, infectedIndices, removedIndices, graph, model = runModel(meanTheta)
        newInfects2 = numpy.setdiff1d(graph.getInfectedSet(), numpy.array(infectedIndices[0]))
        
        self.assertTrue((graph.vlist.V[newInfects2, HIVVertices.genderIndex] == HIVVertices.male).all())
        self.assertTrue(newInfects.shape[0] > newInfects2.shape[0])
  
    @unittest.skip("")
    def testSimulateDetects(self): 
        heteroContactRate = 0.05
        endDate = 100
        
        randDetectRate = 0
        meanTheta = numpy.array([100, 0.95, 1, 1, randDetectRate, 0, heteroContactRate, 0, 0, 0, 0], numpy.float)
        times, infectedIndices, removedIndices, graph, model = runModel(meanTheta)
        detectedSet = graph.getRemovedSet()    
        self.assertEquals(len(detectedSet), 0)
        
        heteroContactRate = 0.0
        randDetectRate = 0.01
        meanTheta = numpy.array([100, 0.95, 1, 1, randDetectRate, 0, heteroContactRate, 0, 0, 0, 0], numpy.float)
        times, infectedIndices, removedIndices, graph, model = runModel(meanTheta)
        detectedSet = graph.getRemovedSet()
        
        self.assertTrue(len(detectedSet) < 100*randDetectRate*endDate)
        
        randDetectRate = 0.005
        meanTheta = numpy.array([100, 0.95, 1, 1, randDetectRate, 0, heteroContactRate, 0, 0, 0, 0], numpy.float)
        times, infectedIndices, removedIndices, graph, model = runModel(meanTheta)
        detectedSet2 = graph.getRemovedSet()
    
        print(len(detectedSet), len(detectedSet2))
        self.assertTrue(abs(len(detectedSet)*2 - len(detectedSet2))<15)   
        
        removedGraph = graph.subgraph(list(graph.getRemovedSet())) 
        edges = removedGraph.getAllEdges()        
        
        for edge in edges: 
            i, j = edge
            self.assertEquals(removedGraph.vlist.V[i, HIVVertices.detectionTimeIndex]. HIVVertices.randomDetect)
            self.assertEquals(removedGraph.vlist.V[j, HIVVertices.detectionTimeIndex]. HIVVertices.randomDetect)
               
        #Test contact tracing 
        randDetectRate = 0
        setCtRatePerPerson = 0.1
        meanTheta = numpy.array([100, 0.95, 1, 1, randDetectRate, setCtRatePerPerson, heteroContactRate, 0, 0, 0, 0], numpy.float)
        times, infectedIndices, removedIndices, graph, model = runModel(meanTheta)
        detectedSet = graph.getRemovedSet()   
        self.assertEquals(len(detectedSet), 0)
        
        randDetectRate = 0.001
        setCtRatePerPerson = 0.1
        meanTheta = numpy.array([100, 0.95, 1, 1, randDetectRate, setCtRatePerPerson, heteroContactRate, 0, 0, 0, 0], numpy.float)
        times, infectedIndices, removedIndices, graph, model = runModel(meanTheta, endDate=500.0)
        detectedSet = graph.getRemovedSet()   
              
        removedGraph = graph.subgraph(list(graph.getRemovedSet())) 
        edges = removedGraph.getAllEdges()
        
        for i in removedGraph.getAllVertexIds(): 
            if removedGraph.vlist.V[i, HIVVertices.detectionTypeIndex] == HIVVertices.contactTrace: 
                self.assertTrue(removedGraph.vlist.V[i, HIVVertices.detectionTimeIndex] >= 180)

        
    
    @unittest.skip("")
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

