import logging
import sys
import numpy
import scipy
import scipy.stats 
from exp.viroscopy.model.HIVGraph import HIVGraph
from exp.viroscopy.model.HIVRates import HIVRates
from exp.viroscopy.model.HIVVertices import HIVVertices
from apgl.graph import *
from apgl.util import *

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
numpy.random.seed(21)

class HIVRatesProfile():
    def __init__(self):
        #Total number of people in population
        self.M = 10000
        numInitialInfected = 5

        #The graph is one in which edges represent a contact
        undirected = True
        self.graph = HIVGraph(self.M, undirected)

        for i in range(self.M):
            vertex = self.graph.getVertex(i)

            #Set the infection time of a number of individuals to 0
            if i < numInitialInfected:
                vertex[HIVVertices.stateIndex] = HIVVertices.infected

        outputDirectory = PathDefaults.getOutputDir()
        directory = outputDirectory + "test/"
        self.profileFileName = directory + "profile.cprof"


    def profileContactRate(self):
        susceptibleList = list(range(1, self.graph.getNumVertices()))
        t = 10

        s = 3
        gen = scipy.stats.zipf(s)
        hiddenDegSeq = gen.rvs(size=self.graph.getNumVertices())
        rates = HIVRates(self.graph, hiddenDegSeq)

        numContactEvents = 5000
        for i in range(numContactEvents):
            vertexInd1 = numpy.random.randint(0, self.graph.getNumVertices())
            vertexInd2 = numpy.random.randint(0, self.graph.getNumVertices())
            rates.contactEvent(vertexInd1, vertexInd2, 5)

        print((self.graph.getNumEdges()))

        infectedList = range(0, 100)
        contactList = range(100, self.M)
        t = 10

        def runContactRates():
            for i in range(100):
                rates.contactRates(infectedList, contactList, t)

        ProfileUtils.profile('runContactRates()', globals(), locals())


    def profileInfectionProbability(self):
        s = 3
        gen = scipy.stats.zipf(s)
        hiddenDegSeq = gen.rvs(size=self.graph.getNumVertices())
        rates = HIVRates(self.graph, hiddenDegSeq)
        t = 5

        #Getting vertices and checking parameters takes the most time 
        def runInfectionProbs():
            for i in range(10000):
                vertexInd1 = numpy.random.randint(0, self.graph.getNumVertices())
                vertexInd2 = numpy.random.randint(0, self.graph.getNumVertices())
                rates.infectionProbability(vertexInd1, vertexInd2, t)

        ProfileUtils.profile('runInfectionProbs()', globals(), locals())

    def profileContactTracingRate(self):
        s = 3
        gen = scipy.stats.zipf(s)
        hiddenDegSeq = gen.rvs(size=self.graph.getNumVertices())
        rates = HIVRates(self.graph, hiddenDegSeq)

        #Create a network of sexual contacts 
        numContactEvents = 10000
        for i in range(numContactEvents):
            vertexInd1 = numpy.random.randint(0, self.graph.getNumVertices())
            vertexInd2 = numpy.random.randint(0, self.graph.getNumVertices())
            rates.contactEvent(vertexInd1, vertexInd2, 5)

        print((self.graph))
        print((self.graph.degreeDistribution()))

        #Choose some individuals as being infected and then detected 
        p = 0.3
        q = 0.4
        for i in range(self.graph.getNumVertices()):
            if numpy.random.rand() < p and not self.graph.getVertex(i)[HIVVertices.stateIndex] == HIVVertices.infected:
                self.graph.getVertexList().setInfected(i, 5.0)

                if numpy.random.rand() < q:
                    self.graph.getVertexList().setDetected(i, 6.0, HIVVertices.randomDetect)

        infectedSet = self.graph.getInfectedSet()
        print((len(infectedSet)))
        print((len(self.graph.getRemovedSet())))

        removedSet = self.graph.getRemovedSet()

        t = 200
        def runContactTracingRate():
            for j in range(2000):
                rates.contactTracingRates(list(infectedSet), removedSet, t)

        ProfileUtils.profile('runContactTracingRate()', globals(), locals())

profiler = HIVRatesProfile()
#profiler.profileInfectionProbability()
#profiler.profileContactTracingRate()
profiler.profileContactRate()

