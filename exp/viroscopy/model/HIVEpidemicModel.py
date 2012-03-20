import logging
import numpy
from apgl.graph import *
from apgl.util import *
from exp.viroscopy.model.HIVGraph import HIVGraph
from exp.viroscopy.model.HIVVertices import HIVVertices

"""
This class models an epidemic occuring via sexual contact. 
"""

class HIVEpidemicModel():
    def __init__(self, graph, rates):
        """
        Initialise with a graph of HIVIndividuals.
        """
        Parameter.checkClass(graph, HIVGraph)

        self.graph = graph
        self.rates = rates
        self.T = 100
        self.recordStep = 10
        self.printStep = 1
        self.breakFunc = None
        self.standardiseResults = True

    def setBreakFunction(self, breakFunc):
        """
        Set up a function to break out of the simulation early. Function is called
        on the HIV graph. 
        """
        self.breakFunc = breakFunc

    def setT(self, T):
        Parameter.checkFloat(T, 0.0, float('inf'))
        self.T = T

    def setRecordStep(self, recordStep):
        Parameter.checkInt(recordStep, 0, float('inf'))
        self.recordStep = recordStep 

    def setPrintStep(self, printStep):
        Parameter.checkInt(printStep, 0, float('inf'))
        self.printStep = printStep

    def simulate(self):
        """
        Simulate epidemic propogation until there are no more infectives or
        time T is reached. 
        """
        if self.graph.getNumEdges()!=0:
            raise ValueError("Must start simulation with empty (no edges) graph: " + str(self.graph.getNumEdges()))

        susceptibleSet = self.graph.getSusceptibleSet()
        infectedSet = self.graph.getInfectedSet()
        removedSet = self.graph.getRemovedSet()
        #This is the set of people who are having sexual contact 
        contactSet = susceptibleSet.union(infectedSet)

        infectedList = list(infectedSet)
        removedList = list(removedSet)
        contactList = list(contactSet)

        vList = self.graph.getVertexList()
        t = 0
        times = [t]
        #A list of lists of infected indices 
        infectedIndices = [infectedList]
        removedIndices = [removedList]
        nextStep = self.recordStep
        nextPrintStep = 0
        numContacts = 0 

        logging.debug("Starting simulation")

        #Now, start the simulation
        while t < self.T and len(infectedSet) != 0:
            contactRates = self.rates.contactRates(infectedList, contactList, t)
            contactTracingRates = self.rates.contactTracingRates(infectedList, removedSet, t)
            randomDetectRates = self.rates.randomDetectionRates(infectedList, t)

            assert contactRates.shape == (len(infectedList), len(contactList))
            assert (contactTracingRates == numpy.abs(contactTracingRates)).all()
            assert (randomDetectRates == numpy.abs(randomDetectRates)).all()

            sigmat = PySparseUtils.sum(contactRates)
            muRSt = numpy.sum(randomDetectRates)
            muCTt = numpy.sum(contactTracingRates)
            #rhot = sigmat + muRSt + muCTt

            assert sigmat >= 0
            assert muRSt >= 0
            assert muCTt >= 0 

            sigmaHat = self.rates.upperContactRates(infectedList)
            muHat = self.rates.upperDetectionRates(infectedList)
            rhoHat = sigmaHat + muHat

            assert rhoHat >= 0

            #Now generate random variable which is the advancement in time
            tauPrime = numpy.random.exponential(1/rhoHat)
            t = t + tauPrime
            assert tauPrime >= 0

            #Now compute the probabilities of each event type
            contactProb = sigmat/rhoHat
            detectionRandom = muRSt/rhoHat
            detectionContact = muCTt/rhoHat

            #In some rare cases this can be false due to floating point errors 
            assert sigmat + muRSt + muCTt <= rhoHat + 10**-6, \
                "sigmat=%f, muRSt=%f, muCTt=%f, sigmaHat=%f, muHat=%f" % (sigmat, muRSt, muCTt, sigmaHat, muHat)

            #Compute random variable
            p = numpy.random.rand()

            if p < contactProb:
                (rows, cols) = PySparseUtils.nonzero(contactRates)
                nzContactRates = numpy.zeros(len(rows))
                contactRates.take(nzContactRates, rows, cols)

                eventInd = Util.randomChoice(nzContactRates)[0]
                infectedIndex = infectedList[rows[eventInd]]
                contactIndex = contactList[cols[eventInd]]
                #Note that each time a sexual contact occurs we weight the edge with the time 
                self.rates.contactEvent(infectedIndex, contactIndex, t)
                numContacts += 1 

                #Check if the contact results in an infection
                q = numpy.random.rand()
                if q < self.rates.infectionProbability(infectedIndex, contactIndex, t):
                    vList.setInfected(contactIndex, t)
                    infectedSet.add(contactIndex)
                    susceptibleSet.remove(contactIndex)
            elif p >= contactProb and p < contactProb+detectionRandom:
                eventInd = Util.randomChoice(randomDetectRates)
                newDetectedIndex = infectedList[eventInd]
                self.rates.removeEvent(newDetectedIndex, HIVVertices.randomDetect, t)
                removedSet.add(newDetectedIndex)
                infectedSet.remove(newDetectedIndex)
                contactSet.remove(newDetectedIndex)
            elif p >= contactProb+detectionRandom and p < contactProb+detectionRandom+detectionContact:
                eventInd = Util.randomChoice(contactTracingRates)
                newDetectedIndex = infectedList[eventInd]
                self.rates.removeEvent(newDetectedIndex, HIVVertices.contactTrace, t)
                removedSet.add(newDetectedIndex)
                infectedSet.remove(newDetectedIndex)
                contactSet.remove(newDetectedIndex)

            assert infectedSet.union(removedSet).union(susceptibleSet) == set(range(self.graph.getNumVertices()))
            assert contactSet == infectedSet.union(susceptibleSet)

            infectedList = list(infectedSet)
            removedList = list(removedSet)
            contactList = list(contactSet)

            if t >= nextStep and t <= self.T:
                infectedIndices.append(infectedList)
                removedIndices.append(removedList)
                times.append(t)
                nextStep += self.recordStep

                if self.breakFunc!=None and self.breakFunc(times, infectedIndices, removedIndices, self.graph):
                    logging.debug("Breaking as distance has become too large")
                    break 

            if t>= nextPrintStep or len(infectedSet) == 0:
                logging.debug("t=" + str(t) + " S=" + str(len(susceptibleSet)) + " I=" + str(len(infectedSet)) + " R=" + str(len(removedSet)) + " C=" + str(numContacts) + " E=" + str(self.graph.getNumEdges()))
                nextPrintStep += self.printStep

        logging.debug("Finished simulation at time " + str(t))

        if self.standardiseResults:
            times, infectedIndices, removedIndices = self.findStandardResults(times, infectedIndices, removedIndices)
            logging.debug("times=" + str(times))

        return times, infectedIndices, removedIndices, self.graph


    def findStandardResults(self, times, infectedIndices, removedIndices):
        #Make sure that the results for the simulations are recorded for all times
        idealTimes = list(range(0, int(self.T), self.recordStep))

        newInfectedIndices = []
        newRemovedIndices = []
        times = numpy.array(times)
        
        for i in range(len(idealTimes)):
            idx = (numpy.abs(times-idealTimes[i])).argmin()
            newInfectedIndices.append(infectedIndices[idx])
            newRemovedIndices.append(removedIndices[idx])

        return idealTimes, newInfectedIndices, newRemovedIndices
