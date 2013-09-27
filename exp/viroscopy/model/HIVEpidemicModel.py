import logging
import numpy
from apgl.graph import *
from apgl.util import *
from exp.viroscopy.model.HIVGraph import HIVGraph
from exp.viroscopy.model.HIVVertices import HIVVertices

class HIVEpidemicModel():
    def __init__(self, graph, rates, T=100.0, T0=0.0, metrics=None):
        """
        This class models an epidemic occuring via sexual contact. We create an 
        epidemic model with a HIVGraph and a class which models the rate of 
        certain events in the model. 
        
        :param graph: Initial HIVGraph to use for modelling 
        
        :param rates: A class modelling the event rates in the model 
        
        :param T0: This is the starting time of the simulation 
        
        :param metrics: A graph metrics object 
        """
        Parameter.checkClass(graph, HIVGraph)

        self.graph = graph
        self.graph.endEventTime = T0
        self.rates = rates
        self.setT(T)
        self.breakFunc = None
        self.T0 = T0
        self.setRecordStep((self.T-self.T0)/10.0)
        self.metrics = metrics 

    def setT(self, T):
        """
        Set the maximum time of the simulation. 
        """
        Parameter.checkFloat(T, 0.0, float('inf'))
        self.T = T
        
    def setT0(self, T0):
        """
        Set the start time of the simulation. 
        """
        Parameter.checkFloat(T0, 0.0, float('inf'))
        self.T0 = T0

    def setRecordStep(self, recordStep):
        """
        Set thetime interval in order to record statistics over the model. 
        """
        if abs((self.T-self.T0) % recordStep) > 10**-6:
            print((self.T-self.T0) % recordStep)
            raise ValueError("Record Step must divide exactly into T-T0")
        self.recordStep = recordStep 

        
    def setParams(self, theta): 
        """
        This is used to set the parameters of the intial state of this model 
        in conjunction with ABC model selection. 
        
        :param theta: An array containing parameter values 
        :type theta: `numpy.ndarray`
        """
        if theta.shape[0] != 7: 
            raise ValueError("Theta should be of length 7")
        
        self.graph.setRandomInfected(int(theta[0]))
        self.rates.setAlpha(theta[1])
        self.rates.setRandDetectRate(theta[2])
        self.rates.setCtRatePerPerson(theta[3])
        self.rates.setMaxDetects(int(theta[4]))
        self.rates.setContactRate(theta[5])
        self.rates.setInfectProb(theta[6])
        
    #@profile
    def simulate(self, verboseOut=False):
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

        t = self.T0
        times = [t]
        #A list of lists of infected indices 
        infectedIndices = [infectedList]
        removedIndices = [removedList]
        nextStep = t + self.recordStep
        numContacts = 0 

        logging.debug("Starting simulation at time " + str(t) + " with graph of size " + str(self.graph.size))

        #Now, start the simulation
        while t < self.T and len(infectedSet) != 0:
            contactInds, contactRates = self.rates.contactRates(infectedList, contactList, t)
            contactTracingRates = self.rates.contactTracingRates(infectedList, removedSet, t)
            randomDetectRates = self.rates.randomDetectionRates(infectedList, len(susceptibleSet) + len(infectedSet))

            #assert contactRates.shape == (len(infectedList), len(contactList))
            assert (contactTracingRates == numpy.abs(contactTracingRates)).all()
            assert (randomDetectRates == numpy.abs(randomDetectRates)).all()
            
            assert (contactTracingRates!=0).sum() <= self.rates.maxDetects 
            assert (randomDetectRates!=0).sum() <= self.rates.maxDetects 

            sigmat = contactRates.sum()
            muRSt = numpy.sum(randomDetectRates)
            muCTt = numpy.sum(contactTracingRates)
            #rhot = sigmat + muRSt + muCTt
            
            
            #print(randomDetectRates)

            assert sigmat >= 0
            assert muRSt >= 0
            assert muCTt >= 0 

            sigmaHat = self.rates.upperContactRates(infectedList)
            muHat = self.rates.upperDetectionRates(infectedList, len(susceptibleSet) + len(infectedSet))
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
                eventInd = Util.randomChoice(contactRates)[0]
                infectedIndex = infectedList[eventInd]
                contactIndex = contactInds[eventInd]
                #Note that each time a sexual contact occurs we weight the edge with the time 
                self.rates.contactEvent(infectedIndex, contactIndex, t)
                numContacts += 1 

                #Check if the contact results in an infection
                q = numpy.random.rand()
                if q < self.rates.infectionProbability(infectedIndex, contactIndex, t):
                    self.graph.vlist.setInfected(contactIndex, t)
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
            
            self.graph.endEventTime = t

            assert infectedSet.union(removedSet).union(susceptibleSet) == set(range(self.graph.getNumVertices()))
            assert contactSet == infectedSet.union(susceptibleSet)

            infectedList = list(infectedSet)
            removedList = list(removedSet)
            contactList = list(contactSet)

            if t >= nextStep:
                logging.debug("t-T0=" + str(t-self.T0) + " S=" + str(len(susceptibleSet)) + " I=" + str(len(infectedSet)) + " R=" + str(len(removedSet)) + " C=" + str(numContacts) + " E=" + str(self.graph.getNumEdges()))
                
                infectedIndices.append(infectedList)
                removedIndices.append(removedList)
                times.append(t)
                nextStep += self.recordStep

                if self.metrics != None: 
                    self.metrics.addGraph(self.graph)
                
                    if self.metrics.shouldBreak(): 
                        logging.debug("Breaking as distance has become too large")
                        break 

        logging.debug("Finished simulation at time " + str(t) + " for a total time of " + str(t-self.T0))
        
        infectedIndices.append(infectedList)
        removedIndices.append(removedList)
        times.append(t)
        
        self.numContacts = numContacts 
            
        if verboseOut: 
            return times, infectedIndices, removedIndices, self.graph
        else: 
            return self.graph

    def distance(self): 
        logging.debug("Distance is " + str(self.metrics.distance()) + ", and final event on graph occured at time " + str(self.graph.endTime() - self.T0))
        return self.metrics.distance() 
        
    def getNumContacts(self): 
        return self.numContacts
