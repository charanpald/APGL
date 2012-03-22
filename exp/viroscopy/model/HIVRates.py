import numpy
from pysparse import spmatrix
from exp.viroscopy.model.HIVVertices import HIVVertices
from apgl.util.Util import *

"""
Model the contact rate of an infected individual and other susceptibles.
"""
class HIVRates():
    def __init__(self, graph, hiddenDegSeq):
        """
        Graph is the initial HIV graph and hiddenDegSeq is the initial degree
        sequence. 
        """
        #Trade off between choosing new contact from known degrees and random is p/q
        self.p = 1
        self.q = 3

        self.graph = graph
        self.vList = self.graph.getVertexList()
        self.V = self.vList.getVertices()

        #First figure out the different types of people in the graph
        self.femaleInds = self.V[:, HIVVertices.genderIndex]==HIVVertices.female
        self.maleInds = self.V[:, HIVVertices.genderIndex]==HIVVertices.male
        self.biMaleInds = numpy.logical_and(self.maleInds, self.V[:, HIVVertices.orientationIndex]==HIVVertices.bi)
        self.heteroMaleInds = numpy.logical_and(self.maleInds, self.V[:, HIVVertices.orientationIndex]==HIVVertices.hetero)

        #We need to store degree sequences for 3 types
        self.expandedDegSeqFemales = Util.expandIntArray(graph.outDegreeSequence()[self.femaleInds]*(self.q-self.p))
        self.expandedDegSeqFemales = numpy.append(self.expandedDegSeqFemales, Util.expandIntArray(hiddenDegSeq[self.femaleInds]*self.p))
        self.expandedDegSeqMales = Util.expandIntArray(graph.outDegreeSequence()[self.maleInds]*(self.q-self.p))
        self.expandedDegSeqMales = numpy.append(self.expandedDegSeqMales, Util.expandIntArray(hiddenDegSeq[self.maleInds]*self.p))
        self.expandedDegSeqBiMales = Util.expandIntArray(graph.outDegreeSequence()[self.biMaleInds]*(self.q-self.p))
        self.expandedDegSeqBiMales = numpy.append(self.expandedDegSeqBiMales, Util.expandIntArray(hiddenDegSeq[self.biMaleInds]*self.p))

        self.hiddenDegSeq = hiddenDegSeq
        self.degSequence = graph.outDegreeSequence() 

        #Parameters for sexual contact
        self.alpha = 2.0
        self.newContactChance = 5.0
        self.heteroContactRate = 0.1
        self.biContactRate = 0.5

        #Infection probabilities are from wikipedia
        self.womanManInfectProb = 38.0/10000
        self.manWomanInfectProb = 30.0/10000
        #Taken from "Per-contact probability of HIV transmission in homosexual men in Sydney in the era of HAART" 
        self.manBiInfectProb = 100.0/10000

        #Random detection
        self.randDetectRate = 1/720.0

        #Contact tracing parameters 
        self.ctRatePerPerson = 0.3
        #The start and end time of contact tracing
        self.ctStartTime = 180
        self.ctEndTime = 1825

        #contactTimesArr is an array of the index of the last sexual contact and contact time
        #Given time starts at zero we set last contact to -inf 
        self.contactTimesArr = numpy.ones((graph.getNumVertices(), 2))*-float('inf')
        self.neighboursList = []
        self.detectedNeighboursList = [] 

        for i in range(graph.getNumVertices()):
            self.neighboursList.append(graph.neighbours(i))
            self.detectedNeighboursList.append(numpy.array([], numpy.int))

    def setAlpha(self, alpha):
        Parameter.checkFloat(alpha, 0.0, float('inf'))
        self.alpha = alpha

    def setNewContactChance(self, newContactChance):
        Parameter.checkFloat(newContactChance, 0.0, float('inf'))
        self.newContactChance = newContactChance

    def setHeteroContactRate(self, heteroContactRate):
        Parameter.checkFloat(heteroContactRate, 0.0, float('inf'))
        self.heteroContactRate = heteroContactRate

    def setBiContactRate(self, biContactRate):
        Parameter.checkFloat(biContactRate, 0.0, float('inf'))
        self.biContactRate = biContactRate

    def setRandDetectRate(self, randDetectRate):
        Parameter.checkFloat(randDetectRate, 0.0, float('inf'))
        self.randDetectRate = randDetectRate

    def setCtRatePerPerson(self, ctRatePerPerson):
        Parameter.checkFloat(ctRatePerPerson, 0.0, float('inf'))
        self.ctRatePerPerson = ctRatePerPerson

    def setWomanManInfectProb(self, womanManInfectProb):
        Parameter.checkFloat(womanManInfectProb, 0.0, 1.0)
        self.womanManInfectProb = womanManInfectProb

    def setManWomanInfectProb(self, manWomanInfectProb):
        Parameter.checkFloat(manWomanInfectProb, 0.0, 1.0)
        self.manWomanInfectProb = manWomanInfectProb

    def setManBiInfectProb(self, manBiInfectProb):
        Parameter.checkFloat(manBiInfectProb, 0.0, 1.0)
        self.manBiInfectProb = manBiInfectProb

    def contactEvent(self, vertexInd1, vertexInd2, t):
        """
        Indicates a sexual contact event between two vertices. 
        """
        if vertexInd1 == vertexInd2:
            return 
        if self.graph.getEdge(vertexInd1, vertexInd2) == None:
            for i in [vertexInd1, vertexInd2]:
                self.degSequence[i] += 1 
                if self.V[i, HIVVertices.genderIndex] == HIVVertices.male:
                    self.expandedDegSeqMales = numpy.append(self.expandedDegSeqMales, numpy.repeat(numpy.array([i]), self.q-self.p))

                    if self.V[i, HIVVertices.orientationIndex]==HIVVertices.bi:
                        self.expandedDegSeqBiMales = numpy.append(self.expandedDegSeqBiMales, numpy.repeat(numpy.array([i]), self.q-self.p))
                else:
                    self.expandedDegSeqFemales = numpy.append(self.expandedDegSeqFemales, numpy.repeat(numpy.array([i]), self.q-self.p))
            
        self.graph.addEdge(vertexInd1, vertexInd2, t)
        self.neighboursList[vertexInd1] = self.graph.neighbours(vertexInd1)
        self.neighboursList[vertexInd2] = self.graph.neighbours(vertexInd2)
        self.contactTimesArr[vertexInd1, :] = numpy.array([vertexInd2, t])
        self.contactTimesArr[vertexInd2, :] = numpy.array([vertexInd1, t])

        assert (self.degSequence == self.graph.outDegreeSequence()).all()

        #assert self.expandedDegSeq.shape[0] == numpy.sum(self.graph.outDegreeSequence()) + self.graph.getNumVertices(), \
        #    "expandedDegSequence.shape[0]=%d, sum(degreeSequence)=%d" % (self.expandedDegSeq.shape[0], self.graph.getNumVertices()+numpy.sum(self.graph.outDegreeSequence()))

    def removeEvent(self, vertexInd, detectionMethod, t):
        """
        We just remove the vertex from expandedDegSeq and expandedHiddenDegSeq
        """
        self.vList.setDetected(vertexInd, t, detectionMethod)

        #Update set of detected neighbours
        for neighbour in self.neighboursList[vertexInd]:
            self.detectedNeighboursList[neighbour] = numpy.append(self.detectedNeighboursList[neighbour], numpy.array([vertexInd])) 

            #Check these are correct
            assert ((numpy.sort(self.detectedNeighboursList[neighbour]) == self.graph.detectedNeighbours(neighbour)).all()), \
                "%s and %s" % (numpy.sort(self.detectedNeighboursList[neighbour]),  self.graph.detectedNeighbours(neighbour))

    def upperContactRates(self, infectedList):
        """
        This is an upper bound on the contact rates not dependent on the time. We
        just return a vector of upper bounds for each infected
        """

        #Note a heterosexual can only have a heterosexual rate but bisexual can have either 
        contactRates = numpy.zeros(len(infectedList))
        contactRates += (self.V[infectedList, HIVVertices.orientationIndex]==HIVVertices.bi)*max(self.biContactRate, self.heteroContactRate) 
        contactRates += (self.V[infectedList, HIVVertices.orientationIndex]==HIVVertices.hetero)*self.heteroContactRate

        return numpy.sum(contactRates)
        
    def upperDetectionRates(self, infectedList):
        """
        An upper bound on the detection rates indepedent of time.This is just the
        random detection rate plus the ctRate per person for each detected neighbour. 
        """
        detectionRates = numpy.ones(len(infectedList))*self.randDetectRate

        for i in range(len(infectedList)):
            ind = infectedList[i]
            detectionRates[i] += self.detectedNeighboursList[ind].shape[0]*self.ctRatePerPerson

        return numpy.sum(detectionRates)

    #@profile 
    def contactRates(self, infectedList, contactList, t):
        """
        Work out contact rates between all infected and all other individuals. The
        set of infected is given in infectedList, and the set of contacts is given
        in contactList. Here we compute rates between an infected and all others
        and then restrict to the people given in contactList. 
        """
        if len(infectedList) == 0:
            return numpy.array([])

        contactRates = spmatrix.ll_mat(len(infectedList), self.graph.getNumVertices())

        infectedV = self.V[infectedList, :]
        maleInfectInds = infectedV[:, HIVVertices.genderIndex]==HIVVertices.male
        femaleInfectInds = numpy.logical_not(maleInfectInds)
        biInfectInds = infectedV[:, HIVVertices.orientationIndex]==HIVVertices.bi
        heteroInfectInds = numpy.logical_not(biInfectInds)
        maleHeteroInds = numpy.logical_and(maleInfectInds, heteroInfectInds)
        maleBiInds = numpy.logical_and(maleInfectInds, biInfectInds)

        #possibleContacts has as its first column the previous contacts.
        #A set of new contacts based on the degree sequence
        #These contacts can be any one of the other contacts that are non-removed
        numPossibleContacts = 2 
        possibleContacts = numpy.zeros((len(infectedList), numPossibleContacts), numpy.int)
        possibleContactWeights = numpy.zeros((len(infectedList), numPossibleContacts))
        
        possibleContacts[:, 0] = self.contactTimesArr[infectedList, 0]

        totalDegSequence = numpy.array(self.hiddenDegSeq*self.p + self.degSequence*(self.q-self.p), numpy.float)
        assert (self.expandedDegSeqFemales.shape[0] + self.expandedDegSeqMales.shape[0]) == totalDegSequence.sum(), \
            "totalDegSequence.sum()=%d, expanded=%d" % (totalDegSequence.sum(), self.expandedDegSeqFemales.shape[0] + self.expandedDegSeqMales.shape[0])

        #Note that we may get duplicates for possible contacts since we don't check for it        
        edsInds = numpy.random.randint(0, self.expandedDegSeqFemales.shape[0], maleHeteroInds.sum())
        contactInds = self.expandedDegSeqFemales[edsInds]
        possibleContacts[maleHeteroInds, 1] = contactInds
        possibleContactWeights[maleHeteroInds, 1] = totalDegSequence[contactInds]/self.expandedDegSeqFemales.shape[0]

        edsInds = numpy.random.randint(0, self.expandedDegSeqMales.shape[0], femaleInfectInds.sum())
        contactInds = self.expandedDegSeqMales[edsInds]
        possibleContacts[femaleInfectInds, 1] = contactInds
        possibleContactWeights[femaleInfectInds, 1] = totalDegSequence[contactInds]/self.expandedDegSeqMales.shape[0]

        if self.expandedDegSeqBiMales.shape[0] != 0:
            edsInds = numpy.random.randint(0, self.expandedDegSeqBiMales.shape[0], maleBiInds.sum())
            contactInds = self.expandedDegSeqBiMales[edsInds]
            possibleContacts[maleBiInds, 1] = contactInds
            possibleContactWeights[maleBiInds, 1] = totalDegSequence[contactInds]/self.expandedDegSeqBiMales.shape[0]
        else:
            edsInds = numpy.array([], numpy.int)

        #Now compute weights for all
        
        #If someone has no previous contact, make sure the probability is zero
        #Could exclude zero probability events from randomChoice if that speed things up
        epsilon = 0.0001

        #If last contact time is infinity then weight should be zero 
        possibleContactWeights[:, 0] = (epsilon + t - self.contactTimesArr[infectedList, 1])**-self.alpha
        possibleContactWeights[:, 1] *= self.newContactChance**-self.alpha

        assert (possibleContactWeights >= numpy.zeros((len(infectedList), numPossibleContacts))).all()

        contactInds = Util.random2Choice(possibleContactWeights).ravel()
        contacts = possibleContacts[(numpy.arange(possibleContacts.shape[0]), contactInds)]
        contactsV = self.V[contacts, :]

        #We compute sexual contact rates between infected and everyone else
        equalGender = infectedV[:, HIVVertices.genderIndex]==contactsV[:, HIVVertices.genderIndex]
        hInds = numpy.nonzero(numpy.logical_not(equalGender))[0]
        contactRates.put(self.heteroContactRate, hInds, contacts[hInds])

        bInds = numpy.nonzero(numpy.logical_and(numpy.logical_and(equalGender, contactsV[:, HIVVertices.orientationIndex]==HIVVertices.bi), biInfectInds))[0]
        contactRates.put(self.biContactRate, bInds, contacts[bInds])

        #Make sure people can't have contact with themselves
        contactRates.put(0, range(len(infectedList)), infectedList)

        #Check there is at most 1 element per row
        #for i in range(contactRates.shape[0]):
        #    assert contactRates[i, :].nnz <= 1
        
        #contactRates.T.to_csr()

        return contactRates[:, contactList]

    """
    Compute the infection probability between an infected and susceptible.
    """
    def infectionProbability(self, vertexInd1, vertexInd2, t):
        """
        This returns the infection probability of an infected person vertexInd1
        and a non-removed vertexInd2. 
        """
        vertex1 = self.V[vertexInd1, :]
        vertex2 = self.V[vertexInd2, :]

        if vertex1[HIVVertices.stateIndex]!=HIVVertices.infected or vertex2[HIVVertices.stateIndex]!=HIVVertices.susceptible:
            return 0.0

        if vertex1[HIVVertices.genderIndex] == HIVVertices.female and vertex2[HIVVertices.genderIndex] == HIVVertices.male:
            return self.womanManInfectProb
        elif vertex1[HIVVertices.genderIndex] == HIVVertices.male and vertex2[HIVVertices.genderIndex] == HIVVertices.female:
            return self.manWomanInfectProb
        elif vertex1[HIVVertices.genderIndex] == HIVVertices.male and vertex1[HIVVertices.orientationIndex]==HIVVertices.bi and vertex2[HIVVertices.genderIndex] == HIVVertices.male and vertex2[HIVVertices.orientationIndex]==HIVVertices.bi:
            return self.manBiInfectProb
        else:
            #Corresponds to 2 bisexual women 
            return 0.0 

    """
    Compute the detection rate of an infected which depends on the entire population.
    """
    def randomDetectionRates(self, infectedList, t):
        detectionRates = numpy.ones(len(infectedList))*self.randDetectRate

        return detectionRates

    def contactTracingRates(self, infectedList, removedSet, t):
        """
        Compute the contact tracing detection rate of a list of infected individuals.
        """
        assert set(infectedList).intersection(removedSet) == set([])

        ctRates = numpy.zeros(len(infectedList))
        #A neigbour that was detected at ctStartDate or earlier can result in detection
        ctStartDate = t - self.ctStartTime
        #A detected neigbour that happened later than cdEndDate can result in detection
        cdEndDate = t - self.ctEndTime

        infectedArray = numpy.array(infectedList)
        infectedArrInds = numpy.argsort(infectedArray)
        infectedArray = infectedArray[infectedArrInds]

        removeIndices = numpy.array(list(removedSet), numpy.int)
        underCT = numpy.zeros(self.graph.getNumVertices(), numpy.bool)
        underCT[removeIndices] = numpy.logical_and(self.V[removeIndices, HIVVertices.detectionTimeIndex] >= cdEndDate, self.V[removeIndices, HIVVertices.detectionTimeIndex] <= ctStartDate)

        if len(infectedList) < len(removedSet):
            #Possibly store set of detected neighbours
            for i in range(len(infectedList)):
                vertexInd = infectedList[i]
                detectedNeighbours = self.detectedNeighboursList[vertexInd]

                for ind in detectedNeighbours:
                    if underCT[ind]:
                        ctRates[i] += self.ctRatePerPerson
                #This is slower for some reason
                #ctRates[i] = numpy.sum(underCT[neighbours]) * self.ctRatePerPerson
        else:
            for vertexInd in removedSet:
                if underCT[vertexInd]:
                    neighbours = self.neighboursList[vertexInd]

                    for ind in neighbours:
                        if self.V[ind, HIVVertices.stateIndex] == HIVVertices.infected:
                            i = numpy.searchsorted(infectedArray, ind)
                            ctRates[infectedArrInds[i]] += self.ctRatePerPerson

        assert (ctRates >= numpy.zeros(len(infectedList))).all()

        return ctRates