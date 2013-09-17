import numpy
from exp.viroscopy.model.HIVVertices import HIVVertices
from apgl.util.Util import *
import numpy.testing as nptst

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

        #First figure out the different types of people in the graph
        self.femaleInds = self.graph.vlist.V[:, HIVVertices.genderIndex]==HIVVertices.female
        self.maleInds = self.graph.vlist.V[:, HIVVertices.genderIndex]==HIVVertices.male
        self.biMaleInds = numpy.logical_and(self.maleInds, self.graph.vlist.V[:, HIVVertices.orientationIndex]==HIVVertices.bi)
        self.heteroMaleInds = numpy.logical_and(self.maleInds, self.graph.vlist.V[:, HIVVertices.orientationIndex]==HIVVertices.hetero)
        self.biFemaleInds = numpy.logical_and(self.femaleInds, self.graph.vlist.V[:, HIVVertices.orientationIndex]==HIVVertices.bi)

        #We need to store degree sequences for 3 types
        self.expandedDegSeqFemales = Util.expandIntArray(graph.outDegreeSequence()[self.femaleInds]*(self.q-self.p))
        self.expandedDegSeqFemales = numpy.append(self.expandedDegSeqFemales, Util.expandIntArray(hiddenDegSeq[self.femaleInds]*self.p))
        self.expandedDegSeqFemales = numpy.arange(graph.size)[self.femaleInds][self.expandedDegSeqFemales]
        
        self.expandedDegSeqMales = Util.expandIntArray(graph.outDegreeSequence()[self.maleInds]*(self.q-self.p))
        self.expandedDegSeqMales = numpy.append(self.expandedDegSeqMales, Util.expandIntArray(hiddenDegSeq[self.maleInds]*self.p))
        self.expandedDegSeqMales = numpy.arange(graph.size)[self.maleInds][self.expandedDegSeqMales]        
        
        self.expandedDegSeqBiMales = Util.expandIntArray(graph.outDegreeSequence()[self.biMaleInds]*(self.q-self.p))
        self.expandedDegSeqBiMales = numpy.append(self.expandedDegSeqBiMales, Util.expandIntArray(hiddenDegSeq[self.biMaleInds]*self.p))
        self.expandedDegSeqBiMales = numpy.arange(graph.size)[self.biMaleInds][self.expandedDegSeqBiMales]   
        
        self.expandedDegSeqBiFemales = Util.expandIntArray(graph.outDegreeSequence()[self.biFemaleInds]*(self.q-self.p))
        self.expandedDegSeqBiFemales = numpy.append(self.expandedDegSeqBiFemales, Util.expandIntArray(hiddenDegSeq[self.biFemaleInds]*self.p))
        self.expandedDegSeqBiFemales = numpy.arange(graph.size)[self.biFemaleInds][self.expandedDegSeqBiFemales]   

        #Check degree sequence         
        if __debug__: 
            binShape = numpy.bincount(self.expandedDegSeqFemales).shape[0]
            assert (numpy.bincount(self.expandedDegSeqFemales)[self.femaleInds[0:binShape]] == 
                (graph.outDegreeSequence()*(self.q-self.p)+hiddenDegSeq*self.p)[self.femaleInds[0:binShape]]).all()
              
            binShape = numpy.bincount(self.expandedDegSeqMales).shape[0]
            assert (numpy.bincount(self.expandedDegSeqMales)[self.maleInds[0:binShape]] == 
                (graph.outDegreeSequence()*(self.q-self.p)+hiddenDegSeq*self.p)[self.maleInds[0:binShape]]).all()
                
            if self.expandedDegSeqBiMales.shape[0]!=0:
                binShape = numpy.bincount(self.expandedDegSeqBiMales).shape[0]                
                assert (numpy.bincount(self.expandedDegSeqBiMales)[self.biMaleInds[0:binShape]] == 
                    (graph.outDegreeSequence()*(self.q-self.p)+hiddenDegSeq*self.p)[self.biMaleInds[0:binShape]]).all()
                    
            if self.expandedDegSeqBiFemales.shape[0]!=0:
                binShape = numpy.bincount(self.expandedDegSeqBiFemales).shape[0]                
                assert (numpy.bincount(self.expandedDegSeqBiFemales)[self.biFemaleInds[0:binShape]] == 
                    (graph.outDegreeSequence()*(self.q-self.p)+hiddenDegSeq*self.p)[self.biFemaleInds[0:binShape]]).all()

        self.hiddenDegSeq = hiddenDegSeq
        self.degSequence = graph.outDegreeSequence() 

        #Parameters for sexual contact
        self.alpha = 2.0
        self.newContactChance = 0.5
        
        self.contactRate = 0.5

        #Infection probabilities are from wikipedia
        self.infectProb = 50.0/10000

        #Random detection
        self.randDetectRate = 1/720.0
        
        #The max number of people who are being simultaneously detected 
        self.maxDetects = 10

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

        for i in range(graph.size):
            self.neighboursList.append(graph.neighbours(i))
            self.detectedNeighboursList.append(numpy.array([], numpy.int))

    def setAlpha(self, alpha):
        Parameter.checkFloat(alpha, 0.0, float('inf'))
        
        if alpha == 0: 
            raise ValueError("Alpha must be greater than zero")
        
        self.alpha = alpha

    def setNewContactChance(self, newContactChance):
        Parameter.checkFloat(newContactChance, 0.0, float('inf'))
    
        if newContactChance == 0: 
            raise ValueError("newContactChance must be greater than zero")        
        
        self.newContactChance = newContactChance

    def setContactRate(self, contactRate):
        Parameter.checkFloat(contactRate, 0.0, float('inf'))
        self.contactRate = contactRate

    def setRandDetectRate(self, randDetectRate):
        Parameter.checkFloat(randDetectRate, 0.0, float('inf'))
        self.randDetectRate = randDetectRate

    def setCtRatePerPerson(self, ctRatePerPerson):
        Parameter.checkFloat(ctRatePerPerson, 0.0, float('inf'))
        self.ctRatePerPerson = ctRatePerPerson

    def setInfectProb(self, infectProf):
        Parameter.checkFloat(infectProf, 0.0, 1.0)
        self.infectProf = infectProf
        
    def setMaxDetects(self, maxDetects): 
        Parameter.checkInt(maxDetects, 1, float("inf"))
        self.maxDetects = maxDetects 

    #@profile
    def contactEvent(self, vertexInd1, vertexInd2, t):
        """
        Indicates a sexual contact event between two vertices. 
        """
        if vertexInd1 == vertexInd2:
            return 
        if self.graph.getEdge(vertexInd1, vertexInd2) == None:
            for i in [vertexInd1, vertexInd2]:
                self.degSequence[i] += 1 
                if self.graph.vlist.V[i, HIVVertices.genderIndex] == HIVVertices.male:
                    self.expandedDegSeqMales = numpy.append(self.expandedDegSeqMales, numpy.repeat(numpy.array([i]), self.q-self.p))

                    if self.graph.vlist.V[i, HIVVertices.orientationIndex]==HIVVertices.bi:
                        self.expandedDegSeqBiMales = numpy.append(self.expandedDegSeqBiMales, numpy.repeat(numpy.array([i]), self.q-self.p))
                else:
                    self.expandedDegSeqFemales = numpy.append(self.expandedDegSeqFemales, numpy.repeat(numpy.array([i]), self.q-self.p))
                    
                    if self.graph.vlist.V[i, HIVVertices.orientationIndex]==HIVVertices.bi:
                        self.expandedDegSeqBiFemales = numpy.append(self.expandedDegSeqBiFemales, numpy.repeat(numpy.array([i]), self.q-self.p))
           
        if __debug__: 
            inds = numpy.unique(self.expandedDegSeqMales)
            assert (self.graph.vlist.V[inds, HIVVertices.genderIndex] == HIVVertices.male).all()
            
            inds = numpy.unique(self.expandedDegSeqFemales)
            assert (self.graph.vlist.V[inds, HIVVertices.genderIndex] == HIVVertices.female).all()
            
            inds = numpy.unique(self.expandedDegSeqBiMales)
            assert (numpy.logical_and(self.graph.vlist.V[inds, HIVVertices.genderIndex] == HIVVertices.male, self.graph.vlist.V[inds, HIVVertices.orientationIndex] == HIVVertices.bi)).all()
            
            inds = numpy.unique(self.expandedDegSeqBiFemales)
            assert (numpy.logical_and(self.graph.vlist.V[inds, HIVVertices.genderIndex] == HIVVertices.female, self.graph.vlist.V[inds, HIVVertices.orientationIndex] == HIVVertices.bi)).all()
           
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
        self.graph.vlist.setDetected(vertexInd, t, detectionMethod)

        #Note that we don't remove the neighbour because he/she can still be a contact 
        #Therefore this is the degree sequence minus removed vertices 
        if self.graph.vlist.V[vertexInd, HIVVertices.genderIndex] == HIVVertices.male:
            self.expandedDegSeqMales = self.expandedDegSeqMales[self.expandedDegSeqMales!=vertexInd]
            
            if self.graph.vlist.V[vertexInd, HIVVertices.orientationIndex]==HIVVertices.bi:
                self.expandedDegSeqBiMales = self.expandedDegSeqBiMales[self.expandedDegSeqBiMales!=vertexInd]    
        else: 
            self.expandedDegSeqFemales = self.expandedDegSeqFemales[self.expandedDegSeqFemales!=vertexInd]
            
            if self.graph.vlist.V[vertexInd, HIVVertices.orientationIndex]==HIVVertices.bi:
                self.expandedDegSeqBiFemales = self.expandedDegSeqBiFemales[self.expandedDegSeqBiFemales!=vertexInd]    
        

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
        contactRates = numpy.ones(len(infectedList))*self.contactRate
        #contactRates += (self.graph.vlist.V[infectedList, HIVVertices.orientationIndex])*self.contactRate


        return numpy.sum(contactRates)
        
    def upperDetectionRates(self, infectedList, seed=21):
        """
        An upper bound on the detection rates indepedent of time.This is just the
        random detection rate plus the ctRate per person for each detected neighbour. 
        """
        detectionRates = numpy.ones(len(infectedList))*self.randDetectRate

        for i, j in enumerate(infectedList):
            detectionRates[i] += self.detectedNeighboursList[j].shape[0]*self.ctRatePerPerson
            
        state = numpy.random.get_state()
        numpy.random.seed(seed)
        inds = numpy.random.permutation(len(infectedList))[self.maxDetects:]
        detectionRates[inds] = 0 
        numpy.random.set_state(state)
        
        assert (detectionRates!=0).sum() <= self.maxDetects 

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

        infectedV = self.graph.vlist.V[infectedList, :]
        
        maleInfectInds = infectedV[:, HIVVertices.genderIndex]==HIVVertices.male
        femaleInfectInds = numpy.logical_not(maleInfectInds)
        biInfectInds = infectedV[:, HIVVertices.orientationIndex]==HIVVertices.bi
        heteroInfectInds = numpy.logical_not(biInfectInds)
        maleHeteroInfectInds = numpy.logical_and(maleInfectInds, heteroInfectInds)
        maleBiInfectInds = numpy.logical_and(maleInfectInds, biInfectInds)
        femaleHeteroInfectInds = numpy.logical_and(femaleInfectInds, heteroInfectInds)
        femaleBiInfectInds = numpy.logical_and(femaleInfectInds, biInfectInds)

        #possibleContacts has as its first column the previous contacts.
        #A set of new contacts based on the degree sequence
        #These contacts can be any one of the other contacts that are non-removed
        numPossibleContacts = 2
        possibleContacts = numpy.zeros((len(infectedList), numPossibleContacts), numpy.int)
        possibleContactWeights = numpy.zeros((len(infectedList), numPossibleContacts))
        
        possibleContacts[:, 0] = self.contactTimesArr[infectedList, 0]

        totalDegSequence = numpy.array(self.hiddenDegSeq*self.p + self.degSequence*(self.q-self.p), numpy.float)
        #assert (self.expandedDegSeqFemales.shape[0] + self.expandedDegSeqMales.shape[0]) == totalDegSequence.sum(), \
        #    "totalDegSequence.sum()=%d, expanded=%d" % (totalDegSequence.sum(), self.expandedDegSeqFemales.shape[0] + self.expandedDegSeqMales.shape[0])

        #Note that we may get duplicates for possible contacts since we don't check for it
        edsInds = numpy.random.randint(0, self.expandedDegSeqFemales.shape[0], maleHeteroInfectInds.sum())
        contactInds = self.expandedDegSeqFemales[edsInds]
        possibleContacts[maleHeteroInfectInds, 1] = contactInds
        possibleContactWeights[maleHeteroInfectInds, 1] = totalDegSequence[contactInds]/self.expandedDegSeqFemales.shape[0]

        edsInds = numpy.random.randint(0, self.expandedDegSeqMales.shape[0], femaleInfectInds.sum())
        contactInds = self.expandedDegSeqMales[edsInds]
        possibleContacts[femaleInfectInds, 1] = contactInds
        possibleContactWeights[femaleInfectInds, 1] = totalDegSequence[contactInds]/self.expandedDegSeqMales.shape[0]

        if self.expandedDegSeqBiMales.shape[0] != 0:
            edsInds = numpy.random.randint(0, self.expandedDegSeqBiMales.shape[0], maleBiInfectInds.sum())
            contactInds = self.expandedDegSeqBiMales[edsInds]
            possibleContacts[maleBiInfectInds, 1] = contactInds
            possibleContactWeights[maleBiInfectInds, 1] = totalDegSequence[contactInds]/self.expandedDegSeqBiMales.shape[0]

        if self.expandedDegSeqBiFemales.shape[0] != 0:
            edsInds = numpy.random.randint(0, self.expandedDegSeqBiFemales.shape[0], femaleBiInfectInds.sum())
            contactInds = self.expandedDegSeqBiFemales[edsInds]
            possibleContacts[femaleBiInfectInds, 1] = contactInds
            possibleContactWeights[femaleBiInfectInds, 1] = totalDegSequence[contactInds]/self.expandedDegSeqBiFemales.shape[0]

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
        contactsV = self.graph.vlist.V[contacts, :]

        #The first column is the contact (if any) and the 2nd is the contact rate 
        contactRateInds = numpy.ones(len(infectedList), numpy.int)*-1
        contactRates = numpy.zeros(len(infectedList))
        
        #We compute sexual contact rates between infected and everyone else
        equalGender = infectedV[:, HIVVertices.genderIndex]==contactsV[:, HIVVertices.genderIndex]
        hInds = numpy.nonzero(numpy.logical_not(equalGender))[0]
        contactRateInds[hInds] = contacts[hInds]
        contactRates[hInds] = self.contactRate
        
        #We only simulate contact between male homosexuals (woman-women contact is not interesting)
        bothMale = numpy.logical_and(equalGender, infectedV[:, HIVVertices.genderIndex]==HIVVertices.male)
        bInds = numpy.nonzero(numpy.logical_and(numpy.logical_and(bothMale, contactsV[:, HIVVertices.orientationIndex]==HIVVertices.bi), biInfectInds))[0]
        contactRateInds[bInds] = contacts[bInds]
        contactRates[bInds] = self.contactRate

        return contactRateInds, contactRates

    """
    Compute the infection probability between an infected and susceptible.
    """
    def infectionProbability(self, vertexInd1, vertexInd2, t):
        """
        This returns the infection probability of an infected person vertexInd1
        and a non-removed vertexInd2. 
        """
        vertex1 = self.graph.vlist.V[vertexInd1, :]
        vertex2 = self.graph.vlist.V[vertexInd2, :]

        if vertex1[HIVVertices.stateIndex]!=HIVVertices.infected or vertex2[HIVVertices.stateIndex]!=HIVVertices.susceptible:
            return 0.0

        if vertex1[HIVVertices.genderIndex] == HIVVertices.female and vertex2[HIVVertices.genderIndex] == HIVVertices.male:
            return self.infectProb
        elif vertex1[HIVVertices.genderIndex] == HIVVertices.male and vertex2[HIVVertices.genderIndex] == HIVVertices.female:
            return self.infectProb
        elif vertex1[HIVVertices.genderIndex] == HIVVertices.male and vertex1[HIVVertices.orientationIndex]==HIVVertices.bi and vertex2[HIVVertices.genderIndex] == HIVVertices.male and vertex2[HIVVertices.orientationIndex]==HIVVertices.bi:
            return self.infectProb
        else:
            #Corresponds to 2 bisexual women 
            return 0.0 


    def randomDetectionRates(self, infectedList, t, seed=21):
        """
        Compute the detection rate of an infected which depends on the entire population.
        In this case it's randDetectRate/|I_t|. 
        """
        detectionRates = numpy.zeros(len(infectedList))
        state = numpy.random.get_state()
        numpy.random.seed(seed)
        inds = numpy.random.permutation(len(infectedList))[0:self.maxDetects]
        detectionRates[inds] = self.randDetectRate #/len(infectedList)
        numpy.random.set_state(state)
        return detectionRates

    def contactTracingRates(self, infectedList, removedSet, t, seed=21):
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
        underCT = numpy.zeros(self.graph.size, numpy.bool)
        underCT[removeIndices] = numpy.logical_and(self.graph.vlist.V[removeIndices, HIVVertices.detectionTimeIndex] >= cdEndDate, self.graph.vlist.V[removeIndices, HIVVertices.detectionTimeIndex] <= ctStartDate)

        if len(infectedList) < len(removedSet):
            #Possibly store set of detected neighbours
            for i in range(len(infectedList)):
                vertexInd = infectedList[i]
                detectedNeighbours = self.detectedNeighboursList[vertexInd]

                for ind in detectedNeighbours:
                    if underCT[ind]:
                        ctRates[i] = self.ctRatePerPerson
                #This is slower for some reason
                #ctRates[i] = numpy.sum(underCT[neighbours]) * self.ctRatePerPerson
        else:
            for vertexInd in removedSet:
                if underCT[vertexInd]:
                    neighbours = self.neighboursList[vertexInd]

                    for ind in neighbours:
                        if self.graph.vlist.V[ind, HIVVertices.stateIndex] == HIVVertices.infected:
                            i = numpy.searchsorted(infectedArray, ind)
                            ctRates[infectedArrInds[i]] = self.ctRatePerPerson

        assert (ctRates >= numpy.zeros(len(infectedList))).all()

        #Only maxDetects can be tested at once 
        """
        state = numpy.random.get_state()
        numpy.random.seed(seed)
        inds = numpy.random.permutation(len(infectedList))[self.maxDetects:]
        ctRates[inds] = 0
        numpy.random.set_state(state)
        """

        return ctRates