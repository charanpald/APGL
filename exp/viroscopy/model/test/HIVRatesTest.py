import apgl
import numpy 
import unittest
import scipy.stats 
import logging
try:
    from apgl.viroscopy.model.HIVRates import HIVRates
    from apgl.viroscopy.model.HIVGraph import HIVGraph
except ImportError:
    pass
from apgl.viroscopy.model.HIVVertices import HIVVertices

@apgl.skipIf(not apgl.checkImport('pysparse'), 'No module pysparse')
class  HIVRateFuncsTestCase(unittest.TestCase):
    def setUp(self):
        numpy.set_printoptions(suppress=True)
        numpy.random.seed(21)

        s = 3
        self.gen = scipy.stats.zipf(s)
        

    def testContactEvent(self):
        undirected = True
        numVertices = 10
        graph = HIVGraph(numVertices, undirected)

        #for i in range(numVertices):
        #    logging.debug(graph.getVertex(i))

        t = 0.2
        hiddenDegSeq = self.gen.rvs(size=graph.getNumVertices())
        rates = HIVRates(graph, hiddenDegSeq)

        V = graph.getVertexList().getVertices()
        femaleInds = V[:, HIVVertices.genderIndex]==HIVVertices.female
        maleInds = V[:, HIVVertices.genderIndex]==HIVVertices.male
        biMaleInds = numpy.logical_and(maleInds, V[:, HIVVertices.orientationIndex]==HIVVertices.bi)

        self.assertEquals(rates.expandedDegSeqFemales.shape[0], hiddenDegSeq[femaleInds].sum()*rates.p)
        self.assertEquals(rates.expandedDegSeqMales.shape[0], hiddenDegSeq[maleInds].sum()*rates.p)
        self.assertEquals(rates.expandedDegSeqBiMales.shape[0], hiddenDegSeq[biMaleInds].sum()*rates.p)

        for i in range(numVertices):
            self.assertTrue((rates.contactTimesArr[i, :] == numpy.array([-float('inf'), -float('inf')])).all())

        rates.contactEvent(0, 9, 0.1)
        rates.contactEvent(0, 3, 0.2)
        
        self.assertEquals(graph.getEdge(0, 3), 0.2)
        self.assertEquals(graph.getEdge(0, 9), 0.1)

        self.assertTrue((rates.contactTimesArr[0, :] == numpy.array([3, 0.2])).all())
        self.assertTrue((rates.contactTimesArr[9, :] == numpy.array([0, 0.1])).all())
        self.assertTrue((rates.contactTimesArr[3, :] == numpy.array([0, 0.2])).all())

        for i in range(numVertices):
            self.assertTrue((rates.neighboursList[i] == graph.neighbours(i)).all())

        #Check that the degree sequence is correct
        degSequence = graph.outDegreeSequence()
        r = rates.q-rates.p 

        self.assertEquals(rates.expandedDegSeqFemales.shape[0], hiddenDegSeq[femaleInds].sum()*rates.p + degSequence[femaleInds].sum()*r)
        self.assertEquals(rates.expandedDegSeqMales.shape[0], hiddenDegSeq[maleInds].sum()*rates.p + degSequence[maleInds].sum()*r)
        self.assertEquals(rates.expandedDegSeqBiMales.shape[0], hiddenDegSeq[biMaleInds].sum()*rates.p + degSequence[biMaleInds].sum()*r)

    def testContactRates(self):
        undirected = True 
        numVertices = 10
        graph = HIVGraph(numVertices, undirected)

        t = 0.2

        contactList = range(numVertices)

        hiddenDegSeq = self.gen.rvs(size=graph.getNumVertices())
        rates = HIVRates(graph, hiddenDegSeq)
        contactRates = rates.contactRates([0, 5, 7], contactList, t)
        self.assertEquals(contactRates.shape[0], 3)
        self.assertEquals(contactRates.shape[1], 10)

        #Now we have that 0 had contact with another
        rates.contactEvent(0, 3, 0.2)
        rates.contactEvent(0, 9, 0.1)

        contactRates = rates.contactRates(list(range(numVertices)), contactList, t)
       
        inds = list(contactRates.keys())

        #Note that in some cases an infected has no contacted as the persons do not match 
        for ind in inds:
            if graph.getVertex(ind[0])[HIVVertices.genderIndex]==graph.getVertex(ind[1])[HIVVertices.genderIndex]:
                self.assertEquals(contactRates[ind],rates.heteroContactRate)
            elif graph.getVertex(ind[0])[HIVVertices.genderIndex]!=graph.getVertex(ind[1])[HIVVertices.genderIndex] and graph.getVertex(ind[0])[HIVVertices.orientationIndex]==HIVVertices.bi and graph.getVertex(ind[1])[HIVVertices.orientationIndex]==HIVVertices.bi:
                self.assertEquals(contactRates[ind],rates.biContactRate)

    def testContactRates2(self):
        undirected = True
        numVertices = 10
        graph = HIVGraph(numVertices, undirected)

        maleVertex = graph.getVertex(0)
        maleVertex[HIVVertices.genderIndex] = HIVVertices.male
        femaleVertex = maleVertex.copy()
        femaleVertex[HIVVertices.genderIndex] = HIVVertices.female

        for i in range(5): 
            graph.setVertex(i, maleVertex)
            graph.setVertex(i+5, femaleVertex)

        V = graph.getVertexList().getVertices()

        contactList = range(numVertices)

        #Test that the parameters alpha and C do the right thing
        hiddenDegSeq = self.gen.rvs(size=graph.getNumVertices())
        rates = HIVRates(graph, hiddenDegSeq)
        t = 0.2
        logging.debug("Rates with no existing contacts")
        contactRates = rates.contactRates(range(numVertices), contactList, t)

        #When there are no contacts the choice is easy and some random new contacts
        #are chosen.
        #Now test differences in choice between existing and new contact.
        t = 0.3
        for i in range(5):
            rates.contactEvent(i, i+5, t)

        logging.debug("Rates with default C=" + str(rates.newContactChance))
        contactRates = rates.contactRates(range(numVertices), contactList, 0.4)

        for i in range(5):
            self.assertTrue(contactRates[i, i+5] == rates.heteroContactRate)

        #Now try changing C
        logging.debug("Rates with C=0.1")
        rates.setNewContactChance(0.1)
        contactRates = rates.contactRates(range(numVertices), contactList, 0.4)
        #Observed probabilities change as expected

        #Now increase time and observe probabilities
        logging.debug("Rates with t=20")
        contactRates = rates.contactRates(range(numVertices), contactList, 20)


        #Test we don't pick from removed
        graph.getVertexList().setInfected(0, t)
        graph.getVertexList().setInfected(4, t)
        graph.getVertexList().setInfected(7, t)
        graph.getVertexList().setInfected(8, t)
        #graph.getVertexList().setDetected(4, t, HIVVertices.randomDetect)
        #graph.getVertexList().setDetected(7, t, HIVVertices.randomDetect)
        rates.removeEvent(4, HIVVertices.randomDetect, t)
        rates.removeEvent(7, HIVVertices.randomDetect, t)

        infectedSet = graph.getInfectedSet()
        susceptibleSet = graph.getSusceptibleSet()
        removedSet = graph.getRemovedSet()
        contactSet = infectedSet.union(susceptibleSet)

        infectedList = list(infectedSet)
        removedList = list(removedSet)
        contactList = list(contactSet)



    def testContactTracingRate(self):
        undirected = True
        numVertices = 10
        graph = HIVGraph(numVertices, undirected)

        hiddenDegSeq = self.gen.rvs(size=graph.getNumVertices())
        rates = HIVRates(graph, hiddenDegSeq)
        t = 0.1
        graph.getVertexList().setInfected(0, t)
        rates.contactEvent(0, 3, 0.2)
        rates.contactEvent(0, 9, 0.1)

        t = 0.3
        graph.getVertexList().setInfected(3, t)
        graph.getVertexList().setInfected(9, t)

        t = 0.4
        rates.removeEvent(0, HIVVertices.randomDetect, t)

        removedSet = graph.getRemovedSet()
        infectedList = [3, 9]
        ctRates = rates.contactTracingRates(infectedList, removedSet, t)
        self.assertTrue((ctRates==numpy.array([0.0, 0.0])).all())

        ctRates = rates.contactTracingRates(infectedList, removedSet, t+rates.ctStartTime)
        self.assertTrue((ctRates == numpy.array([rates.ctRatePerPerson, rates.ctRatePerPerson])).all())

        #Test contact tracing is within correct time period
        ctRates = rates.contactTracingRates(infectedList, removedSet, t+rates.ctEndTime-0.01)
        self.assertTrue((ctRates == numpy.array([rates.ctRatePerPerson, rates.ctRatePerPerson])).all())

        ctRates = rates.contactTracingRates(infectedList, removedSet, t+rates.ctEndTime+1)
        self.assertTrue((ctRates == numpy.array([0, 0])).all())

        rates.contactEvent(3, 5, t)
        graph.getVertexList().setInfected(5, t)
        rates.removeEvent(5, HIVVertices.randomDetect, t)
        removedSet = graph.getRemovedSet()
        ctRates = rates.contactTracingRates(infectedList, removedSet, t+rates.ctStartTime)
        self.assertTrue((ctRates == numpy.array([rates.ctRatePerPerson*2, rates.ctRatePerPerson])).all())
        
        rates.contactEvent(3, 6, t)
        graph.getVertexList().setInfected(6, t)
        infectedList = [3, 6, 9]
        removedSet = graph.getRemovedSet()
 
        ctRates = rates.contactTracingRates(infectedList, removedSet, t+rates.ctStartTime)
        self.assertTrue((ctRates == numpy.array([rates.ctRatePerPerson*2, 0, rates.ctRatePerPerson])).all())

        #Now make removedSet bigger than infectedList
        graph.getVertexList().setInfected(4, t)
        graph.getVertexList().setInfected(7, t)
        graph.getVertexList().setInfected(8, t)
        graph.getVertexList().setDetected(4, t, HIVVertices.randomDetect)
        graph.getVertexList().setDetected(7, t, HIVVertices.randomDetect)
        graph.getVertexList().setDetected(8, t, HIVVertices.randomDetect)

        #Note: InfectedList is out of order 
        infectedList = list(graph.getInfectedSet())
        sortInds = numpy.argsort(numpy.array(infectedList))
        removedSet = graph.getRemovedSet()

        ctRates = rates.contactTracingRates(infectedList, removedSet, t+rates.ctStartTime)
        ctRates2 = numpy.array([rates.ctRatePerPerson*2, 0, rates.ctRatePerPerson])
        self.assertTrue((ctRates[sortInds] == ctRates2).all())

        #Test the case where InfectedList is out of order and removedSet is small
        graph.getVertexList().setInfected(4, t)
        graph.getVertex(7)[HIVVertices.stateIndex] = HIVVertices.susceptible
        graph.getVertex(8)[HIVVertices.stateIndex] = HIVVertices.susceptible

        infectedList = list(graph.getInfectedSet())
        sortInds = numpy.argsort(numpy.array(infectedList))
        removedSet = graph.getRemovedSet()

        ctRates = rates.contactTracingRates(infectedList, removedSet, t+rates.ctStartTime)
        ctRates2 = numpy.array([rates.ctRatePerPerson*2, 0, 0, rates.ctRatePerPerson])
        self.assertTrue((ctRates[sortInds] == ctRates2).all())
        

    def testRandomDetectionRates(self):
        undirected = True
        numVertices = 10
        graph = HIVGraph(numVertices, undirected)

        t = 0.1
        graph.getVertexList().setInfected(0, t)

        hiddenDegSeq = self.gen.rvs(size=graph.getNumVertices())
        rates = HIVRates(graph, hiddenDegSeq)
        infectedList = [0, 2, 9]

        rdRates = rates.randomDetectionRates(infectedList, t)

        self.assertTrue((rdRates == numpy.ones(len(infectedList))*rates.randDetectRate).all())

    def testInfectionProbability(self):
        undirected = True
        numVertices = 10
        graph = HIVGraph(numVertices, undirected)
        hiddenDegSeq = self.gen.rvs(size=graph.getNumVertices())
        rates = HIVRates(graph, hiddenDegSeq)
        t = 0.1

        graph.getVertex(0)[HIVVertices.stateIndex] = HIVVertices.infected
        graph.getVertex(1)[HIVVertices.stateIndex] = HIVVertices.removed
        graph.getVertex(2)[HIVVertices.stateIndex] = HIVVertices.infected

        for vertexInd1 in range(numVertices):
            for vertexInd2 in range(numVertices): 
                vertex1 = graph.getVertex(vertexInd1)
                vertex2 = graph.getVertex(vertexInd2)

                if vertex1[HIVVertices.stateIndex]!=HIVVertices.infected or vertex2[HIVVertices.stateIndex]!=HIVVertices.susceptible:
                    self.assertEquals(rates.infectionProbability(vertexInd1, vertexInd2, t), 0.0)
                elif vertex1[HIVVertices.genderIndex] == HIVVertices.female and vertex2[HIVVertices.genderIndex] == HIVVertices.male:
                    self.assertEquals(rates.infectionProbability(vertexInd1, vertexInd2, t), rates.womanManInfectProb) 
                elif vertex1[HIVVertices.genderIndex] == HIVVertices.male and vertex2[HIVVertices.genderIndex] == HIVVertices.female:
                    self.assertEquals(rates.infectionProbability(vertexInd1, vertexInd2, t), rates.manWomanInfectProb)
                elif vertex1[HIVVertices.genderIndex] == HIVVertices.male and vertex2[HIVVertices.orientationIndex]==HIVVertices.bi:
                    self.assertEquals(rates.infectionProbability(vertexInd1, vertexInd2, t), rates.manBiInfectProb)
                else:
                    self.assertEquals(rates.infectionProbability(vertexInd1, vertexInd2, t), 0.0)

    def testRemoveEvent(self):
        undirected = True
        numVertices = 10
        graph = HIVGraph(numVertices, undirected)
        hiddenDegSeq = self.gen.rvs(size=graph.getNumVertices())
        rates = HIVRates(graph, hiddenDegSeq)
        t = 0.1

        V = graph.getVertexList().getVertices()
        femaleInds = V[:, HIVVertices.genderIndex]==HIVVertices.female
        maleInds = V[:, HIVVertices.genderIndex]==HIVVertices.male
        biMaleInds = numpy.logical_and(maleInds, V[:, HIVVertices.orientationIndex]==HIVVertices.bi)

        self.assertEquals(rates.expandedDegSeqFemales.shape[0], hiddenDegSeq[femaleInds].sum()*rates.p)
        self.assertEquals(rates.expandedDegSeqMales.shape[0], hiddenDegSeq[maleInds].sum()*rates.p)
        self.assertEquals(rates.expandedDegSeqBiMales.shape[0], hiddenDegSeq[biMaleInds].sum()*rates.p)

        graph.getVertexList().setInfected(4, t)
        graph.getVertexList().setInfected(7, t)
        graph.getVertexList().setInfected(8, t)
        rates.removeEvent(4, HIVVertices.randomDetect, t)
        rates.removeEvent(7, HIVVertices.randomDetect, t)

        #Check the new degree sequences are correct 
        self.assertEquals(rates.expandedDegSeqFemales.shape[0], hiddenDegSeq[femaleInds].sum()*rates.p)
        self.assertEquals(rates.expandedDegSeqMales.shape[0], hiddenDegSeq[maleInds].sum()*rates.p)
        self.assertEquals(rates.expandedDegSeqBiMales.shape[0], hiddenDegSeq[biMaleInds].sum()*rates.p)


if __name__ == '__main__':
    unittest.main()

