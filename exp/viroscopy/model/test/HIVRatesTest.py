import apgl
import numpy 
import unittest
import scipy.stats 
import logging
import numpy.testing as nptst 
from exp.viroscopy.model.HIVRates import HIVRates
from exp.viroscopy.model.HIVGraph import HIVGraph
from exp.viroscopy.model.HIVVertices import HIVVertices

@apgl.skipIf(not apgl.checkImport('sppy'), 'No module pysparse')
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
            self.assertEquals(rates.contactTimesArr[i], -1)

        rates.contactEvent(0, 9, 0.1)
        rates.contactEvent(0, 3, 0.2)
        
        self.assertEquals(graph.getEdge(0, 3), 0.2)
        self.assertEquals(graph.getEdge(0, 9), 0.1)

        self.assertTrue((rates.contactTimesArr[0] == numpy.array([3])).all())
        self.assertTrue((rates.contactTimesArr[9] == numpy.array([0])).all())
        self.assertTrue((rates.contactTimesArr[3] == numpy.array([0])).all())

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
        contactRateInds, contactRates = rates.contactRates([0, 5, 7], contactList, t)
        self.assertEquals(contactRates.shape[0], 3)

        #Now we have that 0 had contact with another
        rates.contactEvent(0, 3, 0.2)
        rates.contactEvent(1, 9, 0.1)
        
        infectedInds = numpy.arange(numVertices)
        contactRateInds, contactRates = rates.contactRates(infectedInds, contactList, t)

        #Note that in some cases an infected has no contacted as the persons do not match 
        for i in range(infectedInds.shape[0]): 
            if contactRateInds[i] != -1: 
                if graph.getVertex(infectedInds[i])[HIVVertices.genderIndex]==graph.getVertex(contactRateInds[i])[HIVVertices.genderIndex]:
                    self.assertEquals(contactRates[i], rates.heteroContactRate)
                elif graph.getVertex(infectedInds[i])[HIVVertices.genderIndex]!=graph.getVertex(contactRateInds[1])[HIVVertices.genderIndex] and graph.getVertex(infectedInds[i])[HIVVertices.orientationIndex]==HIVVertices.bi and graph.getVertex(contactRateInds[i])[HIVVertices.orientationIndex]==HIVVertices.bi:
                    self.assertEquals(contactRates[i],rates.biContactRate)

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
        contactRateInds, contactRates = rates.contactRates(range(numVertices), contactList, t)

        #When there are no contacts the choice is easy and some random new contacts
        #are chosen.
        #Now test differences in choice between existing and new contact.
        t = 0.3
        for i in range(5):
            rates.contactEvent(i, i+5, t)

        rates.alpha = 1.0
        logging.debug("Rates with default alpha=" + str(rates.alpha))
        contactRateInds, contactRates = rates.contactRates(range(numVertices), contactList, 0.4)


        for i in range(5):
            self.assertTrue(contactRates[i] == rates.contactRate)
            self.assertTrue(contactRateInds[i] == i+5)

        #Now try changing alpha
        logging.debug("Rates with alpha=0.5")
        rates.setAlpha(0.5)
        contactRateInds, contactRates = rates.contactRates(range(numVertices), contactList, 0.4)
        #Observed probabilities change as expected


        #Now increase time and observe probabilities
        logging.debug("Rates with t=20")
        contactRateInds, contactRates = rates.contactRates(range(numVertices), contactList, 20)


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

        contactRateInds, contactRates = rates.contactRates(infectedList, contactList, 20)
        
        #Contacts cannot be in removed set 
        self.assertTrue(numpy.intersect1d(contactRateInds, numpy.array(removedList)).shape[0]==0)        
            

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

        self.assertTrue((ctRates == numpy.array([rates.ctRatePerPerson, rates.ctRatePerPerson])).all())
        
        rates.contactEvent(3, 6, t)
        graph.getVertexList().setInfected(6, t)
        infectedList = [3, 6, 9]
        removedSet = graph.getRemovedSet()
 
        ctRates = rates.contactTracingRates(infectedList, removedSet, t+rates.ctStartTime)
        self.assertTrue((ctRates == numpy.array([rates.ctRatePerPerson, 0, rates.ctRatePerPerson])).all())

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
        ctRates2 = numpy.array([rates.ctRatePerPerson, 0, rates.ctRatePerPerson])
        self.assertTrue((ctRates[sortInds] == ctRates2).all())

        #Test the case where InfectedList is out of order and removedSet is small
        graph.getVertexList().setInfected(4, t)
        graph.getVertex(7)[HIVVertices.stateIndex] = HIVVertices.susceptible
        graph.getVertex(8)[HIVVertices.stateIndex] = HIVVertices.susceptible

        infectedList = list(graph.getInfectedSet())
        sortInds = numpy.argsort(numpy.array(infectedList))
        removedSet = graph.getRemovedSet()

        ctRates = rates.contactTracingRates(infectedList, removedSet, t+rates.ctStartTime)
        ctRates2 = numpy.array([rates.ctRatePerPerson, 0, 0, rates.ctRatePerPerson])
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

        rdRates = rates.randomDetectionRates(infectedList, float(graph.size - len(graph.getRemovedSet())))

        nptst.assert_array_almost_equal(rdRates, numpy.ones(len(infectedList))*rates.randDetectRate*len(infectedList)/float(graph.size - len(graph.getRemovedSet())))

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
                    self.assertEquals(rates.infectionProbability(vertexInd1, vertexInd2, t), rates.infectProb) 
                elif vertex1[HIVVertices.genderIndex] == HIVVertices.male and vertex2[HIVVertices.genderIndex] == HIVVertices.female:
                    self.assertEquals(rates.infectionProbability(vertexInd1, vertexInd2, t), rates.infectProb)
                elif vertex1[HIVVertices.genderIndex] == HIVVertices.male and vertex2[HIVVertices.orientationIndex]==HIVVertices.bi:
                    self.assertEquals(rates.infectionProbability(vertexInd1, vertexInd2, t), rates.infectProb)
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
        
        removedInds= list(graph.getRemovedSet())    
        
        hiddenDegSeq[removedInds] = 0 
        
        #Check the new degree sequences are correct 
        self.assertEquals(rates.expandedDegSeqFemales.shape[0], hiddenDegSeq[femaleInds].sum()*rates.p)
        self.assertEquals(rates.expandedDegSeqMales.shape[0], hiddenDegSeq[maleInds].sum()*rates.p)
        self.assertEquals(rates.expandedDegSeqBiMales.shape[0], hiddenDegSeq[biMaleInds].sum()*rates.p)


    def testUpperDetectionRates(self): 
        """
        See if the upper bound on detection rates is correct 
        """
        undirected = True
        numVertices = 10
        graph = HIVGraph(numVertices, undirected)
        hiddenDegSeq = self.gen.rvs(size=graph.getNumVertices())
        rates = HIVRates(graph, hiddenDegSeq)
        t = 0.1
        
        graph.getVertexList().setInfected(0, t)
        graph.getVertexList().setInfected(1, t)
        graph.getVertexList().setInfected(8, t)
        
        t = 0.2
        rates.removeEvent(8, HIVVertices.randomDetect, t)
        rates.infectionProbability = 1.0
        
        infectedList = graph.infectedIndsAt(t)
        removedList = graph.removedIndsAt(t)
        n = graph.size-removedList
        self.assertEquals(rates.upperDetectionRates(infectedList, n), rates.randomDetectionRates(infectedList, n, seed=21).sum()) 
        
        t = 0.3
        rates.contactEvent(0, 2, t)
        graph.vlist.setInfected(2, t)
        
        t = 0.4
        rates.removeEvent(0, HIVVertices.randomDetect, t)
        
        infectedList = graph.infectedIndsAt(t)
        removedSet = graph.removedIndsAt(t)
        removedSet = set(removedSet.tolist())
        print(infectedList, removedSet)
        nptst.assert_array_almost_equal(rates.contactTracingRates(infectedList, removedSet, t + rates.ctStartTime + 1), numpy.array([0, rates.ctRatePerPerson]))
        
        upperDetectionRates = rates.ctRatePerPerson + rates.randomDetectionRates(infectedList, n, seed=21).sum()
        self.assertEquals(rates.upperDetectionRates(infectedList, n), upperDetectionRates) 

if __name__ == '__main__':
    unittest.main()

