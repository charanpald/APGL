"""
We simulate a network of sexual contacts and see if we can approximate the degree
using a subset of the edges. 
"""
import numpy 
from apgl.viroscopy.model.HIVGraph import HIVGraph
from apgl.util.Util import Util

numpy.set_printoptions(linewidth=200, threshold=10000)

numVertices = 500
graph = HIVGraph(numVertices)

T = 100
t = 0

expandedDegreeSeq = Util.expandIntArray(graph.outDegreeSequence())
expandedDegreeSeq = numpy.arange(numVertices)
contactRate = 0.01

tau = 1.0
infectedList = range(10)
#infectedList = range(numVertices)

while (t < T):
    contactRatesMatrix = numpy.zeros((len(infectedList), numVertices))

    #Select a random contact based on the degree
    edsInds = numpy.random.randint(0, expandedDegreeSeq.shape[0], len(infectedList))
    #Choose between a contact sampled from those with edges and one from random


    
    newContacts = numpy.zeros((len(infectedList), 2))
    newContacts[:, 0] = expandedDegreeSeq[edsInds]
    newContacts[:, 1] = numpy.random.randint(0, numVertices, len(infectedList))
    p = numpy.random.rand(len(infectedList))
    newContacts = newContacts[(numpy.arange(len(infectedList)), numpy.array(p<tau, numpy.int))]

    

    #newContacts = expandedDegreeSeq[edsInds]
    
    for i in range(len(infectedList)):
        if numpy.sum(contactRatesMatrix[infectedList[i], :]) == 0:
            contactRatesMatrix[infectedList[i], newContacts[i]] = contactRate

    #Find time of next contact
    rhot = numpy.sum(contactRatesMatrix)
    tauPrime = numpy.random.exponential(1/rhot)
    t = t + tauPrime

    #Randomly choose the contact
    nzInds = numpy.nonzero(contactRatesMatrix)
    ind = numpy.random.randint(nzInds[0].shape[0])

    vInd1 = nzInds[0][ind]
    vInd2 = nzInds[1][ind]

    if graph.getEdge(infectedList[vInd1], vInd2) == None:
        graph.addEdge(infectedList[vInd1], vInd2)
        if infectedList[vInd1] != vInd2:
            expandedDegreeSeq = numpy.append(expandedDegreeSeq, numpy.array([infectedList[vInd1], vInd2]))
        else:
            expandedDegreeSeq = numpy.append(expandedDegreeSeq, numpy.array([infectedList[vInd1]]))

    print("t=" + str(t))

#print(expandedDegreeSeq)
print(graph.outDegreeSequence())
print(graph.degreeDistribution())

#Result is power law as expected.
#Now model contacts for subset of people
allNeighbours = numpy.array([])

for i in range(10):
    allNeighbours = numpy.append(allNeighbours, graph.neighbours(i))

print(numpy.unique(allNeighbours).shape)
print(numpy.unique(allNeighbours)) 