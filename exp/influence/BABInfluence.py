"""
A class to maximise influence in a network using the branch and bound method. s
"""
import numpy
import logging
from apgl.util.Parameter import Parameter
from apgl.influence.GreedyInfluence import GreedyInfluence

class BABInfluence(object):
    def __init__(self):
        self.bestInfluence = 0
        self.selectedIndices = []

    def maxBudgetedInfluence(self, P, u, L):
        Parameter.checkFloat(L, 0.0, float('inf'))

        #Make the highest cost vertices appear first so we reach the budget faster 
        sortedInds = numpy.argsort(-u)
        P2 = P[sortedInds, :]
        u2 = u[sortedInds]

        greedyInfluence = GreedyInfluence()
        self.selectedIndices = greedyInfluence.maxBudgetedInfluence(P2, u2, L)
        self.bestInfluence = numpy.sum(numpy.max(P2[self.selectedIndices, :], 0))

        #Now start the branch and bound method
        selectedIndices = []
        self.__followBranch(selectedIndices, P2, u2, L, 0)
        return numpy.sort(sortedInds[self.selectedIndices]).tolist()

    def __followBranch(self, selectedIndices, P, u, L, depth):
        logging.debug("Calling followBranch with " + str(selectedIndices) + ", " + str(depth) + ", cost: " + str(numpy.sum(u[selectedIndices])))
        
        if numpy.sum(u[selectedIndices]) > L:
            return 

        #We have to do this because numpy.max has a problem with empty arrays 
        if selectedIndices != []:
            currentInfluence = numpy.sum(numpy.max(P[selectedIndices, :], 0))
        else:
            currentInfluence = 0 
 
        if currentInfluence > self.bestInfluence:
            self.bestInfluence = currentInfluence
            self.selectedIndices = selectedIndices

        extendedIndices = list(selectedIndices)
        extendedIndices.extend(list(range(depth, P.shape[0])))
        influenceBound = numpy.sum(numpy.max(P[extendedIndices, :], 0))

        if influenceBound <= self.bestInfluence or depth == P.shape[0]:
            return
        else:
            newIndices1 = list(selectedIndices)
            self.__followBranch(newIndices1, P, u, L, depth+1)
            
            newIndices2 = list(selectedIndices) 
            newIndices2.append(depth)
            self.__followBranch(newIndices2, P, u, L, depth+1)
