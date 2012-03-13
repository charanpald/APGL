"""
A class to maximise influence in a network in a greedy manner.
"""
import numpy
import logging
from apgl.util.Parameter import Parameter
from apgl.util.Util import Util

class GreedyInfluence(object):
    def __init__(self):
        pass 

    def maxInfluence(self, P, k):
        """
        The matrix P is one representing the quality of information reaching each
        node from certain input nodes. The i,jth entry is the quality of information
        reaching vertex j from i. Returns the k nodes of maximal influence using
        a greedy method. 

        Complexity is O(k n)
        """
        Parameter.checkInt(k, 0, P.shape[0])

        numVertices = P.shape[0]
        bestActivations = numpy.zeros(numVertices)
        bestTotalActivation = 0
        
        selectedIndices = []
        unselectedIndices = set(range(0, numVertices))
        stepSize = 50

        for i in range(0, k):
            Util.printIteration(i, stepSize, k)

            for j in unselectedIndices:
                activations = numpy.max(numpy.r_['0,2', P[j, :], bestActivations], 0)
                currentActivation = numpy.sum(activations)
                
                if currentActivation > bestTotalActivation:
                    bestIndex = j
                    bestTotalActivation = numpy.sum(currentActivation)
                    
            bestActivations = numpy.max(numpy.r_['0,2', P[bestIndex, :], bestActivations], 0)

            if bestIndex in selectedIndices:
                bestIndex = unselectedIndices.copy().pop()

            selectedIndices.append(bestIndex)
            unselectedIndices.remove(bestIndex)

        return selectedIndices
            

    def maxBudgetedInfluence(self, P, u, L):
        """
        A greedy method for the budgeted maximum influence method. We pick the
        index with maximum residual gain in activation divided by the cost, such
        that the total cost is still within budget. This algorithm has an unbounded
        approximation ratio. 
        """
        Parameter.checkFloat(L, 0.0, float('inf'))

        Q = (P.T/u).T
        numVertices = P.shape[0]
        bestActivations = numpy.zeros(numVertices)
        bestTotalActivation = 0

        selectedIndices = []
        unselectedIndices = set(range(0, numVertices))
        currentBudget = 0
        
        while True: 
            bestIndex = -1
            logging.debug("Budget remaining: " + str(L - currentBudget))

            for j in unselectedIndices:
                activations = numpy.max(numpy.r_['0,2', Q[j, :], bestActivations], 0)
                currentActivation = numpy.sum(activations)

                if currentActivation > bestTotalActivation and currentBudget + u[j] <= L:
                    bestIndex = j
                    bestTotalActivation = numpy.sum(currentActivation)

            if bestIndex == -1:
                break 

            bestActivations = numpy.max(numpy.r_['0,2', Q[bestIndex, :], bestActivations], 0)
            selectedIndices.append(bestIndex)
            unselectedIndices.remove(bestIndex)
            currentBudget = currentBudget + u[bestIndex]

        return selectedIndices