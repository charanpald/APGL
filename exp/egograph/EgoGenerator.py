'''
Created on 5 Jul 2009

@author: charanpal

Take the full set of egos by reading from a file and fit each feature to a normal
probability distribution. 
'''
from apgl.util.Util import Util

from apgl.graph.VertexList import VertexList
import numpy
import numpy.random as random
import logging 

class EgoGenerator:
    def __init__(self):
        pass 
    
    def generateIndicatorVertices(self, numVertices, mu, sigma, p):
        """ 
        Generate a set of vertices from the means and variances of Y using
        the multivariate normal distribution. Also add at
        the end an indicator variables which is 1 with probability p.
        """
        if not (sigma.T == sigma).all():
            raise ValueError("Must use symmetric sigma matrix: " + str(sigma))

        n = mu.shape[0]
        X = numpy.zeros((numVertices, n+1), numpy.int32)
        X[:, 0:n] = numpy.round(random.multivariate_normal(mu, sigma, numVertices)).astype(numpy.int32)
        X[:, n] = (random.rand(numVertices) < p).astype(numpy.int32)
        
        vList = VertexList(numVertices, n+1)
        
        for i in range(0, numVertices): 
            vList.setVertex(i, X[i, :])
            
        return vList 

    def generateIndicatorVertices2(self, numVertices, mu, sigma, p, minVals, maxVals):
        """
        Generate a set of vertices from the means and variances of Y using the multivariate
        normal distribution. Also add at the end an indicator variables which is 1
        with probability p. Make sure the values fall within minVals and maxVals. 
        """
        if  numpy.linalg.norm(sigma.T - sigma) > 10**-8 :
            raise ValueError("Must use symmetric sigma matrix: " + str(sigma.T - sigma))

        if (minVals > maxVals).any():
            raise ValueError("minVals must be less than or equal to maxVals")

        logging.info("Generating " + str(numVertices) + " vertices")

        n = mu.shape[0]
        X = numpy.round(random.multivariate_normal(mu, sigma, numVertices)).astype(numpy.int32)
        indVector = (random.rand(numVertices) < p).astype(numpy.int32)
        constantValueInds = numpy.nonzero(minVals == maxVals)[0]
        constantValues = minVals[constantValueInds]

        blockSize = 10000
        X = X[numpy.logical_and(X >= minVals, X <= maxVals).all(1), :]

        while X.shape[0] < numVertices:
            logging.info("Generated " + str(X.shape[0]) + " vertices so far")
            XBlock = numpy.round(random.multivariate_normal(mu, sigma, blockSize)).astype(numpy.int32)
            XBlock[:, constantValueInds] = constantValues
            XBlock = XBlock[numpy.logical_and(XBlock >= minVals, XBlock <= maxVals).all(1), :]

            X = numpy.r_[X, XBlock]

        X = X[0:numVertices, :]
        X = numpy.c_[X, indVector]

        vList = VertexList(numVertices, n+1)
        vList.setVertices(X)
        return vList

    """
    A third way can be to sample from a file - Fabrice can generate this. 
    """