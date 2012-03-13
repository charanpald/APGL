
"""
Do some clustering using the igraph method community_leading_eigenvector which
uses Newman's leading eigenvector method for detecting community structure.
"""
import time 
import logging

from apgl.graph import *
from apgl.util.Parameter import Parameter

class IterativeModularityClustering(object):
    def __init__(self, k):
        """
        Intialise this object with integer k which is the number of clusters.
        """
        Parameter.checkInt(k, 0, float('inf'))
        self.k = k

    def clusterFromIterator(self, graphListIterator, timeIter=False):
        """
        Find a set of clusters for the graphs given by the iterator. 
        """
        clustersList = []
        timeList = [] 

        for subW in graphListIterator:
            logging.debug("Clustering graph of size " + str(subW.shape))
            #Create a SparseGraph
            startTime = time.time()
            graph = SparseGraph(GeneralVertexList(subW.shape[0]))
            graph.setWeightMatrixSparse(subW)
            iGraph = graph.toIGraph()

            vertexCluster = iGraph.community_leading_eigenvector(self.k)
            clustersList.append(vertexCluster.membership)
            timeList.append(time.time()-startTime)

        if timeIter:
            return clustersList, timeList
        else:
            return clustersList

  