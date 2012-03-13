

from apgl.util.Parameter import Parameter
from apgl.graph.AbstractMatrixGraph import AbstractMatrixGraph
from apgl.generator.AbstractGraphGenerator import AbstractGraphGenerator
import numpy
import logging

class  BarabasiAlbertGenerator(AbstractGraphGenerator):
    '''
    A class which generates graphs according to the Barabasi-Albert Model. We start with
    ell vertices, and then add a new vertex at each iteration with m edges. The probability
    of attachment to vertex i is d_i / \sum_j d_j where d_i is the degree of the ith vertex.
    '''

    def __init__(self, ell, m):
        """
        Create a random graph generator according to the Barabasi-Albert Model

        :param ell: the initial number of vertices.
        :type ell: :class:`int`

        :param m: the number of edges to be added at each step
        :type m: :class:`int`
        """
        self.setEll(ell)
        self.setM(m)

    def setEll(self, ell):
        """
        :param ell: the initial number of vertices.
        :type ell: :class:`int`
        """
        Parameter.checkInt(ell, 2, float('inf'))
        self.ell = ell 

    def setM(self, m):
        """
        :param m: the number of edges to be added at each step
        :type m: :class:`int`
        """
        Parameter.checkInt(m, 0, self.ell)
        self.m = m 

    def generate(self, graph):
        """
        Create a random graph using the input graph according to the Barabasi-Albert
        model. Note that the input graph is modified.

        :param graph: the empty input graph.
        :type graph: :class:`apgl.graph.AbstractMatrixGraph`

        :returns: The modified input graph. 
        """
        Parameter.checkClass(graph, AbstractMatrixGraph)
        if graph.getNumEdges()!= 0:
            raise ValueError("Graph must have no edges")
        
        #Keep a list of node indices with degrees
        #First start off with ell vertices assume they each have degree 1, without
        #adding edges. This is a bit weird but seems the way to do it. 
        vertexList = list(range(0, self.ell))

        #Now perform preferential attachment, making sure we add m edges at each
        #iteration. 
        for i in range(self.ell, graph.getNumVertices()):
            perm = numpy.random.permutation(len(vertexList))
            numEdgesAdded = 0
            j = 0 

            while numEdgesAdded != self.m:
                ind = perm[j]
                vertexIndex = vertexList[ind]
                
                if graph.getEdge(i, vertexIndex) == None:
                    graph.addEdge(i, vertexIndex)
                    vertexList.append(i)
                    vertexList.append(vertexIndex)
                    numEdgesAdded += 1
                
                j = j+1

        return graph

    def __str__(self):
        return "BarabasiAlbertGenerator_ell="+str(self.ell)+",m="+str(self.m)