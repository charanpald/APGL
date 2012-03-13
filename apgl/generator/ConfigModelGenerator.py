
from apgl.util.Parameter import Parameter
from apgl.util.Util import Util
from apgl.graph.AbstractMatrixGraph import AbstractMatrixGraph
from apgl.generator.AbstractGraphGenerator import AbstractGraphGenerator
import numpy
import math 

class ConfigModelGenerator(AbstractGraphGenerator):
    '''
    A class which generates graphs according to the configuration model. In this model
    one specifies a degree sequence and the generated graph fits the sequence as closely
    as possible. 
    '''
    def __init__(self, outDegSequence, inDegSequence=None):
        '''
        Create a ConfigModelGenerator object with the given degree sequence. If inDegreeSequence
        is None then we assume an undirected graph, otherwise it is directed. Note that the sum
        of the values in inDegSequence and outDegSequence should ideally be the same to avoid
        unconnected spokes. 

        :param outDegSequence: a vector of (out)degrees for each vertex in the graph.
        :type outDegSequence: :class:`numpy.ndarray`

        :param inDegSequence: a vector of in-degrees for each vertex in the graph or None for undirected graphs. 
        :type inDegSequence: :class:`numpy.ndarray`

        '''
        self.setOutDegSequence(outDegSequence)
        if inDegSequence != None:
            self.setInDegSequence(inDegSequence)
        else:
            self.inDegSequence = None 

    def setOutDegSequence(self, outDegSequence):
        '''
        Set the (out)degree sequence of this object.

        :param outDegSequence: a vector of degrees for each vertex in the graph.
        :type outDegSequence: :class:`numpy.ndarray`
        '''
        Parameter.checkClass(outDegSequence, numpy.ndarray)
        if outDegSequence.ndim != 1:
            raise ValueError("Degree sequence must be one dimensional")
        Parameter.checkList(outDegSequence, Parameter.checkInt, [0, outDegSequence.shape[0]])

        self.outDegSequence = outDegSequence

    def setInDegSequence(self, inDegSequence):
        '''
        Set the (in)degree sequence of this object.

        :param inDegSequence: a vector of degrees for each vertex in the graph.
        :type inDegSequence: :class:`numpy.ndarray`
        '''
        Parameter.checkClass(inDegSequence, numpy.ndarray)
        if inDegSequence.ndim != 1:
            raise ValueError("Degree sequence must be one dimensional")
        if inDegSequence.shape[0] != self.outDegSequence.shape[0]:
            raise ValueError("In-degree sequence must be same length as out-degree sequence")
        Parameter.checkList(inDegSequence, Parameter.checkInt, [0, inDegSequence.shape[0]])

        self.inDegSequence = inDegSequence

    def getOutDegSequence(self):
        """
        :returns: A vector of integers corresponding to the (out)degree sequence.
        """
        return self.outDegSequence

    def getInDegSequence(self):
        """
        :returns: A vector of integers corresponding to the (in)degree sequence.
        """
        return self.inDegSequence

    def generate(self, graph, requireEmpty=True):
        '''
        Create an Configuration Model graph. Note the the degree sequence(s) given
        in the constructor cannot be guarenteed. The algorithm randomly selects
        two free "spokes" and then tried to connect them. If two vertices are
        already connected the corresponding spokes are not used again. In the case
        that requireEmpty is False then a non-empty graph can be used and the given
        degree sequence(s) is(are) the difference(s) in degrees between the output graph and
        input one. 

        :param graph: a graph to populate with edges
        :type graph: :class:`apgl.graph.AbstractMatrixGraph`

        :param requireEmpty: if this is set to true then we require an empty graph.
        :type requireEmpty: :class:`bool`

        :returns: The modified input graph. 
        '''
        Parameter.checkClass(graph, AbstractMatrixGraph)
        if requireEmpty and graph.getNumEdges()!= 0:
            raise ValueError("Graph must have no edges")
        if graph.getNumVertices() != self.outDegSequence.shape[0]:
            raise ValueError("Graph must have same number of vertices as degree sequence")
        if self.getInDegSequence()!=None and graph.isUndirected():
            raise ValueError("In-degree sequence must be used in conjunction with directed graphs")

        if self.getInDegSequence()==None:
            expandedInds = Util.expandIntArray(self.outDegSequence)
            numpy.random.shuffle(expandedInds)
            for i in range(0, len(expandedInds), 2):
                if i != len(expandedInds)-1:
                    graph.addEdge(expandedInds[i], expandedInds[i+1])
        else:
            expandedOutInds = Util.expandIntArray(self.outDegSequence)
            expandedInInds = Util.expandIntArray(self.inDegSequence)
            numpy.random.shuffle(expandedOutInds)
            numpy.random.shuffle(expandedInInds)

            for i in range(numpy.min(numpy.array([expandedOutInds.shape[0], expandedInInds.shape[0]]))):
                graph.addEdge(expandedOutInds[i], expandedInInds[i])

        return graph

