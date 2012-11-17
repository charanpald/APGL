
import numpy
import scipy.io
from apgl.graph.AbstractVertexList import AbstractVertexList
from apgl.util.Util import Util
from apgl.util.Parameter import Parameter

class VertexList(AbstractVertexList):
    """
    A VertexList is a list of vertices (immutable in size), with a vector describing each vertex.
    """
    def __init__(self, numVertices, numFeatures=0, dtype=numpy.float):
        """
        Create an empty (zeroed) VertexList with the specified number of features
        for each vertex and number of vertices.

        :param numVertices: The number of vertices.
        :type numVertices: :class:`int`

        :param numFeatures: The number of features for each vertex.
        :type numFeatures: :class:`int`

        :param dtype: the data type for the vertex matrix, e.g numpy.int8.
        """
        Parameter.checkInt(numVertices, 0, float('inf'))
        Parameter.checkInt(numFeatures, 0, float('inf'))
        
        self.V = numpy.zeros((numVertices, numFeatures), dtype)
    
    def getNumVertices(self):
        """
        Returns the number of vertices contained in this object.
        """
        return self.V.shape[0]
    
    def getNumFeatures(self):
        """
        Returns the number of features of the vertices of this object.
        """
        return self.V.shape[1]

    #TODO: Make this function take indices 
    def setVertices(self, vertices):
        """
        Set the vertices to the given numpy array.

        :param vertices: a set of vertices of the same shape as this object. 
        :type vertices: :class:`numpy.ndarray`
        """
        if vertices.shape[0] != self.V.shape[0]:
            raise ValueError("Incorrect number of vertices " + str(vertices.shape[0]) + ", expecting " + str(self.V.shape[0]))
        if vertices.shape[1] != self.V.shape[1]:
            raise ValueError("Incorrect number of features " + str(self.V.shape[1]))
        
        self.V = vertices

    def replaceVertices(self, vertices):
        """
        Replace all the vertices within this class with a new set. Must have the
        same number vertices, but can alter the number of features.

        :param vertices: a set of vertices of the same number of rows as this object.
        :type vertices: :class:`numpy.ndarray`
        """

        if vertices.shape[0] != self.V.shape[0]:
            raise ValueError("Incorrect number of vertices " + str(vertices.shape[0]) + ", expecting " + str(self.V.shape[0]))

        self.V = vertices

    def getVertices(self, vertexIndices=None):
        """
        Returns a set of vertices specified by vertexIndices. If vertexIndices
        is None then all vertices are returned. 

        :param vertexIndices: a list of vertex indices.
        :type vertexIndices: :class:`list`

        :returns: A set of vertices corresponding to the input indices. 
        """
        if vertexIndices == None:
            return self.V
        else:
            Parameter.checkList(vertexIndices, Parameter.checkIndex, (0, self.V.shape[0]))
            return self.V[vertexIndices]

    def setVertex(self, index, value):
        """
        Set a vertex to the corresponding value.

        :param index: the index of the vertex to assign a value.
        :type index: :class:`int`

        :param value: the value to assign to the vertex.
        :type value: :class:`numpy.ndarray`
        """
        Parameter.checkIndex(index, 0, self.V.shape[0])
        Parameter.checkClass(value, numpy.ndarray)
        #Parameter.checkFloat(value, -float('inf'), float('inf'))
        if value.shape[0] != self.V.shape[1]:
            raise ValueError("All vertices must be arrays of length " + str(self.V.shape[1]))
        
        self.V[index, :] = value
        
    def clearVertex(self, index):
        """
        Sets a vertex to the all-zeros array.

        :param index: the index of the vertex to assign a value.
        :type index: :class:`int`
        """
        Parameter.checkIndex(index, 0, self.V.shape[0])
        self.V[index, :] = numpy.zeros((1, self.V.shape[1]))
        
    def getVertex(self, index):
        """
        Returns the value of a vertex. 

        :param index: the index of the vertex.
        :type index: :class:`int`

        :returns: the value of the vertex.
        """
        Parameter.checkIndex(index, 0, self.V.shape[0])
        return self.V[index, :]
    
    def getFeatureDistribution(self, fIndex, vIndices=None):
        """
        Returns a tuple (frequencies, items) about a particular feature given
        by fIndex. This method is depricated. 
        """
        Parameter.checkIndex(fIndex, 0, self.getNumFeatures())

        if vIndices == None:
            (freqs, items) = Util.histogram(self.V[:, fIndex])
        else:
            (freqs, items) = Util.histogram(self.V[vIndices, fIndex])
            
        return (freqs, items)  

    def __str__(self):
        """
        Returns the string representation of this object. 
        """
        return self.V.__str__()

    def copy(self):
        """
        Returns a copy of this object. 
        """
        vList = VertexList(self.V.shape[0], self.V.shape[1])
        vList.setVertices(numpy.copy(self.V))
        return vList

    def subList(self, indices):
        """
        Returns a subset of this object, indicated by the given indices.
        """
        Parameter.checkList(indices, Parameter.checkIndex, (0, self.getNumVertices()))
        vList = VertexList(len(indices), self.getNumFeatures())
        vList.setVertices(self.getVertices(indices))

        return vList 

    def save(self, filename):
        """
        Save this object to filename.nvl.

        :param filename: The name of the file to save.
        :type filename: :class:`str`

        :returns: The name of the saved file including extension.
        """
        file = open(filename + VertexList.ext, 'wb')
        scipy.io.mmwrite(file, self.V)
        file.close()
        
        return filename + VertexList.ext

    @staticmethod
    def load(filename):
        """
        Load this object from filename.nvl.

        :param filename: The name of the file to load.
        :type filename: :class:`str`
        """
        file = open(filename + VertexList.ext, 'rb')
        V = scipy.io.mmread(file)
        file.close()

        vList = VertexList(V.shape[0], V.shape[1])
        vList.V = V

        return vList

    def __getitem__(self, ind):
        """
        This is called when using numpy square bracket notation and returns the value
        of the specified vertex, e.g. vList[i, :] returns the ith vertex.

        :param ind: a vertex index
        :type ind: :class:`int`

        :returns: The value of the vertex.
        """
        return self.V[ind]

    def __setitem__(self, ind, value):
        """
        This is called when using square bracket notation and sets the value
        of the specified vertex, e.g. vList[i, :] = v.

        :param vertexIndex: a vertex index
        :type vertexIndex: :class:`int`

        :param value: the value of the vertex
        """
        self.V[ind] =  value

    def __len__(self): 
        return len(self.V)

    V = None
    ext = ".nvl"
    
