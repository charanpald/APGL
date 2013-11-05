
from apgl.graph.AbstractVertexList import AbstractVertexList
from apgl.util.Parameter import Parameter
from apgl.util.Util import Util 

class GeneralVertexList(AbstractVertexList):
    """
    A GeneralVertexList is a list of vertices (immutable in size), in which any
    object can label each vertex. The underlying data structure is a dict. 
    """
    def __init__(self, numVertices):
        """
        Create an empty GeneralVertexList with the specified number of features
        for each vertex (initialised as None) and number of vertices.

        :param numVertices: The number of vertices.
        :type numVertices: :class:`int`
        """
        Parameter.checkInt(numVertices, 0, float('inf'))

        self.V = {}
        
        for i in range(numVertices):
            self.V[i] = None

    def getNumVertices(self):
        """
        Returns the number of vertices contained in this object.
        """
        return len(self.V)

    def getVertices(self, vertexIndices=None):
        """
        Returns a list of vertices specified by vertexIndices, or all vertices if
        vertexIndices == None. 

        :param vertexIndices: a list of vertex indices.
        :type vertexIndices: :class:`list`

        :returns: A set of vertices corresponding to the input indices. 
        """
        if vertexIndices != None:
            Parameter.checkList(vertexIndices, Parameter.checkIndex, (0, len(self.V)))
        else:
            vertexIndices = range(len(self.V))

        vertices = []
        for i in vertexIndices:
            vertices.append(self.V[i])

        return vertices

    def setVertex(self, index, value):
        """
        Set a vertex to the corresponding value.

        :param index: the index of the vertex to assign a value.
        :type index: :class:`int`

        :param value: the value to assign to the vertex.
        """
        Parameter.checkIndex(index, 0, len(self.V))

        self.V[index] = value

    def setVertices(self, vertices, indices=None):
        """
        Set the vertices to the given list of vertices. If indices = None then
        all vertices are replaced, and if not the given indices are used. 

        :param vertices: a list of vertices..
        :type vertices: :class:`list`

        :param indices: a list of indices of the same length as vertices or None for all indices in this object.
        :type indices: :class:`list`
        """
        if indices!=None:
            Parameter.checkList(indices, Parameter.checkIndex, [0, len(self.V)])
            if len(vertices) != len(indices):
                raise ValueError("Length of indices list must be same as that of vertices list")
        if indices==None and len(vertices) != len(self.V):
            raise ValueError("Incorrect number of vertices " + str(len(vertices)) + ", expecting " + str(len(self.V)))

        if indices == None:
            for i in range(len(vertices)):
                self.V[i] = vertices[i]
        else:
            for i in range(len(indices)):
                self.V[indices[i]] = vertices[i]

    def clearVertex(self, index):
        """
        Sets a vertex to None

        :param index: the index of the vertex to assign a value.
        :type index: :class:`int`
        """
        Parameter.checkIndex(index, 0, len(self.V))
        self.V[index] = None

    def getVertex(self, index):
        """
        Returns the value of a vertex.

        :param index: the index of the vertex.
        :type index: :class:`int`
        """
        Parameter.checkIndex(index, 0, len(self.V))
        return self.V[index]

    def __str__(self):
        """
        Returns the string representation of this object.
        """
        return self.V.__str__()

    def copy(self):
        """
        Returns a copy of this object.
        """
        vList = GeneralVertexList(len(self.V))
        vList.setVertices(list(self.V.values()))
        return vList

    def subList(self, indices):
        """
        Returns a subset of this object, indicated by the given indices.
        """
        Parameter.checkList(indices, Parameter.checkIndex, (0, self.getNumVertices()))
        vList = GeneralVertexList(len(indices))
        vList.setVertices(self.getVertices(indices))

        return vList

    def save(self, filename):
        """
        Save this object to filename.nvl.

        :param filename: The name of the file to save to.
        :type filename: :class:`str`

        :returns: The name of the saved file including extension.
        """
        Util.savePickle(self.V, filename + self.ext, overwrite=True)
        return filename + self.ext

    @staticmethod
    def load(filename):
        """
        Load this object from filename.pkl.

        :param filename: The name of the file to load.
        :type filename: :class:`str`
        """
        V = Util.loadPickle(filename + GeneralVertexList.ext)
        vList = GeneralVertexList(len(V))
        vList.V = V

        return vList
    
    def __len__(self): 
        return len(self.V)

    def addVertices(self, n): 
        """
        Adds n vertices to this object. 
        """
        oldN = len(self.V)
        for i in range(oldN, oldN+n):
            self.V[i] = None

    V = None
    ext = '.gvl'
