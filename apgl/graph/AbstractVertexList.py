
from apgl.util import * 

class AbstractVertexList(object):
    """
    An AbstractVertexList is a list of vertices (immutable in size), indexed
    by integers starting from zero.
    """

    def getVertices(self, vertexIndices):
        """
        Returns a list of vertices specified by vertexIndices.

        :param vertexIndices: a list of vertex indices.
        """
        Util.abstract()

    def setVertices(self, vertices):
        """
        Set the vertices to the given list of vertices.

        :param vertices: a set of vertices of the same shape as this object.
        """
        Util.abstract()

    def getVertex(self, index):
        """
        Returns the value of a vertex.

        :param index: the index of the vertex.
        :type index: :class:`int`
        """
        Util.abstract()

    def setVertex(self, index, value):
        """
        Set a vertex to the corresponding value.

        :param index: the index of the vertex to assign a value.
        :type index: :class:`int`

        :param value: the value to assign to the vertex.
        """
        Util.abstract()

    def getNumVertices(self):
        """
        Returns the number of vertices contained in this object.
        """
        Util.abstract()

    def subList(self, indices):
        """
        Returns a subset of this object, indicated by the given indices.
        """
        Util.abstract()

    def copy(self):
        """
        Returns a copy of this object.
        """
        Util.abstract()

    def clearVertex(self, index):
        """
        Sets a vertex to None

        :param index: the index of the vertex to assign a value.
        :type index: :class:`int`
        """
        Util.abstract()

    def save(self, filename):
        """
        Save this object to filename.nvl.

        :param filename: The name of the file to save.
        :type filename: :class:`str`
        """
        Util.abstract()

    @staticmethod
    def load(filename):
        """
        Load this object from filename.

        :param filename: The name of the file to load.
        :type filename: :class:`str`
        """
        Util.abstract()
        
    def __getitem__(self, vertexIndex):
        """
        This is called when using square bracket notation and returns the value
        of the specified vertex, e.g. vList[i] returns the ith vertex.

        :param vertexIndices: a vertex index
        :type vertexIndices: :class:`int`

        :returns: The value of the vertex.
        """
        return self.getVertex(vertexIndex)

    def __setitem__(self, vertexIndex, value):
        """
        This is called when using square bracket notation and sets the value
        of the specified vertex, e.g. vList[i] = v.

        :param vertexIndex: a vertex index
        :type vertexIndex: :class:`int`

        :param value: the value of the vertex
        """
        self.setVertex(vertexIndex, value)