"""
A tree structure based on DictGraph
"""
import numpy 
from apgl.graph.DictGraph import DictGraph
from apgl.util.Parameter import Parameter

class DictTree(DictGraph):
    def __init__(self):
        """
        Create an empty tree. 
        """
        super(DictTree, self).__init__(False)
        self.rootVertex = None 

    def addEdge(self, vertex1Id, vertex2Id, value=1.0):
        """
        Add an edge from vertex1 to vertex2 with a given value.

        :param vertex1Id: The parent vertex name

        :param vertex2Id: The child vertex name

        :param value: The value along the edge. 
        """
        if len(self.neighbourOf(vertex2Id)) == 1 and self.neighbourOf(vertex2Id) != [vertex1Id]:
            raise ValueError("Vertex cannot have more than one parent")

        if self.getNumVertices()!=0 and not self.vertexExists(vertex1Id) and not self.vertexExists(vertex2Id):
            raise ValueError("Cannot add isolated edge")

        super(DictTree, self).addEdge(vertex1Id, vertex2Id, value)

    def addEdges(self, edgeList, edgeValues=None):
        """
        Add a set of edges to the tree

        :param edgeList: A list of pairs of vertex names
        :type edgeList: :class:`list`

        :param edgeValues: A list of corresponding vertex values
        :type edgeValues: :class:`list`
        """
        i = 0
        for edge in edgeList:
            (vertex1, vertex2) = edge
            if edgeValues == None:
                value = 1
            else:
                value = edgeValues[i]

            self.addEdge(vertex1, vertex2)

    def getRoot(self):
        """
        Find the root vertex. 
        """
        if self.getNumVertices() == 0:
            return None
        
        inDegSeq, vertices = self.inDegreeSequence()
        root = numpy.nonzero(inDegSeq==0)[0][0]
        return vertices[root]

    def setVertex(self, vertexName, vertex=None):
        """
        Assign a value to a vertex with given name

        :param vertexName: The name of the vertex.

        :param vertex: The value of the vertex.
        """
        if self.getNumVertices()==0 and not self.vertexExists(vertexName):
            super(DictTree, self).setVertex(vertexName, vertex)
        elif self.vertexExists(vertexName):
            super(DictTree, self).setVertex(vertexName, vertex)
        else:
            raise RuntimeError("Can only set a vertex in an empty tree: " + str(vertexName))

    def removeEdges(self, vertex1, vertex2):
        """
        This method is not currently implemented.
        """
        raise RuntimeError("Method not implemented")

    def depth(self):
        """
        Returns the depth which is the longest path length from the root to a
        leaf node.

        :returns: The depth of the tree.
        """
        if self.getNumVertices()==0:
            return 0 

        root = self.getRoot()
        stack = [(root, 0)]
        maxDepth = 0

        while(len(stack) != 0):
            (vertexId, depth) = stack.pop()

            if depth > maxDepth:
                maxDepth = depth 

            neighbours = self.neighbours(vertexId)

            for neighbour in neighbours:
                stack.append((neighbour, depth+1))

        return maxDepth 

    def cut(self, d):
        """
        Return a new tree containing all the vertices of the current one up to
        a depth of d. The edge and vertex labels are copied by reference only. 

        :param d: The depth of the new cut tree
        :type d: :class:`int`
        """
        Parameter.checkInt(d, 0, float("inf"))

        root = self.getRoot()
        newTree = DictTree()
        stack = [(root, 0)]

        newTree.setVertex(root)

        while(len(stack) != 0):
            (vertexId, depth) = stack.pop()
            neighbours = self.neighbours(vertexId)

            if depth <= d:
                newTree.setVertex(vertexId, self.getVertex(vertexId))

            for neighbour in neighbours:
                stack.append((neighbour, depth+1))

                if depth+1 <= d:
                    newTree.addEdge(vertexId, neighbour, self.getEdge(vertexId, neighbour))

        return newTree

    def leaves(self):
        """
        Return a list of the vertex ids of all the leaves of this tree.

        :returns: The vertex ids of the leaves. 
        """
        root = self.getRoot()
        stack = [(root, 0)]
        leafList = [] 

        while(len(stack) != 0):
            (vertexId, depth) = stack.pop()
            neighbours = self.neighbours(vertexId)

            if len(neighbours) == 0:
                leafList.append(vertexId)

            for neighbour in neighbours:
                stack.append((neighbour, depth+1))
                
        return leafList 

    def __str__(self):
        outputStr = super(DictTree, self).__str__() + "\n"
        root = self.getRoot()

        stack = [(root, 0)]

        while(len(stack) != 0):
            (vertexId, depth) = stack.pop()
            outputStr += "\t"*depth + str(vertexId) + ": " +  str(self.getVertex(vertexId)) + "\n"
            neighbours = self.neighbours(vertexId)

            for neighbour in neighbours:
                stack.append((neighbour, depth+1))

        return outputStr