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
            (vertexId1, vertexId2) = edge
            if edgeValues == None:
                value = 1
            else:
                value = edgeValues[i]

            self.addEdge(vertexId1, vertexId2, value)

    def getRootId(self):
        """
        Find the id of the root vertex. 
        """
        if self.getNumVertices() == 0:
            return None
        
        inDegSeq, vertices = self.inDegreeSequence()
        root = numpy.nonzero(inDegSeq==0)[0][0]
        return vertices[root]
        
    def getRoot(self):
        """
        Return the value of the root vertex. 
        """
        return self.getVertex(self.getRootId())

    def setVertex(self, vertexId, vertex=None):
        """
        Assign a value to a vertex with given name

        :param vertexId: The id of the vertex.

        :param vertex: The value of the vertex.
        """
        if self.getNumVertices()==0 and not self.vertexExists(vertexId):
            super(DictTree, self).setVertex(vertexId, vertex)
        elif self.vertexExists(vertexId):
            super(DictTree, self).setVertex(vertexId, vertex)
        else:
            raise RuntimeError("Can only set a vertex in an empty tree: " + str(vertexId))

    def addChild(self, parentId, childId, childVertex=None): 
        """
        This is basically a convenience function to allow one to easily add children. 
        """
        self.addEdge(parentId, childId)
        self.setVertex(childId, childVertex)

    def removeEdges(self, vertexId1, vertexId2):
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

        root = self.getRootId()
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

        root = self.getRootId()
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

    def nonLeaves(self, startVertexId=None):
        """
        Return a list of the vertex ids of all the non-leaves of this tree.

        :returns: The vertex ids of the non-leaves. 
        """
        if startVertexId == None: 
            subtreeIds = self.subtreeIds(self.getRootId())
        else:
            subtreeIds = self.subtreeIds(startVertexId)
        
        leafList = [] 

        for vertexId in subtreeIds: 
            neighbours = self.neighbours(vertexId)

            if len(neighbours) != 0:
                leafList.append(vertexId)
                
        return leafList 

    def leaves(self, startVertexId=None):
        """
        Return a list of the vertex ids of all the leaves of this tree. One can 
        specify a starting vertex id (if None, assume the root) and in this case, 
        we return the leaves of the corresponding subtree. 

        :param startVertexId: The vertex id of the subtree to find leaves of. 

        :returns: The vertex ids of the leaves. 
        """
                
        
        if startVertexId == None: 
            subtreeIds = self.subtreeIds(self.getRootId())
        else:
            subtreeIds = self.subtreeIds(startVertexId)
        
        leafList = [] 

        for vertexId in subtreeIds: 
            neighbours = self.neighbours(vertexId)

            if len(neighbours) == 0:
                leafList.append(vertexId)
                
        return leafList 

    def __str__(self):
        outputStr = super(DictTree, self).__str__() + "\n"
        root = self.getRootId()

        stack = [(root, 0)]

        while(len(stack) != 0):
            (vertexId, depth) = stack.pop()
            outputStr += "\t"*depth + str(vertexId) + ": " +  str(self.getVertex(vertexId)) + "\n"
            neighbours = self.neighbours(vertexId)

            for neighbour in neighbours:
                stack.append((neighbour, depth+1))

        return outputStr
        
    def children(self, vertexId): 
        """
        Returns the children of the current vertex. This is the same as neighbours. 
        
        :param startVertexId: The vertex id of the parent node.  
        """
        return self.neighbours(vertexId)

    
    def subtreeIds(self, vertexId): 
        """
        Return a list of all vertex ids that are descendants of this one, and include 
        this one. 
        
        :param vertexId: A vertex id 
        """
        stack = [(vertexId, 0)]
        subtreeList = [] 

        while(len(stack) != 0):
            (vertexId, depth) = stack.pop()
            neighbours = self.neighbours(vertexId)
            
            subtreeList.append(vertexId)

            for neighbour in neighbours:
                stack.append((neighbour, depth+1))
                
        return subtreeList 
        
    def pruneVertex(self, vertexId): 
        """
        Remove all the descendants of the current vertex. 
        
        :param vertexId: The vertex id of the parent node. 
        """
        subtreeIds = self.subtreeIds(vertexId) 
        
        for vertexId2 in subtreeIds:
            if vertexId != vertexId2: 
                self.removeVertex(vertexId2)
        
    def isLeaf(self, vertexId): 
        """
        Returns true if the input vertex id is a leaf otherwise false. 
        
        :param vertexId: The vertex id to test 
        """
        return len(self.neighbours(vertexId)) == 0
        
    def isNonLeaf(self, vertexId): 
        """
        Returns true if the input vertex id is not a leaf otherwise false. 
        
        :param vertexId: The vertex id to test 
        """
        return len(self.neighbours(vertexId)) != 0
        
    def copy(self): 
        """
        Return a copied version of this tree. This is a shallow copy in 
        that vertex values are not copied. 
        """
        newTree = DictTree()    
        newTree.adjacencies = {} 
        
        for key in self.adjacencies.keys():         
            newTree.adjacencies[key] = self.adjacencies[key].copy() 
            
        newTree.vertices = self.vertices.copy()
        
        return newTree
        
    def isSubtree(self, supertree): 
        """
        Test if this tree is a subtree of the input tree supertree. This is based 
        on the vertexIds and structure alone. 
        
        :param supertree: A DictTree object to compare against
        :type supertree: `apgl.graph.DictTree`
        """
        
        rootId = self.getRootId()
        vertexStack = [rootId]
        
        while len(vertexStack) != 0: 
            vertexId = vertexStack.pop()
            
            if not supertree.vertexExists(vertexId): 
                return False 
                
            children = self.neighbours(vertexId)
            for childId in children:
                if not self.edgeExists(vertexId, childId): 
                    return False 
                    
            vertexStack.extend(children)
        
        return True 
            