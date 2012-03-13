

from apgl.io.GraphWriter import GraphWriter
import logging

class SimpleGraphWriter(GraphWriter):
    '''
    A class to output all edges of a graph in a simple text format
    '''
    def __init__(self):
        #Map from the IDs given in the graph to 0 ... n
        self.vertexIdDict = {}

    def writeToFile(self, fileName, graph):
        """
        Write vertices and edges of the graph in a text format to the given
        file name. The file has a first line "Vertices" followed by a list of
        vertex indices (one per line). Then the lines following "Arcs" or "Edges"
        have a list of pairs of vertex indices represented directed or undirected
        edges. 
        """

        self.vertexIdDict = {}
        fileName = fileName + ".txt"
        numVertices = graph.getNumVertices()
        index = 0

        f = open(fileName, 'w')
        f.write("Vertices\n")
        logging.info('Writing to SimpleGraph file: ' + fileName)

        for i in graph.getAllVertexIds():
            self.vertexIdDict[i] = index
            vertexString = str(index) + "\n"
            f.write(vertexString)
            index += 1

        if graph.isUndirected():
            f.write("Edges\n")
        else:
            f.write("Arcs\n")
        f.write(self.__getArcString(graph))

        f.close()
        logging.info("Finished, wrote " + str(numVertices) + " vertices & " + str(graph.getNumEdges()) + " edges.")

    def __getArcString(self, graph):
        arcString = ""

        for edge in graph.getAllEdges():
            vertex1 = edge[0]
            vertex2 = edge[1]

            index1 = self.vertexIdDict[vertex1]
            index2 = self.vertexIdDict[vertex2]
            arcString = arcString + str(index1) + ", " + str(index2) + ", " + str(graph.getEdge(vertex1, vertex2)) + "\n"

        return arcString