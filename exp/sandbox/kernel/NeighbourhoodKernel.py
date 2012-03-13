from apgl.kernel.VertexKernel import VertexKernel

class NeighbourhoodKernel(VertexKernel):
    def __init__(self, r):
        self.r = r

    def computeNeighbourhoodGraphs(self, graph):
        """
        Get all neighbourhood graphs from a given vertex of radius r, and return
        the list of all subGraphs as a set of indices.
        """
        P = graph.floydWarshall(useWeights=False)

        #The ith element is the list of all subgraphs of radius <= r excluding current
        subGraphList = []

        for i in range(graph.getNumVertices()):
            currentGraph = set((P[i, :] <= self.r ).nonzero()[0].tolist())
            currentGraph.add(i)
            subGraphList.append(currentGraph)

        return subGraphList

    def computeNeighbourhoodKernel(self, graph):
        """
        The number of isomorphic graph neighbours is counted between vertices within
        the input graph, for a radius r.
        TODO: Test this method
        """

        numVertices = graph.getNumVertices()
        K = numpy.zeros((numVertices, numVertices))
        subGraphList = self.computeNeighbourhoodGraphs(graph, self.r)

        for i in range(numVertices):
            K[i, i] = 1
            subgraph1 = graph.subgraph(subGraphList[i])
            for j in range(i+1, numVertices):
                subgraph2 = graph.subgraph(subGraphList[j])

                K[i, j] = subgraph1.maybeIsomorphicWith(subgraph2)
                K[j, i] = K[i, j]

        return K

    def evaluate(self, graph, vIndices1, vIndices2):
        K = self.computeNeighbourhoodKernel(graph)
        return K[vIndices1, vIndices2]