
"""
This is just a toy example on a made up graph which can be visualised.
"""
from apgl.graph import * 
from exp.sandbox.IterativeSpectralClustering import *
from exp.sandbox.GraphIterators import * 

numVertices = 14
numFeatures = 0

vList = VertexList(numVertices, numFeatures)
graph = SparseGraph(vList)

graph.addEdge(0, 1)
graph.addEdge(0, 2)
graph.addEdge(1, 2)

graph.addEdge(3, 4)
graph.addEdge(3, 5)
graph.addEdge(4, 5)

graph.addEdge(6, 7)
graph.addEdge(6, 8)
graph.addEdge(7, 7)

graph.addEdge(1, 4)
graph.addEdge(1, 6)

graph.addEdge(4, 9)
graph.addEdge(5, 9)
graph.addEdge(9, 10)
graph.addEdge(5, 10)

graph.addEdge(11, 0)
graph.addEdge(11, 1)
graph.addEdge(11, 2)
graph.addEdge(7, 12)
graph.addEdge(8, 12)
graph.addEdge(12, 13)

subgraphIndicesList = []
subgraphIndicesList.append(range(9))
subgraphIndicesList.append(range(11))
subgraphIndicesList.append(range(14))

k1 = 3
k2 = 5
clusterer = IterativeSpectralClustering(k1, k2)
#Test full computation of eigenvectors
graphIterator = IncreasingSubgraphListIterator(graph, subgraphIndicesList)
clustersList = clusterer.clusterFromIterator(graphIterator, True)
#clustersList = clusterer.cluster(graph, subgraphIndicesList, True)

print(clustersList)

#Seems to work fine and same in exact case, but not very interesting
#End clustering: array([1, 1, 1, 0, 0, 0, 2, 2, 2, 0, 0, 1, 2, 2])]
