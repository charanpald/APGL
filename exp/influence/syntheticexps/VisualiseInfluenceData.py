import numpy
import logging
import random
import sys
import matplotlib
#matplotlib.use('WXAgg') # do this before importing pylab
import matplotlib.pyplot 
import networkx
from apgl.graph import *
from apgl.util import *
from apgl.influence.GreedyInfluence import GreedyInfluence

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

random.seed(21)
numpy.random.seed(21)
outputDir = PathDefaults.getOutputDir() + "influence/"

numFeatures = 1
numVertices = 20
vList = VertexList(numVertices, numFeatures)
graph = SparseGraph(vList, False)

p = 0.05
generator = ErdosRenyiGenerator(graph)
graph = generator.generateGraph(p) 

allEdges = graph.getAllEdges()
edgeValues = numpy.random.rand(allEdges.shape[0])
graph.addEdges(allEdges, edgeValues)

logging.info("Graph edges " + str(graph.getNumEdges()) + " vertices " + str(graph.getNumVertices()))
nxGraph = graph.toNetworkXGraph()

#Compute most influencial nodes
k = 5
P = graph.maxProductPaths()
P = P + numpy.eye(numVertices)
P2 = numpy.array(P != 0, numpy.int32)
influence = GreedyInfluence()
inds = numpy.array(influence.maxInfluence(P, k))
inds2 = numpy.array(influence.maxInfluence(P2, k))

print(P)
print(P2)

print(inds)
print(inds2)

print((numpy.sum(numpy.max(P[inds, :], 0))))
print((numpy.sum(numpy.max(P[inds2, :], 0))))


#Decide on edge properties in visualisation 
edgeColours = graph.getEdgeValues(graph.getAllEdges())
edgeVmin = numpy.min(edgeColours)
edgeVmax = numpy.max(edgeColours)
logging.info("Min edge value " + str(edgeVmin))
logging.info("Max edge value " + str(edgeVmax))

nodeSizes = numpy.ones(graph.getNumVertices())*200
nodeSizes[inds] = 400
nodeColours = nodeSizes/400

nodeSizes2 = numpy.ones(graph.getNumVertices())*200
nodeSizes2[inds2] = 400
nodeColours2 = nodeSizes2/400

nodeVmin = 0.5
nodeVmax = 2.0
#nodePos = networkx.spring_layout(nxGraph, iterations=200, weighted=False)
nodePos = networkx.graphviz_layout(nxGraph)

matplotlib.pyplot.figure(1, figsize=(8, 8))
networkx.draw_networkx(nxGraph, width=2.0, with_labels=True,  vmin=nodeVmin, vmax=nodeVmax, node_color=nodeColours, edge_vmin=edgeVmin, edge_vmax=edgeVmax, pos=nodePos,node_size=nodeSizes, cmap=matplotlib.pyplot.cm.Blues, edge_color=edgeColours, edge_cmap=matplotlib.pyplot.cm.hsv)
ax = matplotlib.pyplot.axes()
ax.set_xticks([])
ax.set_yticks([])
matplotlib.pyplot.colorbar()

matplotlib.pyplot.figure(2, figsize=(8, 8))
networkx.draw_networkx(nxGraph, width=2.0, with_labels=True,  vmin=nodeVmin, vmax=nodeVmax, node_color=nodeColours2, edge_vmin=edgeVmin, edge_vmax=edgeVmax, pos=nodePos,node_size=nodeSizes2, cmap=matplotlib.pyplot.cm.Blues, edge_color=edgeColours, edge_cmap=matplotlib.pyplot.cm.hsv)
ax = matplotlib.pyplot.axes()
ax.set_xticks([])
ax.set_yticks([])
matplotlib.pyplot.colorbar()

matplotlib.pyplot.show()

#Final thing- try to print edge labels 
