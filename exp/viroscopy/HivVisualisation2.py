
import logging
import sys
import matplotlib
matplotlib.use('WXAgg') # do this before importing pylab
import matplotlib.pyplot 
import networkx
import numpy
import wx
import math
from apgl.viroscopy.HIVGraphReader import HIVGraphReader
from apgl.util.GraphLayoutEngine import GraphLayoutEngine
from apgl.util.DateUtils import DateUtils

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

hivReader = HIVGraphReader()
graph = hivReader.readHIVGraph()

#The set of edges indexed by zeros is the contact graph
#The ones indexed by 1 is the infection graph
edgeTypeIndex1 = 0
edgeTypeIndex2 = 1
sGraphContact = graph.getSparseGraph(edgeTypeIndex1)
sGraphInfect = graph.getSparseGraph(edgeTypeIndex2)
sGraphContact = sGraphContact.union(sGraphInfect)

sGraph = sGraphContact
nxGraph = sGraph.toNetworkXGraph()

print(("Number of vertices: " + str(sGraph.getNumVertices())))
print(("Number of features: " + str(sGraph.getVertexList().getNumFeatures())))
print(("Number of edges: " + str(sGraph.getNumEdges())))

componentIndex = 0
components = networkx.connected_components(nxGraph)
nxGraph = networkx.subgraph(nxGraph , components[componentIndex])
print(("Number of components: " + str(len(components))))
print(("Size of biggest component: " + str(len(components[0]))))
print(("Size of new graph: " + str(nxGraph.number_of_nodes())))


nodeList = nxGraph.nodes()

#nodePositions = networkx.spring_layout(nxGraph)
#nodePositions = networkx.shell_layout(nxGraph)
#nodePositions = networkx.spectral_layout(nxGraph)
#nodePositions = networkx.graphviz_layout(nxGraph, prog="neato")

dobIndex = 0

dobs = numpy.array([float(sGraph.getVertex(v)[dobIndex]) for v in nodeList])
ages = 1 - dobs / dobs.max()
layoutEngine = GraphLayoutEngine()
nodePositions, allPositions, iterations = layoutEngine.layout(nxGraph, ages=None)

daysInYear = 365
monthsInYear = 12
monthLength = 30 
startDay = 86*daysInYear
endDay = 106*daysInYear

[1, 3, 5, 6, 8, 9, 10]

detectionIndex = 1
provinceIndex = 3
deathIndex = 4
genderIndex = 5
orientIndex = 6

nodeScale = 10

#The colors are given by the number of days after 1900, so the max is about 365*106
vmin = 0.0
vmax = 40000.0

alpha = 0.8
shapeDict = {0: "s", 1: "o"}

fig = matplotlib.pyplot.figure(figsize=(8,8))

def updateGraph(event):
    if updateGraph.i >= endDay:
        return False

    day = updateGraph.i
    graphNodeIndices = numpy.nonzero(sGraph.getVertexList().getVertices(nodeList)[:, detectionIndex] <= day)[0]
    #graphNodeIndices = numpy.nonzero(sGraph.getVertexList().getVertices(graphNodeIndices)[:, deathIndex] >= day)[0]

    tempNodeList = numpy.array(nodeList)[graphNodeIndices].tolist()
    tempGraph = networkx.subgraph(nxGraph, tempNodeList)
    tempNodeColour = [float(sGraph.getVertex(v)[dobIndex]) for v in tempNodeList]
    tempNodeSize = [float(tempGraph.degree(v)*nodeScale) for v in tempNodeList]

    if len(tempNodeList) != 0:
        matplotlib.pyplot.clf()
        networkx.draw_networkx(tempGraph, pos=nodePositions, node_size=tempNodeSize, node_color=tempNodeColour, nodelist=tempNodeList, vmin=vmin, vmax=vmax, alpha=alpha, labels=None, with_labels=False)
        matplotlib.pyplot.suptitle(str("Date: ") + DateUtils.getDateStrFromDay(updateGraph.i, 1900))
        #matplotlib.pyplot.axis([0, 1, 0, 1])
        matplotlib.pyplot.colorbar()
    #matplotlib.pyplot.axis([-500, 2000, -500, 2000])

    fig.canvas.draw()
    updateGraph.i = updateGraph.i + monthLength

updateGraph.i = startDay

id = wx.NewId()
actor = fig.canvas.manager.frame
timer = wx.Timer(actor, id=id)
timer.Start(3000)
wx.EVT_TIMER(actor, id, updateGraph)

matplotlib.pyplot.show()

#TODO: Make edges more apparent
#Notes  - mean connection time, optimise graph over all times, edge weights
#Cluster edges, diameter is important (number of paths) 