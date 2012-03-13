
from apgl.util import * 
from apgl.viroscopy.HIVGraphReader import HIVGraphReader
from apgl.graph import * 
import logging
import sys
import matplotlib
import numpy
from apgl.data.Standardiser import Standardiser
matplotlib.use('WXAgg') # do this before importing pylab
import matplotlib.pyplot as plt


"""
Let's compute some basic statistics of the infection and contact graph at the
end of the epidemic.
"""

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

figureDir = PathDefaults.getOutputDir() + "viroscopy/figures/"

undirected = False 
hivReader = HIVGraphReader()
graph = hivReader.readHIVGraph(undirected)
fInds = hivReader.getIndicatorFeatureIndices()

#The set of edges indexed by zeros is the contact graph
#The ones indexed by 1 is the infection graph
edgeTypeIndex1 = 0
edgeTypeIndex2 = 1
sGraphContact = graph.getSparseGraph(edgeTypeIndex1)
sGraphInfect = graph.getSparseGraph(edgeTypeIndex2)

numpy.set_printoptions(precision=3, suppress=True)

logging.info("Statistics over Verticies ")
logging.info("===============================")
#Other measures :  infection period of tree, infection types
#PCA to find variance of data, correlation matrix, center matrix
X = graph.getVertexList().copy().getVertices(list(range(0, graph.getNumVertices())))

standardiser = Standardiser()
X = standardiser.standardiseArray(X)
centerArray = standardiser.getCentreVector()

C = numpy.dot(X.T, X)

print((Latex.array2DToRows(numpy.reshape(centerArray, (5, 8)))))

C2 = numpy.abs(C - numpy.eye(X.shape[1]))
C2[numpy.tril_indices(C.shape[0])] = 0

inds = numpy.flipud(numpy.argsort(C2, None))
numEls = 10

for i in range(numEls):
    corr = "%.3f" % C[numpy.unravel_index(inds[i], C2.shape)]
    print((str(numpy.unravel_index(inds[i], C2.shape)) + " & " + corr + "\\\\"))


"""
Statistics over the infection graph.
"""

logging.info("Statistics over Infection Graph")
logging.info("===============================")

logging.info("Number of vertices: " + str(sGraphInfect.getNumVertices()))
logging.info("Number of features: " + str(sGraphInfect.getVertexList().getNumFeatures()))
logging.info("Number of edges: " + str(sGraphInfect.getNumEdges()))
logging.info("Density: " + str(sGraphInfect.density()))

#Find the largest tree
trees = sGraphInfect.findTrees()
treeSizes = [len(x) for x in trees]
treeCounts = numpy.bincount(treeSizes)

treeDepths = []
treeInfectionRanges = []

for i in range(len(trees)):
    treeGraph = sGraphInfect.subgraph(trees[i])
    treeGraphVertices = sGraphInfect.getVertexList().getVertices(list(trees[i]))
    treeInfectionRange = numpy.max(treeGraphVertices[:, fInds['detectDate']]) - numpy.min(treeGraphVertices[:, fInds['detectDate']])
    #treeDepths.append(treeGraph.diameter())
    treeDepths.append(GraphUtils.treeDepth(treeGraph))
    treeInfectionRanges.append(treeInfectionRange)

treeDepthsDists = numpy.bincount(treeDepths)
infectHist, infectBinEdges = numpy.histogram(treeInfectionRanges, 20)

logging.info("Frequency of trees: " + str(infectHist))
logging.info("Range of infection dates in trees: " + str(infectBinEdges))

logging.info("Frequency of trees: " + str(treeCounts[treeCounts!=0]))
logging.info("Size of trees: " + str(numpy.nonzero(treeCounts!=0)))

logging.info("Frequency of trees: " + str(treeDepthsDists[treeDepthsDists!=0]))
logging.info("Depths of trees: " + str(numpy.nonzero(treeDepthsDists!=0)))

logging.info("Out-degree distribution: " + str(sGraphInfect.degreeDistribution()))
logging.info("In-degree distribution: " + str(sGraphInfect.inDegreeDistribution()))

j = 1
plt.figure(j)
plt.plot(list(range(len(treeCounts))), treeCounts)
plt.xlabel("Tree sizes")
plt.ylabel("Frequency")
plt.savefig(figureDir + "InfectTreeSizes.eps")
j += 1

plt.figure(j)
plt.plot(list(range(len(treeDepthsDists))), treeDepthsDists)
plt.xlabel("Tree depths")
plt.ylabel("Frequency")
plt.savefig(figureDir + "InfectTreeDepths.eps")
j += 1

plt.figure(j)
plt.plot(infectBinEdges[1:], infectHist)
plt.xlabel("Infection ranges")
plt.ylabel("Frequency")
plt.savefig(figureDir + "InfectTreeInfectRanges.eps")
j += 1


"""
Statistics over contact graph 
"""
logging.info("Statistics over Contact Graph")
logging.info("===============================")

logging.info("Number of vertices: " + str(sGraphContact.getNumVertices()))
logging.info("Number of features: " + str(sGraphContact.getVertexList().getNumFeatures()))
logging.info("Number of edges: " + str(sGraphContact.getNumEdges()))
logging.info("Density: " + str(sGraphContact.density()))

#Find the largest tree
trees = sGraphContact.findTrees()
treeSizes = [len(x) for x in trees]
treeCounts = numpy.bincount(treeSizes)

treeDepths = []
treeInfectionRanges = []

for i in range(len(trees)):
    Util.printIteration(i, 10, len(trees))
    treeGraph = sGraphContact.subgraph(trees[i])
    treeGraphVertices = sGraphContact.getVertexList().getVertices(list(trees[i]))
    treeInfectionRange = numpy.max(treeGraphVertices[:, fInds['detectDate']]) - numpy.min(treeGraphVertices[:, fInds['detectDate']])
    #treeDepths.append(treeGraph.diameter())
    treeDepths.append(GraphUtils.treeDepth(treeGraph)) 
    treeInfectionRanges.append(treeInfectionRange)

treeDepthsDists = numpy.bincount(treeDepths)
infectHist, infectBinEdges = numpy.histogram(treeInfectionRanges, 20)

logging.info("Frequency of trees: " + str(infectHist))
logging.info("Range of infection dates in trees: " + str(infectBinEdges))

logging.info("Frequency of trees: " + str(treeCounts[treeCounts!=0]))
logging.info("Size of trees: " + str(numpy.nonzero(treeCounts!=0)))

logging.info("Frequency of trees: " + str(treeDepthsDists[treeDepthsDists!=0]))
logging.info("Depths of trees: " + str(numpy.nonzero(treeDepthsDists!=0)))

logging.info("Out-degree distribution: " + str(sGraphContact.degreeDistribution()))
logging.info("In-degree distribution: " + str(sGraphContact.inDegreeDistribution()))



plt.figure(j)
plt.plot(list(range(len(treeCounts))), treeCounts) 
plt.xlabel("Tree sizes")
plt.ylabel("Frequency")
plt.savefig(figureDir + "ContactTreeSizes.eps")
j += 1

plt.figure(j)
plt.plot(list(range(len(treeDepthsDists))), treeDepthsDists)
plt.xlabel("Tree depths")
plt.ylabel("Frequency")
plt.savefig(figureDir + "ContactTreeDepths.eps")
j += 1

plt.figure(j)
plt.plot(infectBinEdges[1:], infectHist)
plt.xlabel("Infection ranges")
plt.ylabel("Frequency")
plt.savefig(figureDir + "ContactTreeInfectRanges.eps")
j += 1

plt.show()
#Extend to other properties of trees - homogeniety of location, gender etc. 


