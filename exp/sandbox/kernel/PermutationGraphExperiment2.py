"""
We come up with a simple graph dataset. We come up with a set of random graphs
and then a paired graph is created by adding and removing a few edges, and also
permuting.

The aim is to find the corresponding pair - it is the one with the lowest
distance in the kenrel space. In essence it is a simple pairwise clustering task. 
"""
import numpy 
from apgl.graph.ErdosRenyiGenerator import ErdosRenyiGenerator
from apgl.graph.SparseGraph import SparseGraph
from apgl.graph.VertexList import VertexList
from apgl.kernel.PermutationGraphKernel import PermutationGraphKernel
from apgl.kernel.LinearKernel import LinearKernel
from apgl.kernel.KernelUtils import KernelUtils
from apgl.kernel.RandWalkGraphKernel import RandWalkGraphKernel
from apgl.util.Evaluator import Evaluator

ps = numpy.array(list(range(1, 5)), numpy.float64)/5
sizes = list(range(10, 100, 20))

print(("ps = " + str(ps)))
print(("sizes = " + str(sizes)))

graphsA = []
graphsB = []

numFeatures = 0
maxEdges = 100

for p in ps:
    for size in sizes:
        vList = VertexList(size, numFeatures)
        sGraph = SparseGraph(vList)

        generator = ErdosRenyiGenerator(sGraph)
        sGraph = generator.generateGraph(p)
        graphsA.append(sGraph)

        #Form graphB by shuffling edges and adding/deleting ones
        sGraph2 = SparseGraph(vList)
        graphsB.append(sGraph2)

        #Permute edges here
        inds = numpy.random.permutation(size)
        edges = sGraph.getAllEdges()

        for i in range(0, edges.shape[0]):
            sGraph2.addEdge(inds[edges[i, 0]], inds[edges[i, 1]])

        numExtraEdges = numpy.random.randint(0, maxEdges)

        for i in range(0, numExtraEdges):
            edgeIndex1 = numpy.random.randint(0, size)
            edgeIndex2 = numpy.random.randint(0, size)
            #print(str(edgeIndex1) + " " + str(edgeIndex2))
            sGraph2.addEdge(edgeIndex1, edgeIndex2)

#Check graphs are correct

#Now, compute the kernel matrix between all edges
graphs = graphsA
graphs.extend(graphsB)
numGraphs = len(graphs)

tau = 1.0
lmbda = 0.1
linearKernel = LinearKernel()
permutationKernel = PermutationGraphKernel(tau, linearKernel)
randomWalkKernel = RandWalkGraphKernel(lmbda)

K1 = numpy.zeros((numGraphs, numGraphs))
K2 = numpy.zeros((numGraphs, numGraphs))

for i in range(0, numGraphs):
    print(("i="+str(i)))
    for j in range(0, numGraphs):
        print(("j="+str(j)))
        K1[i, j] = permutationKernel.evaluate(graphs[i], graphs[j])
        K2[i, j] = randomWalkKernel.evaluate(graphs[i], graphs[j])

D1 = KernelUtils.computeDistanceMatrix(K1)
D2 = KernelUtils.computeDistanceMatrix(K2)

numPairs = numGraphs/2
windowSize = 3
pairIndices = numpy.array([list(range(numPairs)),  list(range(numPairs))]).T
pairIndices[:, 1] = numPairs + pairIndices[:, 1]

error1 = Evaluator.evaluateWindowError(D1, windowSize, pairIndices)
error2 = Evaluator.evaluateWindowError(D2, windowSize, pairIndices)

print(("Error 1: " + str(error1)))
print(("Error 2: " + str(error2)))
