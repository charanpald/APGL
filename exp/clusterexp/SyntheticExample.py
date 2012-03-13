"""
Generate a synthetic sequence of graphs, and then cluster.
"""
import copy
import numpy
import logging
import sys
import itertools
import matplotlib.pyplot as plt
from apgl.util.PathDefaults import PathDefaults
from apgl.util.Util import Util
from apgl.graph import *
from apgl.generator import *
from apgl.sandbox.IterativeSpectralClustering import IterativeSpectralClustering
from apgl.sandbox.GraphIterators import IncreasingSubgraphListIterator, toDenseGraphListIterator
from apgl.sandbox.NingSpectralClustering import NingSpectralClustering

numpy.random.seed(21)
numpy.set_printoptions(suppress=True, precision=3, linewidth=200, threshold=40000)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

numClusters = 3
startClusterSize = 20
endClusterSize = 60
clusterStep = 5
#clusterStep = 20

numVertices = numClusters*endClusterSize
vList = GeneralVertexList(numVertices)

subgraphIndicesList = [range(0, startClusterSize)]
subgraphIndicesList[0].extend(range(endClusterSize, endClusterSize+startClusterSize))
subgraphIndicesList[0].extend(range(2*endClusterSize, 2*endClusterSize+startClusterSize))
for i in range(startClusterSize, endClusterSize-clusterStep+1, clusterStep):
	subgraphIndices = copy.copy(subgraphIndicesList[-1])
	subgraphIndices.extend(range(i, i+clusterStep))
	subgraphIndices.extend(range(endClusterSize+i, endClusterSize+i+clusterStep))
	subgraphIndices.extend(range(2*endClusterSize+i, 2*endClusterSize+i+clusterStep))
	subgraphIndicesList.append(subgraphIndices)

# to test removing
# - increasing graph
# do nothing
# - decreasing graph
#subgraphIndicesList.reverse()
# - increasing and decreasing graph
tmp = copy.copy(subgraphIndicesList[:-1])
tmp.reverse()
subgraphIndicesList.extend(tmp)

#print [len(x) for x in subgraphIndicesList]
k1 = numClusters
k2 = 20
clusterer = IterativeSpectralClustering(k1, k2)
ningsClusterer = NingSpectralClustering(k1)
T = 8 # index of iteration where exact decomposition is computed

#After 0.20 results are really bad 
ps = numpy.arange(0.05, 0.20, 0.05)
#ps = numpy.arange(0.05, 0.20, 0.1)
pClust = 0.3

perms = [l for l in itertools.permutations([0, 1, 2])]
numRepetitions = 50
#numRepetitions = 2
do_Nings = False

meanClustErrApprox = numpy.zeros((ps.shape[0], len(subgraphIndicesList)))
meanClustErrExact = numpy.zeros((ps.shape[0], len(subgraphIndicesList)))
meanClustErrNings = numpy.zeros((ps.shape[0], len(subgraphIndicesList)))

for r in range(numRepetitions):
	Util.printIteration(r, 1, numRepetitions)
	clustErrApprox = numpy.zeros((len(subgraphIndicesList), ps.shape[0]))
	clustErrExact = numpy.zeros((len(subgraphIndicesList), ps.shape[0]))

	for t in range(ps.shape[0]):
		logging.info("Run " + str(r) + "  p " + str(ps[t]))
		#Generate matrix of probabilities
		p = ps[t]
		W = numpy.ones((numVertices, numVertices))*p
		for i in range(numClusters):
			W[endClusterSize*i:endClusterSize*(i+1), endClusterSize*i:endClusterSize*(i+1)] = pClust
		P = numpy.random.rand(numVertices, numVertices)
		W = numpy.array(P < W, numpy.float)
		upTriInds = numpy.triu_indices(numVertices)
		W[upTriInds] = 0
		W = W + W.T
		graph = SparseGraph(vList)
		graph.setWeightMatrix(W)

		# run with exact eigenvalue decomposition
		logging.info("Running exact method")
		graphIterator = IncreasingSubgraphListIterator(graph, subgraphIndicesList)
		clustListExact = clusterer.clusterFromIterator(graphIterator, False)

		# run with our incremental approximation
		logging.info("Running approximate method")
		graphIterator = IncreasingSubgraphListIterator(graph, subgraphIndicesList)
		clustListApprox = clusterer.clusterFromIterator(graphIterator, True, T=T)

		# run with Ning's incremental approximation
		if do_Nings:
			logging.info("Running Nings method")
			graphIterator = IncreasingSubgraphListIterator(graph, subgraphIndicesList)
			clustListNings = ningsClusterer.cluster(toDenseGraphListIterator(graphIterator), T=T)

#		# print clusters
#		logging.info("learned clustering with exact eigenvalue decomposition")
#		for i in range(len(clustListExact)):
#			clusters = clustListExact[i]
#			print(clusters)
#		logging.info("learned clustering with our approximation approach")
#		for i in range(len(clustListApprox)):
#			clusters = clustListApprox[i]
#			print(clusters)
#		logging.info("learned clustering with Nings approximation approach")
#		for i in range(len(clustListNings)):
#			clusters = clustListNings[i]
#			print(clusters)

		# compute error for each iteration
		# error: proportion of pairs of vertices (x,y) s.t.
		#	(cl(x) == cl(y)) != (learned_cl(x) == learned_cl(y))
		for it in range(len(subgraphIndicesList)):
			indicesList = subgraphIndicesList[it]
			numUsedVertices = len(indicesList)
			# error with exact eigenvalue decomposition
			clusters = clustListExact[it]
			error = 0
			print clusters
			for v1 in range(numUsedVertices):
				for v2 in range(v1+1, numUsedVertices):
					same_cl = (indicesList[v1] / endClusterSize) == (indicesList[v2] / endClusterSize)
					same_learned_cl = clusters[v1] == clusters[v2]
					error += same_cl != same_learned_cl
			meanClustErrExact[t, it] += float(error)*2/(numUsedVertices)/(numUsedVertices-1)

			# error with our incremental approximation
			clusters = clustListApprox[it]
			error = 0
			for v1 in range(numUsedVertices):
				for v2 in range(v1+1, numUsedVertices):
					same_cl = (indicesList[v1] / endClusterSize) == (indicesList[v2] / endClusterSize)
					same_learned_cl = clusters[v1] == clusters[v2]
					error += same_cl != same_learned_cl
			meanClustErrApprox[t, it] += float(error)*2/(numUsedVertices)/(numUsedVertices-1)

			# error with Ning incremental approximation
			if do_Nings:
				clusters = clustListNings[it]
				error = 0
				for v1 in range(numUsedVertices):
					for v2 in range(v1+1, numUsedVertices):
						same_cl = (indicesList[v1] / endClusterSize) == (indicesList[v2] / endClusterSize)
						same_learned_cl = clusters[v1] == clusters[v2]
						error += same_cl != same_learned_cl
				meanClustErrNings[t, it] += float(error)*2/(numUsedVertices)/(numUsedVertices-1)

meanClustErrExact = meanClustErrExact/numRepetitions
meanClustErrApprox = meanClustErrApprox/numRepetitions
meanClustErrNings = meanClustErrNings/numRepetitions

print(meanClustErrExact)
print(meanClustErrApprox)
print(meanClustErrNings)


resultsDir = PathDefaults.getOutputDir() + "cluster/"
#Save results in a file
file_name = resultsDir + "ThreeClustErrors.dat"
try:
	res_file = open(file_name, 'w')
except(IOError), e:
	print "Warning: unable to open file '", file_name, "'\n", e
	print "=> results not saved"
else:
	comment = "# error for"
	for method in ["exact", "approx", "nings"]:
		for p in ps:
			comment = comment + " " + method + "_p" + str(p)
	res_file.write(comment + "\n")
	res = numpy.hstack((meanClustErrExact.T, meanClustErrApprox.T, meanClustErrNings.T))
	numpy.savetxt(res_file, res)
	


#Now lets plot the results
iterations = numpy.arange(startClusterSize, endClusterSize+1, clusterStep)*numClusters
plotStyles = {}
plotStyles[0] = ['ko-', 'kx-', 'k+-', 'k.-', 'k*-', 'ks-']
plotStyles[1] = ['ko--', 'kx--', 'k+--', 'k.--', 'k*--', 'ks--']
plotStyles[2] = ['ko:', 'kx:', 'k+:', 'k:', 'k*:', 'ks:']

plt.hold(True)
for i_res in range(3):
	res = [meanClustErrExact, meanClustErrApprox, meanClustErrNings][i_res]
	for i_p in range(len(ps)):
		plt.plot(iterations, res[i_p, :], plotStyles[i_res][i_p])
plt.xlabel("Number of Vertices")
plt.ylabel("Error")
plt.savefig(resultsDir + "ThreeClustErrors.eps")
#	plt.show()

# to run
# python -c "execfile('apgl/clusterexp/SyntheticExample.py')"

