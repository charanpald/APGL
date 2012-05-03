"""
Generate a synthetic sequence of graphs, and then cluster.
"""
import numpy
import scipy
import logging
import sys
import itertools
import matplotlib.pyplot as plt
from apgl.util.PathDefaults import PathDefaults
from apgl.util.Util import Util
from apgl.graph import *
from apgl.generator import *
from exp.sandbox.IterativeSpectralClustering import IterativeSpectralClustering
from exp.sandbox.GraphIterators import IncreasingSubgraphListIterator, toDenseGraphListIterator
from exp.sandbox.NingSpectralClustering import NingSpectralClustering

numpy.seterr(all="raise", under="ignore")
numpy.random.seed(21)
numpy.set_printoptions(suppress=True, precision=3, linewidth=200, threshold=40000)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


# at each lvl, clusters are splits in $numSubClustersPerLevel$ parts
# at each iterations, edges are added. With time, probability to pick an edge
# in a high lvl cluster increase, while the probability decreases for low lvl
# clusters.
_numLevel = 3
_numSubClustersPerLevel = 3
_maxP = 0.01
class GrowingContrastGraphIterator(object):
	def __init__(self):
		self.numLevel = _numLevel
		self.numSubClustersPerLevel = _numSubClustersPerLevel
		self.numSmallestCluster = self.numSubClustersPerLevel**(self.numLevel-1)
		self.numVertices = self.numSmallestCluster*20
		self.minP = 0.0005
		self.maxP = _maxP
		self.deltaP = (self.maxP-self.minP)/(self.numLevel-1)
		self.iter = 0
		# level matrix
		self.levelMatrix = numpy.zeros((self.numVertices, self.numVertices))
		self.fillLevelMatrix(0, self.numVertices, 1)
		# edges matrix
		self.edgesMatrix = scipy.sparse.csr_matrix((self.numVertices, self.numVertices))
		
	def fillLevelMatrix(self, i, size, lvl):
		if lvl == self.numLevel:
			return
		subSize = size/self.numSubClustersPerLevel
		for ii in range(i, i+size, subSize):
			j = ii + subSize
			self.levelMatrix[ii:j,ii:j] = lvl * numpy.ones((subSize, subSize))
			self.fillLevelMatrix(ii, subSize, lvl+1)

	def __iter__(self):
		return self

	def next(self):
		# new edges
		numTry = 100
		for dev_null in range(0, numTry):
			R = numpy.random.rand(self.numVertices, self.numVertices)
			P = (self.minP + self.levelMatrix *self.deltaP)*self.iter
			newEdges = numpy.array(R < P, dtype=float)
			newEdges = numpy.array(newEdges - self.edgesMatrix.todense() > 0, dtype=float)
			newEdges = numpy.array( (newEdges + newEdges.T)>0, dtype=float)
			self.edgesMatrix = self.edgesMatrix + scipy.sparse.csr_matrix(newEdges)
			if numpy.count_nonzero(newEdges) != 0:
				break
		else: # if leaving loop normaly
			logging.warning("GrowingContrastGraphIterator did not add new edges in " + str(numTry) + " draws")
#			print P
#			print newEdges
#			print self.edgesMatrix*7 +1
#			print sum(newEdges.ravel()), sum(newEdges.ravel()) == 0.0
		self.iter += 1
		return self.edgesMatrix




#===========================================
# cluster

k1 = 9 # numCluster to learn
k2 = 8*k1 # numEigenVector kept
startingIter = 3
endIter = 23
numIter = endIter - startingIter

#Variables to choose which methods to run
runIASC = True
runExact = True
runNing = False

numRepetitions = 50
#numRepetitions = 2

print "compute clusters"
clusterer = IterativeSpectralClustering(k1, k2)
ningsClusterer = NingSpectralClustering(k1)

def getGraphIterator():
	return itertools.islice(GrowingContrastGraphIterator(), startingIter, endIter)

for r in range(numRepetitions):
	logging.info("run " + str(r))
	
	if runExact:
		# run with exact eigenvalue decomposition
		logging.info("Running exact method")
		graphIterator = getGraphIterator()
		clustersExact = clusterer.clusterFromIterator(graphIterator, False)

	if runIASC:
		# run with our incremental approximation
		logging.info("Running approximate method")
		graphIterator = getGraphIterator()
		clustListApprox = clusterer.clusterFromIterator(graphIterator, True, T=1000)

	if runNing:
		# run with Ning's incremental approximation
		logging.info("Running Nings method")
		graphIterator = getGraphIterator()
		clustListNings = ningsClusterer.cluster(toDenseGraphListIterator(graphIterator))

	# print clusters
	if runExact:
		logging.info("learned clustering with exact eigenvalue decomposition")
		for i in range(len(clustersExact)):
			clusters = clustersExact[i]
			print(clusters)
	if runIASC:
		logging.info("learned clustering with our approximation approach")
		for i in range(len(clustListApprox)):
			clusters = clustListApprox[i]
			print(clusters)
	if runNing:
		logging.info("learned clustering with Nings approximation approach")
		for i in range(len(clustListNings)):
			clusters = clustListNings[i]
			print(clusters)

	# compute error for each iteration and lvl
	# error: proportion of pairs of vertices (x,y) s.t.
	#    (cl(x) == cl(y)) != (learned_cl(x) == learned_cl(y))
	if not 'meanClustErrExact' in locals():
		meanClustErrExact = numpy.zeros((_numLevel, numIter))
		meanClustErrApprox = numpy.zeros((_numLevel, numIter))
		meanClustErrNings = numpy.zeros((_numLevel, numIter))
		if runExact:
			numVertices = clustersExact[0].size
		if runIASC:
			numVertices = clustListApprox[0].size
		if runNing:
			numVertices = clustListNings[0].size
			

	clust_size = numVertices # number of vertices per cluster
	for lvl in range(_numLevel):

		if runExact:
			# error with exact eigenvalue decomposition
			for it in range(numIter):
				clusters = clustersExact[it]
				error = 0
				for v1 in range(numVertices):
					for v2 in range(v1+1, numVertices):
						same_cl = (v1 / clust_size) == (v2 / clust_size)
						same_learned_cl = clusters[v1] == clusters[v2]
						error += same_cl != same_learned_cl
				meanClustErrExact[lvl, it] += float(error)*2/(numVertices)/(numVertices-1)

		if runIASC:
			# error with our incremental approximation
			for it in range(numIter):
				clusters = clustListApprox[it]
				error = 0
				for v1 in range(numVertices):
					for v2 in range(v1+1, numVertices):
						same_cl = (v1 / clust_size) == (v2 / clust_size)
						same_learned_cl = clusters[v1] == clusters[v2]
						error += same_cl != same_learned_cl
				meanClustErrApprox[lvl, it] += float(error)*2/(numVertices)/(numVertices-1)

		if runNing:
			# error with Ning incremental approximation
			for it in range(numIter):
				clusters = clustListNings[it]
				error = 0
				for v1 in range(numVertices):
					for v2 in range(v1+1, numVertices):
						same_cl = (v1 / clust_size) == (v2 / clust_size)
						same_learned_cl = clusters[v1] == clusters[v2]
						error += same_cl != same_learned_cl
				meanClustErrNings[lvl, it] += float(error)*2/(numVertices)/(numVertices-1)

		# update variables related to lvl
		clust_size /= _numSubClustersPerLevel

if runExact:
	meanClustErrExact = meanClustErrExact/numRepetitions
	print(meanClustErrExact)
if runIASC:
	meanClustErrApprox = meanClustErrApprox/numRepetitions
	print(meanClustErrApprox)
if runNing:
	meanClustErrNings = meanClustErrNings/numRepetitions
	print(meanClustErrNings)



resultsDir = PathDefaults.getOutputDir() + "cluster/"
#Save results in a file
file_name = resultsDir + "IncreasingContrastClustErrors_pmax" + str(_maxP) + "_nEigen" + str(k2) + ".dat"
try:
	res_file = open(file_name, 'w')
except(IOError), e:
	print "Warning: unable to open file '", file_name, "'\n", e
	print "=> results not saved"
else:
	res_file.write("# error for exact_lvl0 exact_lvl1 exact_lvl2 approx_lvl0 approx_lvl1 approx_lvl2 nings_lvl0 nings_lvl1 nings_lvl2\n")
	res = numpy.hstack((meanClustErrExact.T, meanClustErrApprox.T, meanClustErrNings.T))
	numpy.savetxt(res_file, res)
	


#Now lets plot the results
iterations = numpy.arange(startingIter, endIter)
plotStyles = ['ko-', 'kx-', 'k+-', 'k.-', 'k*-', 'ks-']
plotStyles = ['ko-', 'kx-', 'k+-', 'k.-', 'k*-', 'ks-']
plotStyles2 = ['ko--', 'kx--', 'k+--', 'k.--', 'k*--', 'ks--']
plotStyles2 = ['ko--', 'kx--', 'k+--', 'k.--', 'k*--', 'ks--']
plotStyles3 = ['ko:', 'kx:', 'k+:', 'k:', 'k*:', 'ks:']

for lvl in range(_numLevel):
	plt.hold(False)
	if runExact:
		plt.plot(iterations, meanClustErrExact[lvl, :], plotStyles[0])
		plt.hold(True)
	if runIASC:
		plt.plot(iterations, meanClustErrApprox[lvl, :], plotStyles2[0])
		plt.hold(True)
	if runNing:
		plt.plot(iterations, meanClustErrNings[lvl, :], plotStyles3[0])
		plt.hold(True)
	plt.xlabel("Number of Iterations")
	plt.ylabel("Error")
	plt.savefig(resultsDir + "IncreasingContrastClustErrors_lvl"+ str(lvl)+"_pmax" + str(_maxP) + "_nEigen" + str(k2) + ".eps")
#	plt.show()

# to run
# python -c "execfile('exp/clusterexp/GrowingClusterContrast.py')"

