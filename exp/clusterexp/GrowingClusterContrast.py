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
from exp.sandbox.GraphIterators import toDenseGraphListIterator
from exp.sandbox.NingSpectralClustering import NingSpectralClustering
import argparse

#=========================================================================
#=========================================================================
# arguments (overwritten by the command line)
#=========================================================================
#=========================================================================
# Arguments related to the dataset
# at each lvl, clusters are splits in $args.numSubClustersPerLevel$ parts
# at each iterations, edges are added. With time, probability to pick an edge
# in a high lvl cluster increase, while the probability decreases for low lvl
# clusters.
args = argparse.Namespace()
args.numLevel = 3 # including the level where all nodes are grouped
args.numSubClustersPerLevel = 3
args.numVerticesPerSmallestCluser = 20
args.maxP = 0.01
args.startingIteration = 2
args.endingIteration = 22

# Arguments related to the algorithm
args.runIASC = False
args.runExact = False
args.runModularity = False
args.runNystrom = False
args.runEfficientNystrom = False
args.runRandomisedSvd = False
args.runNing = False

args.k1 = 9                 # numCluster to learn
args.k2 = 1*(args.k1+1)         # numEigenVector kept
args.k3 = int(1.1*args.k2)  # numRowsColumns used by Nystrom
args.k4 = int(1.1*args.k2)  # number of random projections to use with randomised SVD

args.exactFreq = 10

args.numRepetitions = 2

args.computeBound = False

#=========================================================================
#=========================================================================
# init (reading/writting command line arguments, print options, logging)
#=========================================================================
#=========================================================================

# parser #
parser = argparse.ArgumentParser(description="")
parser.add_argument("--numLevel", type=int, help="", default=args.numLevel)
parser.add_argument("--numSubClustersPerLevel", type=int, help="", default=args.numSubClustersPerLevel)
parser.add_argument("--numVerticesPerSmallestCluser", type=int, help="", default=args.numVerticesPerSmallestCluser)
parser.add_argument("--maxP", type=float, help="", default=args.maxP)
parser.add_argument("--startingIteration", type=int, help="At which iteration to start clustering algorithms", default=args.startingIteration)
parser.add_argument("--endingIteration", type=int, help="At which iteration to end clustering algorithms", default=args.endingIteration)

for method in ["runIASC", "runExact", "runModularity", "runNystrom", "runEfficientNystrom", "runNing", "runRandomisedSvd"]:
    parser.add_argument("--" + method, action="store_true", default=args.__getattribute__(method))
parser.add_argument("--k1", type=int, help="Number of clusers", default=args.k1)
parser.add_argument("--k2", type=int, help="Rank of the approximation", default=args.k2)
parser.add_argument("--k3", type=int, help="Number of row/cols used by to find the approximate eigenvalues with Nystrom approach", default=args.k3)
parser.add_argument("--k4", type=int, help="Number of random projections to use with the randomised SVD approach", default=args.k3)
parser.add_argument("--exactFreq", type=int, help="Number of iteration between each exact decomposition", default=args.exactFreq)
parser.add_argument("--numRepetitions", type=int, help="Printed results are averaged on NUMREPETITION runs", default=args.numRepetitions)
parser.add_argument("--computeBound", action="store_true", default=args.computeBound, help="Compute bounds on spaces angles")
parser.parse_args(namespace=args)

# miscellnious #
numpy.random.seed(31)
numpy.seterr(all="raise", under="ignore")
numpy.set_printoptions(suppress=True, precision=3, linewidth=200, threshold=40000)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# print args #
logging.info("Data params:")
keys = list(vars(args).keys())
keys.sort()
for key in keys:
    logging.info("    " + str(key) + ": " + str(args.__getattribute__(key)))

#=========================================================================
#=========================================================================
# Graph Iterator
#=========================================================================
#=========================================================================
class GrowingContrastGraphIterator(object):
    def __init__(self):
        self.numLevel = args.numLevel
        self.numSubClustersPerLevel = args.numSubClustersPerLevel
        self.numSmallestCluster = self.numSubClustersPerLevel**(self.numLevel-1)
        self.numVertices = self.numSmallestCluster*args.numVerticesPerSmallestCluser
        self.minP = 0.0005
        self.maxP = args.maxP
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
        subSize = size//self.numSubClustersPerLevel
        for ii in range(i, i+size, subSize):
            j = ii + subSize
            self.levelMatrix[ii:j,ii:j] = lvl * numpy.ones((subSize, subSize))
            self.fillLevelMatrix(ii, subSize, lvl+1)

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()
        
    def __next__(self):
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
#            print P
#            print newEdges
#            print self.edgesMatrix*7 +1
#            print sum(newEdges.ravel()), sum(newEdges.ravel()) == 0.0
        self.iter += 1
        return self.edgesMatrix


#=========================================================================
#=========================================================================
# useful
#=========================================================================
#=========================================================================
def randIndex(learnedClustering, clust_size):
    numVertices = len(learnedClustering)
    error = 0
    for v1 in range(numVertices):
        for v2 in range(v1+1, numVertices):
            same_cl = (v1 // clust_size) == (v2 // clust_size)
            same_learned_cl = learnedClustering[v1] == learnedClustering[v2]
            error += same_cl != same_learned_cl
    
    return (float(error)*2/(numVertices)/(numVertices-1))

#=========================================================================
#=========================================================================
# run
#=========================================================================
#=========================================================================
numIter = len(range(args.startingIteration, args.endingIteration))

logging.info("compute clusters")
exactClusterer = IterativeSpectralClustering(args.k1, alg="exact", computeSinTheta=True)
approxClusterer = IterativeSpectralClustering(args.k1, args.k2, T=args.exactFreq, alg="IASC", computeSinTheta=True)
nystromClusterer = IterativeSpectralClustering(args.k1, k3=args.k3, alg="nystrom", computeSinTheta=True)
RSvdClusterer = IterativeSpectralClustering(args.k1, k4=args.k4, alg="randomisedSvd", computeSinTheta=True)
ningsClusterer = NingSpectralClustering(args.k1, T=args.exactFreq, computeSinTheta=True)


exactClusterer.nb_iter_kmeans = 20
approxClusterer.nb_iter_kmeans = 20
nystromClusterer.nb_iter_kmeans = 20

#exactClusterer.computeBound = args.computeBound        # computeBound not implemented for exactClusterer
approxClusterer.computeBound = args.computeBound
#nystromClusterer.computeBound = args.computeBound      # computeBound not implemented for nystromClusterer
#RSvdClusterer.computeBound = args.computeBound      # computeBound not implemented for RSvdClusterer
#ningsClusterer.computeBound = args.computeBound        # computeBound not implemented for ningsClusterer
            
def getGraphIterator():
    return itertools.islice(GrowingContrastGraphIterator(), args.startingIteration, args.endingIteration)

for r in range(args.numRepetitions):
    logging.info("run " + str(r))
    
    if args.runExact:
        # run with exact eigenvalue decomposition
        logging.info("Running exact method")
        graphIterator = getGraphIterator()
        clustersExact, timeListExact, eigenQualityExact = exactClusterer.clusterFromIterator(graphIterator, verbose=True)

    if args.runIASC:
        # run with our incremental approximation
        logging.info("Running approximate method")
        graphIterator = getGraphIterator()
        clustListApprox, timeListApprox, eigenQualityApprox = approxClusterer.clusterFromIterator(graphIterator, verbose=True)

    if args.runNystrom:
        # run with Nystrom approximation
        logging.info("Running nystrom method")
        graphIterator = getGraphIterator()
        clustListNystrom, timeListNystrom, eigenQualityNystrom = nystromClusterer.clusterFromIterator(graphIterator, verbose=True)

    if args.runRandomisedSvd:
        # run with Nystrom approximation
        logging.info("Running randomised svd method")
        graphIterator = getGraphIterator()
        clustListRSvd, timeListRSvd, eigenQualityRSvd = RSvdClusterer.clusterFromIterator(graphIterator, verbose=True)

    if args.runNing:
        # run with Ning's incremental approximation
        logging.info("Running Nings method")
        graphIterator = getGraphIterator()
#        clustListNings = ningsClusterer.cluster(toDenseGraphListIterator(graphIterator))
        clustListNings, timeListNings, eigenQualityNings = ningsClusterer.cluster(graphIterator, verbose=True)

    # print clusters
    if args.runExact:
        logging.info("learned clustering with exact eigenvalue decomposition")
        for i in range(len(clustersExact)):
            clusters = clustersExact[i]
            print(clusters)
    if args.runIASC:
        logging.info("learned clustering with our approximation approach")
        for i in range(len(clustListApprox)):
            clusters = clustListApprox[i]
            print(clusters)
    if args.runNing:
        logging.info("learned clustering with Nings approximation approach")
        for i in range(len(clustListNings)):
            clusters = clustListNings[i]
            print(clusters)
    if args.runNystrom:
        logging.info("learned clustering with Nystrom approximation approach")
        for i in range(len(clustListNystrom)):
            clusters = clustListNystrom[i]
            print(clusters)
    if args.runRandomisedSvd:
        logging.info("learned clustering with randomised svd approximation approach")
        for i in range(len(clustListRSvd)):
            clusters = clustListRSvd[i]
            print(clusters)

    # compute error for each iteration and lvl
    # error: proportion of pairs of vertices (x,y) s.t.
    #    (cl(x) == cl(y)) != (learned_cl(x) == learned_cl(y))
    if not 'meanClustErrExact' in locals():
        meanClustErrExact = numpy.zeros((numIter, args.numLevel))
        meanClustErrApprox = numpy.zeros((numIter, args.numLevel))
        meanClustErrNings = numpy.zeros((numIter, args.numLevel))
        meanClustErrNystrom = numpy.zeros((numIter, args.numLevel))
        meanClustErrRSvd = numpy.zeros((numIter, args.numLevel))
        meanBoundListExact = numpy.zeros((numIter,4))
        meanBoundListApprox = numpy.zeros((numIter,4))
        meanBoundListNings = numpy.zeros((numIter,4))
        meanBoundListNystrom = numpy.zeros((numIter,4))
        meanBoundListRSvd = numpy.zeros((numIter,4))
        meanSinThetaListExact = numpy.zeros((numIter,1))
        meanSinThetaListApprox = numpy.zeros((numIter,1))
        meanSinThetaListNings = numpy.zeros((numIter,1))
        meanSinThetaListNystrom = numpy.zeros((numIter,1))
        meanSinThetaListRSvd = numpy.zeros((numIter,1))
        if args.runExact:
            numVertices = clustersExact[0].size
        if args.runIASC:
            numVertices = clustListApprox[0].size
        if args.runNing:
            numVertices = clustListNings[0].size
        if args.runNystrom:
            numVertices = clustListNystrom[0].size
        if args.runRandomisedSvd:
            numVertices = clustListRSvd[0].size
            

    clust_size = numVertices # number of vertices per cluster
    for lvl in range(args.numLevel):

        if args.runExact:
            # error with exact eigenvalue decomposition
            for it in range(numIter):
                meanClustErrExact[it, lvl] += randIndex(clustersExact[it], clust_size)
            if exactClusterer.computeBound:
                meanBoundListExact += eigenQualityExact["boundList"]
            meanSinThetaListExact += scipy.array([eigenQualityExact["sinThetaList"]]).T

        if args.runIASC:
            # error with our incremental approximation
            for it in range(numIter):
                meanClustErrApprox[it, lvl] += randIndex(clustListApprox[it], clust_size)
            if approxClusterer.computeBound:
                meanBoundListApprox += eigenQualityApprox["boundList"]
            meanSinThetaListApprox += scipy.array([eigenQualityApprox["sinThetaList"]]).T

        if args.runNing:
            # error with Ning incremental approximation
            for it in range(numIter):
                meanClustErrNings[it, lvl] += randIndex(clustListNings[it], clust_size)
            if ningClusterer.computeBound:
                meanBoundListNings += eigenQualityNings["boundList"]
            meanSinThetaListNings += scipy.array([eigenQualityNings["sinThetaList"]]).T

        if args.runNystrom:
            # error with Ning incremental approximation
            for it in range(numIter):
                meanClustErrNystrom[it, lvl] += randIndex(clustListNystrom[it], clust_size)
            if nystromClusterer.computeBound:
                meanBoundListNystrom += eigenQualityNystrom["boundList"]
            meanSinThetaListNystrom += scipy.array([eigenQualityNystrom["sinThetaList"]]).T

        if args.runRandomisedSvd:
            # error with Ning incremental approximation
            for it in range(numIter):
                meanClustErrRSvd[it, lvl] += randIndex(clustListRSvd[it], clust_size)
            if RSvdClusterer.computeBound:
                meanBoundListRSvd += eigenQualityRSvd["boundList"]
            meanSinThetaListRSvd += scipy.array([eigenQualityRSvd["sinThetaList"]]).T

        # update variables related to lvl
        clust_size //= args.numSubClustersPerLevel

if args.runExact:
    meanClustErrExact = meanClustErrExact/args.numRepetitions
    meanBoundListExact = meanBoundListExact/args.numRepetitions
    meanSinThetaListExact = meanSinThetaListExact/args.numRepetitions
    print(meanClustErrExact)
if args.runIASC:
    meanClustErrApprox = meanClustErrApprox/args.numRepetitions
    meanBoundListApprox = meanBoundListApprox/args.numRepetitions
    meanSinThetaListApprox = meanSinThetaListApprox/args.numRepetitions
    print(meanClustErrApprox)
if args.runNing:
    meanClustErrNings = meanClustErrNings/args.numRepetitions
    meanBoundListNings = meanBoundListNings/args.numRepetitions
    meanSinThetaListNings = meanSinThetaListNings/args.numRepetitions
    print(meanClustErrNings)
if args.runNystrom:
    meanClustErrNystrom = meanClustErrNystrom/args.numRepetitions
    meanBoundListNystrom = meanBoundListNystrom/args.numRepetitions
    meanSinThetaListNystrom = meanSinThetaListNystrom/args.numRepetitions
    print(meanClustErrNystrom)
if args.runRandomisedSvd:
    meanClustErrRSvd = meanClustErrRSvd/args.numRepetitions
    meanBoundListRSvd = meanBoundListRSvd/args.numRepetitions
    meanSinThetaListRSvd = meanSinThetaListRSvd/args.numRepetitions
    print(meanClustErrRSvd)



resultsDir = PathDefaults.getOutputDir() + "cluster/"
#Save randindex results in a file
file_name = resultsDir + "IncreasingContrastClustErrors_pmax" + str(args.maxP) + "_nEigen" + str(args.k2) + ".dat"
try: 
    res_file = open(file_name, 'wb')
except IOError as e:
    logging.warning(" unable to open file '", file_name, "'\n", e)
    logging.warning("=> results not saved")
else:
    res_file.write("# error for exact_lvl0 exact_lvl1 exact_lvl2 IASC_lvl0 IASC_lvl1 IASC_lvl2 nings_lvl0 nings_lvl1 nings_lvl2 Nystrom_lvl0 Nystrom_lvl1 Nystrom_lvl2 RSvd_lvl0 RSvd_lvl1 RSvd_lvl2\n".encode('utf-8'))
    res = numpy.hstack((meanClustErrExact, meanClustErrApprox, meanClustErrNings, meanClustErrNystrom, meanClustErrRSvd))
    numpy.savetxt(res_file, res)
    logging.debug("Wrote " + file_name)
    
#Save bound in a file
file_name = resultsDir + "IncreasingContrastBound_pmax" + str(args.maxP) + "_nEigen" + str(args.k2) + ".dat"
try: 
    res_file = open(file_name, 'wb')
except IOError as e:
    logging.warning(" unable to open file '", file_name, "'\n", e)
    logging.warning("=> results not saved")
else:
    res_file.write("# bound for exact IASC nings Nystrom RSvd\n".encode('utf-8'))
    res = numpy.hstack((meanBoundListExact, meanBoundListApprox, meanBoundListNings, meanBoundListNystrom, meanBoundListRSvd))
    numpy.savetxt(res_file, res)
    logging.debug("Wrote " + file_name)
    
#Save sin(Theta) in a file
file_name = resultsDir + "IncreasingContrastSinTheta_pmax" + str(args.maxP) + "_nEigen" + str(args.k2) + ".dat"
try: 
    res_file = open(file_name, 'wb')
except IOError as e:
    logging.warning(" unable to open file '", file_name, "'\n", e)
    logging.warning("=> results not saved")
else:
    res_file.write("# sinTheta for exact IASC nings Nystrom RSvd\n".encode('utf-8'))
    res = numpy.hstack((meanSinThetaListExact, meanSinThetaListApprox, meanSinThetaListNings, meanSinThetaListNystrom, meanSinThetaListRSvd))
    numpy.savetxt(res_file, res)
    logging.debug("Wrote " + file_name)
    

#=============================================================================
#Now lets plot the results
#=============================================================================
# lvl 0: 1 cluster
# lvl 1: args.numSubClustersPerLevel clusters
# lvl i: args.numSubClustersPerLevel^i clusters
iterations = numpy.arange(args.startingIteration, args.endingIteration)

for lvl in range(args.numLevel):
    plt.hold(False)
    if args.runExact:
        plt.plot(iterations, meanClustErrExact[:, lvl], 'ko-', label="Exact")
        plt.hold(True)
    if args.runIASC:
        plt.plot(iterations, meanClustErrApprox[:, lvl], 'rx-', label="IASC")
        plt.hold(True)
    if args.runNystrom:
        plt.plot(iterations, meanClustErrNystrom[:, lvl], 'bx--', label="Nystrom")
        plt.hold(True)
    if args.runRandomisedSvd:
        plt.plot(iterations, meanClustErrRSvd[:, lvl], 'bx--', label="RSvd")
        plt.hold(True)
    if args.runNing:
        plt.plot(iterations, meanClustErrNings[:, lvl], 'gx:', label="Ning et al.")
        plt.hold(True)
    plt.xlabel("Number of Iterations")
    plt.ylabel("Error")
    plt.legend()
    plt.grid(True)
    plt.savefig(resultsDir + "IncreasingContrastClustErrors_lvl"+ str(lvl)+"_pmax" + str(args.maxP) + "_nEigen" + str(args.k2) + ".eps")
    logging.debug(resultsDir + "IncreasingContrastClustErrors_lvl"+ str(lvl)+"_pmax" + str(args.maxP) + "_nEigen" + str(args.k2) + ".eps")
#    plt.show()


#=============================================================================
# SinTheta / Bound
iterations = numpy.arange(args.startingIteration, args.endingIteration)

plt.hold(False)
if args.runExact:
    plt.plot(iterations, meanSinThetaListExact[:, 0], 'ko-', label="Exact")
    plt.hold(True)
#    plt.plot(iterations, meanBoundListExact[:, 0], 'ko--', label="Exact")
if args.runIASC:
    plt.plot(iterations, meanSinThetaListApprox[:, 0], 'rx-', label="IASC")
    plt.hold(True)
    plt.plot(iterations, meanBoundListApprox[:, 0], 'rx--', label="IASC")
if args.runNystrom:
    plt.plot(iterations, meanSinThetaListNystrom[:, 0], 'bx-', label="Nystrom")
    plt.hold(True)
#    plt.plot(iterations, meanBoundListNystrom[:, 0], 'bx--', label="Nystrom")
if args.runRandomisedSvd:
    plt.plot(iterations, meanSinThetaListRSvd[:, 0], 'bx-', label="RSvd")
    plt.hold(True)
#    plt.plot(iterations, meanBoundListRSvd[:, 0], 'bx--', label="RSvd")
if args.runNing:
    plt.plot(iterations, meanSinThetaListNings[:, 0], 'gx-', label="Ning et al.")
    plt.hold(True)
#    plt.plot(iterations, meanBoundListNings[:, 0], 'gx--', label="Ning et al.")
plt.xlabel("Number of Iterations")
plt.ylabel("Error")
plt.legend()
plt.grid(True)
plt.savefig(resultsDir + "IncreasingContrastEigenQuality_pmax" + str(args.maxP) + "_nEigen" + str(args.k2) + ".eps")
logging.debug(resultsDir + "IncreasingContrastEigenQuality_pmax" + str(args.maxP) + "_nEigen" + str(args.k2) + ".eps")



