"""
Compare the R version of TreeRank to the python version.
"""

"""
Perform cross validation using TreeRank
"""
import os
import numpy
import sys
import logging
import multiprocessing
import datetime
from apgl.util.PathDefaults import PathDefaults
from apgl.util.Util import Util
from apgl.util.Evaluator import Evaluator
from apgl.metabolomics.TreeRank import TreeRank
from apgl.metabolomics.TreeRankForest import TreeRankForest
from apgl.metabolomics.TreeRankR import TreeRankR
from apgl.metabolomics.MetabolomicsUtils import MetabolomicsUtils
from socket import gethostname
from apgl.data.Standardiser import Standardiser
from apgl.metabolomics.leafrank.LinearSVM import LinearSvmGS
from apgl.metabolomics.leafrank.SVC import SvcGS
from apgl.metabolomics.leafrank.DecisionTree import DecisionTree
from apgl.metabolomics.leafrank.LinearSvmFGs import LinearSvmFGs
from apgl.metabolomics.leafrank.LinearSvmPca import LinearSvmPca
from apgl.metabolomics.leafrank.SvcFGs import SvcFGs
from apgl.metabolomics.leafrank.DecisionTreeF import DecisionTreeF

import sklearn.decomposition as decomp

class MetabolomicsExpRunner(multiprocessing.Process):
    def __init__(self, YList, X, featuresName, ages, args):
        super(MetabolomicsExpRunner, self).__init__(args=args)
        self.X = X
        self.YList = YList #The list of concentrations
        self.featuresName = featuresName
        self.args = args
        self.ages = ages

        self.maxDepth = 5
        self.numTrees = 10
        self.folds = 3
        self.resultsDir = PathDefaults.getOutputDir() + "metabolomics/"

        self.leafRankGenerators = []
        #self.leafRankGenerators.append((SvcGS.generate(), "SVC"))
        #self.leafRankGenerators.append((LinearSvmGS.generate(), "LinearSVM"))
        self.leafRankGenerators.append((LinearSvmPca.generate(), "LinearSVM-PCA"))

        self.funcLeafRankGenerators = []
        #self.funcLeafRankGenerators.append((LinearSvmFGs.generate, "SVMF"))
        #self.funcLeafRankGenerators.append((DecisionTreeF.generate, "CARTF"))
        self.funcLeafRankGenerators.append((SvcFGs.generate, "SVCF"))

        #Store all the label vectors and their missing values
        YIgf1Inds, YICortisolInds, YTestoInds = MetabolomicsUtils.createIndicatorLabels(YList)
        self.hormoneInds = [YIgf1Inds, YICortisolInds, YTestoInds]
        self.hormoneNames = MetabolomicsUtils.getLabelNames()

    def saveResults(self, leafRankGenerators, standardise=True):
        """
        Compute the results and save them for a particular hormone. Does so for all
        leafranks
        """
        j = 0
        nonNaInds = self.YList[j][1]
        hormoneInd = self.hormoneInds[j]

        k = 2
        if type(self.X) == numpy.ndarray:
            X = self.X[nonNaInds, :]
        else:
            X = self.X[j][nonNaInds, :]
        X = numpy.c_[X, self.ages[nonNaInds]]
        if standardise:
            X = Standardiser().standardiseArray(X)
        Y = hormoneInd[k]

        waveletInds = numpy.arange(X.shape[1]-1)

        logging.debug("Shape of examples: " + str(X.shape))
        logging.debug("Distribution of labels: " + str(numpy.bincount(Y)))

        #pca = decomp.PCA(n_components=40)
        #X = pca.fit_transform(X)
        #print(X.shape)

        #Go through all the leafRanks
        for i in range(len(leafRankGenerators)):
            #Compute TreeRankForest here
            fileName = self.resultsDir + "TreeRankForest-" + self.hormoneNames[j] + "_" + str(k) + "-" +  leafRankGenerators[i][1]  + "-" + self.featuresName +  ".dat"
            try:
                logging.debug("Computing file " + fileName)
                #treeRankForest = TreeRankForest(self.funcLeafRankGenerators[0][0](waveletInds))
                treeRankForest = TreeRankForest(self.leafRankGenerators[0][0])
                treeRankForest.setMaxDepth(10)
                treeRankForest.setNumTrees(5)
                #Setting this low definitely helps 
                #treeRankForest.setFeatureSize(1.0)
                treeRankForest.setFeatureSize(0.05)
                #The following 2 lines definitely improve stability and the AUC 
                treeRankForest.setSampleSize(1.0)
                #Setting this to true results in slightly worse results 
                treeRankForest.setSampleReplace(True)
                mean, var = treeRankForest.evaluateStratifiedCv(X, Y, self.folds, metricMethod=Evaluator.auc)
                print(mean)

                #treeRank = TreeRank(self.leafRankGenerators[0][0])
                #treeRank.setMaxDepth(self.maxDepth)
                #(bestParams, allMetrics, bestMetaDicts) = treeRank.evaluateCvOuter(X, Y, self.folds)
                #print(str(allMetrics))


                #Util.savePickle(cvResults, fileName)
            except:
                logging.debug("Caught an error in the code ... skipping")
                raise
            else:
                logging.debug("File exists: " + fileName)
        return

    def run(self):
        logging.debug('module name:' + __name__)
        logging.debug('parent process:' +  str(os.getppid()))
        logging.debug('process id:' +  str(os.getpid()))

        self.saveResults(self.leafRankGenerators, True)

    def run2(self):
        logging.debug('module name:' + __name__)
        logging.debug('parent process:' +  str(os.getppid()))
        logging.debug('process id:' +  str(os.getpid()))

        self.saveResults(self.funcLeafRankGenerators, False)

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.debug("Running from machine " + str(gethostname()))
numpy.random.seed(21)

dataDir = PathDefaults.getDataDir() +  "metabolomic/"
X, X2, Xs, XOpls, YList, ages, df = MetabolomicsUtils.loadData()

waveletStr = 'db4'
mode = "cpd"
level = 10
XwDb4 = MetabolomicsUtils.getWaveletFeatures(X, 'db4', level, mode)
XwDb8 = MetabolomicsUtils.getWaveletFeatures(X, 'db8', level, mode)
XwHaar = MetabolomicsUtils.getWaveletFeatures(X, 'haar', level, mode)

dataList = []
dataList.extend([(XwDb4, "db4")])

lock = multiprocessing.Lock()

numpy.random.seed(datetime.datetime.now().microsecond)
#numpy.random.seed(21)
permInds = numpy.random.permutation(len(dataList))
numpy.random.seed(21)

try:
    for ind in permInds:
        MetabolomicsExpRunner(YList, dataList[ind][0], dataList[ind][1], ages, args=(lock,)).run()

    logging.info("All done - see you around!")
except Exception as err:
    print(err)
    raise



