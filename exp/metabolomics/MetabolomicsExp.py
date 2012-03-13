"""
Perform cross validation using TreeRank
"""
import os
import numpy
import sys
import logging
import multiprocessing
import datetime
import gc 
from apgl.util.PathDefaults import PathDefaults
from apgl.util.Util import Util
from apgl.metabolomics.TreeRank import TreeRank
from apgl.metabolomics.TreeRankForest import TreeRankForest
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
from apgl.metabolomics.RankSVM import RankSVM
from apgl.metabolomics.RankBoost import RankBoost

class MetabolomicsExpRunner(multiprocessing.Process):
    def __init__(self, YList, X, featuresName, ages, args):
        super(MetabolomicsExpRunner, self).__init__(args=args)
        self.X = X
        self.YList = YList #The list of concentrations 
        self.featuresName = featuresName
        self.args = args
        self.ages = ages 

        self.maxDepth = 10
        self.numTrees = 10
        self.sampleSize = 1.0
        self.sampleReplace = True
        self.folds = 5
        self.resultsDir = PathDefaults.getOutputDir() + "metabolomics/"

        self.leafRankGenerators = []
        self.leafRankGenerators.append((LinearSvmGS.generate(), "SVM"))
        self.leafRankGenerators.append((SvcGS.generate(), "RBF-SVM"))
        self.leafRankGenerators.append((DecisionTree.generate(), "CART"))

        self.pcaLeafRankGenerators = [(LinearSvmPca.generate(), "LinearSVM-PCA")]

        self.funcLeafRankGenerators = []
        self.funcLeafRankGenerators.append((LinearSvmFGs.generate, "SVMF"))
        self.funcLeafRankGenerators.append((SvcFGs.generate, "RBF-SVMF"))
        self.funcLeafRankGenerators.append((DecisionTreeF.generate, "CARTF"))

        #Store all the label vectors and their missing values
        YIgf1Inds, YICortisolInds, YTestoInds = MetabolomicsUtils.createIndicatorLabels(YList)
        self.hormoneInds = [YIgf1Inds, YICortisolInds, YTestoInds]
        self.hormoneNames = MetabolomicsUtils.getLabelNames()

    def saveResult(self, X, Y, learner, fileName):
        """
        Save a single result to file, checking if the results have already been computed
        """
        fileBaseName, sep, ext = fileName.rpartition(".")
        lockFileName = fileBaseName + ".lock"
        gc.collect()

        if not os.path.isfile(fileName) and not os.path.isfile(lockFileName):
            try:
                lockFile = open(lockFileName, 'w')
                lockFile.close()
                logging.debug("Created lock file " + lockFileName)

                logging.debug("Computing file " + fileName)
                logging.debug(learner)
                (bestParams, allMetrics, bestMetaDicts) = learner.evaluateCvOuter(X, Y, self.folds)
                cvResults = {"bestParams":bestParams, "allMetrics":allMetrics, "metaDicts":bestMetaDicts}
                Util.savePickle(cvResults, fileName)
                
                os.remove(lockFileName)
                logging.debug("Deleted lock file " + lockFileName)
            except:
                logging.debug("Caught an error in the code ... skipping")
                raise
        else:
            logging.debug("File exists, or is locked: " + fileName)

    def saveResults(self, leafRankGenerators, mode="std"):
        """
        Compute the results and save them for a particular hormone. Does so for all
        leafranks
        """
        for j in range(len(self.hormoneInds)):
            nonNaInds = self.YList[j][1]
            hormoneInd = self.hormoneInds[j]

            for k in range(len(hormoneInd)):
                if type(self.X) == numpy.ndarray:
                    X = self.X[nonNaInds, :]
                else:
                    X = self.X[j][nonNaInds, :]
                X = numpy.c_[X, self.ages[nonNaInds]]

                if mode != "func":
                    X = Standardiser().standardiseArray(X)
                    
                Y = hormoneInd[k]
                waveletInds = numpy.arange(X.shape[1]-1)

                logging.debug("Shape of examples: " + str(X.shape))
                logging.debug("Distribution of labels: " + str(numpy.bincount(Y)))

                #Go through all the leafRanks
                for i in range(len(leafRankGenerators)):

                    leafRankName = leafRankGenerators[i][1]
                    if mode != "func":
                        leafRankGenerator = leafRankGenerators[i][0]
                    else:
                        leafRankGenerator = leafRankGenerators[i][0](waveletInds)

                    fileName = self.resultsDir + "TreeRank-" + self.hormoneNames[j] + "_" + str(k) + "-" +  leafRankName  + "-" + self.featuresName +  ".dat"
                    treeRank = TreeRank(leafRankGenerator)
                    treeRank.setMaxDepth(self.maxDepth)
                    self.saveResult(X, Y, treeRank, fileName)

                    fileName = self.resultsDir + "TreeRankForest-" + self.hormoneNames[j] + "_" + str(k) + "-" +  leafRankName  + "-" + self.featuresName +  ".dat"
                    treeRankForest = TreeRankForest(leafRankGenerator)
                    treeRankForest.setMaxDepth(self.maxDepth)
                    treeRankForest.setNumTrees(self.numTrees)
                    
                    treeRankForest.setSampleReplace(self.sampleReplace)
                    #Set the number of features to be the root of the total number if not functional
                    if mode == "std":
                        treeRankForest.setFeatureSize(numpy.round(numpy.sqrt(X.shape[1]))/float(X.shape[1]))
                    else:
                        treeRankForest.setFeatureSize(1.0)
                        
                    
                    self.saveResult(X, Y, treeRankForest, fileName)

                if mode == "std":
                    #Run RankSVM
                    fileName = self.resultsDir + "RankSVM-" + self.hormoneNames[j] + "_" + str(k)  + "-" + self.featuresName +  ".dat"
                    rankSVM = RankSVM()
                    self.saveResult(X, Y, rankSVM, fileName)

                    #fileName = self.resultsDir + "RBF-RankSVM-" + self.hormoneNames[j] + "_" + str(k)  + "-" + self.featuresName +  ".dat"
                    #rankSVM = RankSVM()
                    #rankSVM.setKernel("rbf")
                    #self.saveResult(X, Y, rankSVM, fileName)

                    #Run RankBoost
                    fileName = self.resultsDir + "RankBoost-" + self.hormoneNames[j] + "_" + str(k)  + "-" + self.featuresName +  ".dat"
                    rankBoost = RankBoost()
                    self.saveResult(X, Y, rankBoost, fileName)
                        
    def run(self):
        logging.debug('module name:' + __name__) 
        logging.debug('parent process:' +  str(os.getppid()))
        logging.debug('process id:' +  str(os.getpid()))

        self.saveResults(self.leafRankGenerators, "std")

    def runF(self):
        logging.debug('module name:' + __name__)
        logging.debug('parent process:' +  str(os.getppid()))
        logging.debug('process id:' +  str(os.getpid()))

        self.saveResults(self.funcLeafRankGenerators, "func")

    def runPCA(self):
        logging.debug('module name:' + __name__)
        logging.debug('parent process:' +  str(os.getppid()))
        logging.debug('process id:' +  str(os.getpid()))

        self.saveResults(self.pcaLeafRankGenerators, "pca")

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.debug("Running from machine " + str(gethostname()))
numpy.random.seed(21)

dataDir = PathDefaults.getDataDir() +  "metabolomic/"
X, X2, Xs, XOpls, YList, ages, df = MetabolomicsUtils.loadData()

mode = "cpd"
level = 10
XwDb4 = MetabolomicsUtils.getWaveletFeatures(X, 'db4', level, mode)
XwDb8 = MetabolomicsUtils.getWaveletFeatures(X, 'db8', level, mode)
XwHaar = MetabolomicsUtils.getWaveletFeatures(X, 'haar', level, mode)

#Filter the wavelets
Ns = [10, 25, 50, 75, 100]
dataList = []

for i in range(len(Ns)):
    N = Ns[i]
    XwDb4F, inds = MetabolomicsUtils.filterWavelet(XwDb4, N)
    dataList.append((XwDb4F[:, inds], "Db4-" + str(N)))

    XwDb8F, inds = MetabolomicsUtils.filterWavelet(XwDb8, N)
    dataList.append((XwDb8F[:, inds], "Db8-" + str(N)))

    XwHaarF, inds = MetabolomicsUtils.filterWavelet(XwHaar, N)
    dataList.append((XwHaarF[:, inds], "Haar-" + str(N)))

dataList.extend([(Xs, "raw_std"), (XwDb4, "Db4"), (XwDb8, "Db8"), (XwHaar, "Haar"), (X2, "log"), (XOpls, "opls")])

#Data for functional TreeRank
dataListF = [(XwDb4, "Db4"), (XwDb8, "Db8"), (XwHaar, "Haar")]
dataListPCA = ([(Xs, "raw_std"), (XwDb4, "Db4"), (XwDb8, "Db8"), (XwHaar, "Haar"), (X2, "log"), (XOpls, "opls")])

lock = multiprocessing.Lock()

numpy.random.seed(datetime.datetime.now().microsecond)
#numpy.random.seed(21)
permInds = numpy.random.permutation(len(dataList))
permIndsF = numpy.random.permutation(len(dataListF))
permIndsPCA = numpy.random.permutation(len(dataListPCA))
numpy.random.seed(21)

try:
    for ind in permInds:
        MetabolomicsExpRunner(YList, dataList[ind][0], dataList[ind][1], ages, args=(lock,)).run()
        
    for ind in permIndsF:
        MetabolomicsExpRunner(YList, dataListF[ind][0], dataListF[ind][1], ages, args=(lock,)).runF()

    for ind in permIndsPCA:
        MetabolomicsExpRunner(YList, dataListPCA[ind][0], dataListPCA[ind][1], ages, args=(lock,)).runPCA()

    logging.info("All done - see you around!")
except Exception as err:
    print(err)
    raise 
    


