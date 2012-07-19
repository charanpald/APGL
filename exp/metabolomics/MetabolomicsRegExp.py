#Perform regression on the metabolomics data using various regression methods 
import os 
import numpy
import sys
import logging
import multiprocessing
import gc
import datetime 
from apgl.util.PathDefaults import PathDefaults
from apgl.util.Sampling import Sampling
from apgl.util.Evaluator import Evaluator
from apgl.util.Util import Util 
from apgl.data.Standardiser import Standardiser 
from apgl.predictors.AbstractPredictor import AbstractPredictor  
from exp.metabolomics.MetabolomicsUtils import MetabolomicsUtils
from sklearn import svm, linear_model
from socket import gethostname
import rpy2 

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

class MetabolomicsRegExpRunner(multiprocessing.Process):
    def __init__(self, df, X, featuresName, ages, args):
        super(MetabolomicsRegExpRunner, self).__init__(args=args)
        self.df = df
        self.X = X
        self.featuresName = featuresName
        self.args = args
        self.ages = ages 

        self.labelNames = MetabolomicsUtils.getLabelNames()
        self.YList = MetabolomicsUtils.createLabelList(df, self.labelNames)
        self.boundsList = MetabolomicsUtils.getBounds()

        self.resultsDir = PathDefaults.getOutputDir() + "metabolomics/"

    def computeRankMetrics(self, X, Y, indexList, bestLearners, standardiserY, labelIndex):
        #Some code to do ranking using the learner predictors
        i = 0
        rankMetrics = numpy.zeros((len(indexList), self.boundsList[labelIndex].shape[0]-1))
        for idxtr, idxts in indexList:
            logging.info("Iteration " + str(i))

            trainX, testX = X[idxtr, :], X[idxts, :]
            trainY, testY = Y[idxtr], Y[idxts]

            bestLearners[i].learnModel(trainX, trainY)
            predY = bestLearners[i].predict(testX)
            gc.collect()

            #Now output 3 sets of ranked scores
            predY = standardiserY.unstandardiseArray(predY)
            testY = standardiserY.unstandardiseArray(testY)

            YScores = MetabolomicsUtils.scoreLabels(predY, self.boundsList[labelIndex])
            YIndList = MetabolomicsUtils.createIndicatorLabel(testY, self.boundsList[labelIndex])

            for j in range(self.boundsList[labelIndex].shape[0]-1):
                rankMetrics[i, j] = Evaluator.auc(YScores[:, j], YIndList[j])
            i += 1

        logging.debug(rankMetrics)

        return rankMetrics

    def meanAUC(self, predY, testY, labelIndex, standardiserY):
        predY = standardiserY.unstandardiseArray(predY)
        testY = standardiserY.unstandardiseArray(testY)

        YScores = MetabolomicsUtils.scoreLabels(predY, self.boundsList[labelIndex])
        YIndList = MetabolomicsUtils.createIndicatorLabel(testY, self.boundsList[labelIndex])

        rankMetrics = numpy.zeros(self.boundsList[labelIndex].shape[0]-1)

        for j in range(rankMetrics.shape[0]):
            rankMetrics[j] = Evaluator.auc(YScores[:, j], YIndList[j])

        return numpy.mean(rankMetrics)

    def saveResult(self, X, Y, indexList, splitFunction, learnerIterator, metricMethods, fileName, labelIndex, standardiserY):

        gc.collect()

        try:
            if not os.path.isfile(fileName):
                logging.debug("Computing file " + fileName)
                allMetrics, bestLearners = AbstractPredictor.evaluateLearners(X, Y, indexList, splitFunction, learnerIterator, metricMethods)
                rankMetrics = self.computeRankMetrics(X, Y, indexList, bestLearners, standardiserY, labelIndex)

                #Create objects we can serialise
                paramStrList = []
                for bestLearner in bestLearners:
                    paramStrList.append(str(bestLearner))

                Util.savePickle((allMetrics, rankMetrics, paramStrList), fileName)
            else:
                logging.debug("File exists: " + fileName)
        except:
            logging.debug("Caught an error in the code ... skipping")
            raise

    def saveResults(self, labelIndex):
        """
        Compute the results and save them for a particular hormone. Does so for all
        leafranks
        """
        folds = 5
        if type(self.X) == numpy.ndarray:
            X = self.X[self.YList[labelIndex][1], :]
        else:
            X = self.X[labelIndex][self.YList[labelIndex][1], :]

        X = numpy.c_[X, self.ages[self.YList[labelIndex][1]]]
        Y = self.YList[labelIndex][0]
        numExamples = X.shape[0]

        logging.debug("Shape of examples: " + str(X.shape))

        standardiserX = Standardiser()
        X = standardiserX.standardiseArray(X)

        standardiserY = Standardiser()
        Y = standardiserY.standardiseArray(Y)

        #We need to include the ROC curves
        indexList = Sampling.crossValidation(folds, numExamples)
        splitFunction = lambda trainX, trainY: Sampling.crossValidation(folds, trainX.shape[0])
    
        #We need a metric to minimise 
        def invMeanAUC(predY, testY):
            return 1 - self.meanAUC(predY, testY, labelIndex, standardiserY)

        metricMethods = [invMeanAUC]

        #Now create a learnerIterator based on the SVM
        Cs = 2**numpy.arange(-8, 2, dtype=numpy.float)
        gammas = 2**numpy.arange(-10, 0, dtype=numpy.float)
        epsilons = 2**numpy.arange(-5, 0, dtype=numpy.float)

        fileName = self.resultsDir + self.labelNames[labelIndex] + "-svr_rbf-" + self.featuresName +  ".dat"
        learnerIterator = []

        for C in Cs:
            for gamma in gammas:
                for epsilon in epsilons:
                    learner = svm.SVR(C=C, gamma=gamma, epsilon=epsilon)
                    learner.learnModel = learner.fit
                    learnerIterator.append(learner)

        self.saveResult(X, Y, indexList, splitFunction, learnerIterator, metricMethods, fileName, labelIndex, standardiserY)

        #Try the polynomial SVM
        fileName = self.resultsDir + self.labelNames[labelIndex] + "-svr_poly-" + self.featuresName +  ".dat"
        degrees = numpy.array([2, 3])

        for C in Cs:
            for degree in degrees:
                for epsilon in epsilons:
                    learner = svm.SVR(kernel='poly', C=C, degree=degree, epsilon=epsilon)
                    learner.learnModel = learner.fit
                    learnerIterator.append(learner)

        self.saveResult(X, Y, indexList, splitFunction, learnerIterator, metricMethods, fileName, labelIndex, standardiserY)
            
        #Now try Lasso and ElasticNet
        fileName = self.resultsDir + self.labelNames[labelIndex] + "-lasso-" + self.featuresName +  ".dat"
        alphas = 2**numpy.arange(-9, 0, dtype=numpy.float)
        learnerIterator = []

        for alpha in alphas:
            learner = linear_model.Lasso(alpha = alpha)
            learner.learnModel = learner.fit
            learnerIterator.append(learner)

        self.saveResult(X, Y, indexList, splitFunction, learnerIterator, metricMethods, fileName, labelIndex, standardiserY)

    def run(self):
        logging.debug('module name:' + __name__)
        logging.debug('parent process:' +  str(os.getppid()))
        logging.debug('process id:' +  str(os.getpid()))

        for i in range(len(self.labelNames)):
            self.saveResults(i)

logging.debug("Running from machine " + str(gethostname()))

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

lock = multiprocessing.Lock()

numpy.random.seed(datetime.datetime.now().microsecond)
#numpy.random.seed(21)
permInds = numpy.random.permutation(len(dataList))
numpy.random.seed(21)

try:
    for ind in permInds:
        MetabolomicsRegExpRunner(df, dataList[ind][0], dataList[ind][1], ages, args=(lock,)).run()

    logging.info("All done - see you around!")
except rpy2.rinterface.RRuntimeError as err:
    print(err)
    baseLib.traceback()

