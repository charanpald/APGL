import numpy
import logging 
import scikits.learn.cross_val as cross_val
from apgl.util.Parameter import Parameter
from apgl.util.Util import Util
from apgl.util.Evaluator import Evaluator
from apgl.predictors.AbstractPredictor import AbstractPredictor

class AbstractTreeRank(AbstractPredictor):
    """
    Some common functionality between TreeRank and TreeRankForest.
    """
    def __init__(self, leafRanklearner):
        """
        Create the AbstractTreeRank with the given leaf rank generator. 
        """
        self.maxDepth = 2
        self.minSplit = 50
        self.bestResponse = 1
        self.featureSize = 1.0
        self.leafRanklearner = leafRanklearner
        
        self.minLabelCount = 5

    def setFeatureSize(self, featureSize):
        """
        Set the number of features to use for node computation.

        :param featureSize: the proportion of features to randomly select to compute each node.
        :type featureSize: :class:`float`
        """
        Parameter.checkFloat(featureSize, 0.0, 1.0)
        self.featureSize = featureSize

    def getFeatureSize(self):
        """
        :return: The proportion of features to randomly select to compute each node. 
        """
        return self.featureSize

    def setMinSplit(self, minSplit):
        """
        :param minSplit: the minimum number of examples in a node for it to be split. 
        :type minSplit: :class:`int`
        """
        Parameter.checkInt(minSplit, 2, float("inf"))
        self.minSplit = minSplit

    def getMinSplit(self):
        """
        :return: The minimum number of examples in a node for it to be split.
        """
        return self.minSplit

    def setMaxDepth(self, maxDepth):
        """
        :param maxDepth: the maximum depth of the learnt tree. 
        :type maxDepth: :class:`int`
        """
        Parameter.checkInt(maxDepth, 1, float("inf"))
        self.maxDepth = maxDepth

    def getMaxDepth(self):
        """
        :return: the maximum depth of the learnt tree. 
        """
        return self.maxDepth

    def setBestResponse(self, bestResponse):
        """
        The best response is the label which corresponds to "positive" 

        :param bestResponse: the label corresponding to "positive" 
        :type bestResponse: :class:`int`
        """
        Parameter.checkInt(bestResponse, -float("inf"), float("inf"))
        self.bestResponse = bestResponse

    def getBestResponse(self):
        """
        :return: the label corresponding to "positive"
        """
        return self.bestResponse

    def evaluateCvOuter(self, X, Y, folds):
        """
        Run cross validation and output some ROC curves. In this case Y is a 1D array.

        :param X: A matrix with examples as rows
        :type X: :class:`ndarray`

        :param y: A vector of labels
        :type y: :class:`ndarray`

        :param folds: The number of cross validation folds
        :type folds: :class:`int`
        """
        Parameter.checkClass(X, numpy.ndarray)
        Parameter.checkClass(Y, numpy.ndarray)
        Parameter.checkInt(folds, 2, float('inf'))
        if Y.ndim != 1:
            raise ValueError("Expecting Y to be 1D")

        indexList = cross_val.StratifiedKFold(Y, folds)

        bestParams = []
        bestTrainAUCs = numpy.zeros(folds)
        bestTrainROCs = []
        bestTestAUCs = numpy.zeros(folds)
        bestTestROCs = []
        bestMetaDicts = []
        i = 0

        for trainInds, testInds in indexList:
            Util.printIteration(i, 1, folds, "Outer CV: ")
            trainX, trainY = X[trainInds, :], Y[trainInds]
            testX, testY = X[testInds, :], Y[testInds]

            self.learnModel(trainX, trainY)
            #self.learnModelCut(trainX, trainY)

            predTrainY = self.predict(trainX)
            predTestY = self.predict(testX)
            bestTrainAUCs[i] = Evaluator.auc(predTrainY, trainY)
            bestTestAUCs[i] = Evaluator.auc(predTestY, testY)

            #Store the parameters and ROC curves
            bestTrainROCs.append(Evaluator.roc(trainY, predTrainY))
            bestTestROCs.append(Evaluator.roc(testY, predTestY))

            metaDict = {}
            bestMetaDicts.append(metaDict)

            i += 1

        logging.debug("Mean test AUC = " + str(numpy.mean(bestTestAUCs)))
        logging.debug("Std test AUC = " + str(numpy.std(bestTestAUCs)))
        allMetrics = [bestTrainAUCs, bestTrainROCs, bestTestAUCs, bestTestROCs]

        return (bestParams, allMetrics, bestMetaDicts)