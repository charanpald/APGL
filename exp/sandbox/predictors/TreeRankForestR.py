import numpy
import logging
import sklearn.cross_val as cross_val
from apgl.util.Util import Util
from apgl.util.Parameter import Parameter
from apgl.util.Evaluator import Evaluator
from exp.metabolomics.AbstractTreeRankR import AbstractTreeRankR

class TreeRankForestR(AbstractTreeRankR):
    """
    A wrapper for the TreeRankForest code written in R.
    """
    def __init__(self):
        super(TreeRankForestR, self).__init__()

        self.numTrees = 3
        self.replace = True
        self.sampleSize = 0.5

    def learnModelDataFrame(self, formula, XY):
        """
        Learn a tree using a DataFrame XY and formula.
        """
        if not self.printDebug:
            self.baseLib.sink("/dev/null")
        try:
            self.forest = self.treeRankLib.TreeRankForest(formula, XY, ntree=self.numTrees, replace=self.replace, sampsize=self.sampleSize, bestresponse=self.bestResponse, LeafRank=self.leafRank, nfcv=self.nfcv, varsplit=self.varSplit, growing=self.growing)
        except:
            logging.debug("TreeRankForest function failed, saving results to file.")
            robjects.r["write.csv"](XY, file="XY.csv")
            raise
        if not self.printDebug:
            self.baseLib.sink()

    def getModel(self):
        return self.forest

    def setNumTrees(self, numTrees):
        Parameter.checkInt(numTrees, 0, float('inf'))
        self.numTrees = numTrees

    def setSampleSize(self, sampleSize):
        Parameter.checkFloat(sampleSize, 0.0, 1.0)
        self.numTrees = sampleSize

    def __str__(self):
        #Just write out the parameters
        outStr = "TreeRankForest:"
        if self.leafRank == self.treeRankLib.LRCart:
            outStr += " LeafRank=CART"
        elif self.leafRank == self.treeRankLib.LRsvm:
            outStr += " LeafRank=SVM"
        elif self.leafRank == self.treeRankLib.LRforest:
            outStr += " LeafRank=Random Forests"
        else:
            outStr += " LeafRank=Unknown"
        outStr += " maxDepth=" + str(self.maxDepth)
        outStr += " varSplit=" + str(self.varSplit)
        outStr += " nfcv=" + str(self.nfcv)
        outStr += " minSplit=" + str(self.minSplit)
        outStr += " numTrees=" + str(self.numTrees)
        outStr += " replace=" + str(self.replace)
        outStr += " sampleSize=" + str(self.sampleSize)
        return outStr

    def evaluateCvOuter(self, X, Y, folds, leafRank):
        """
        Run cross validation and output some ROC curves. In this case Y is a 1D array.
        """
        Parameter.checkClass(X, numpy.ndarray)
        Parameter.checkClass(Y, numpy.ndarray)
        Parameter.checkInt(folds, 2, float('inf'))
        if Y.ndim != 1:
            raise ValueError("Expecting Y to be 1D")

        indexList = cross_val.StratifiedKFold(Y, folds)
        self.setLeafRank(leafRank)

        bestParams = []
        bestTrainAUCs = numpy.zeros(folds)
        bestTrainROCs = []
        bestTestAUCs = numpy.zeros(folds)
        bestTestROCs = []
        bestMetaDicts = []
        i = 0

        for trainInds, testInds in indexList:
            Util.printIteration(i, 1, folds)
            trainX, trainY = X[trainInds, :], Y[trainInds]
            testX, testY = X[testInds, :], Y[testInds]

            logging.debug("Distribution of labels in train: " + str(numpy.bincount(trainY)))
            logging.debug("Distribution of labels in test: " + str(numpy.bincount(testY)))

            self.learnModel(trainX, trainY)
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