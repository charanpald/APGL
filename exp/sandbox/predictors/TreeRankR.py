
import gc
import numpy
import logging 
import rpy2.robjects as robjects
import sklearn.cross_val as cross_val
from apgl.util.Util import Util
from apgl.util.Parameter import Parameter
from apgl.util.Evaluator import Evaluator
from exp.metabolomics.AbstractTreeRankR import AbstractTreeRankR

class TreeRankR(AbstractTreeRankR):
    """
    A wrapper for the TreeRank code written in R.
    """
    def __init__(self):
        super(TreeRankR, self).__init__()

    def learnModelDataFrame(self, formula, XY):
        """
        Learn a tree using a DataFrame XY and formula. 
        """
        if not self.printDebug:
            self.baseLib.sink("/dev/null")
        self.tree = self.treeRankLib.TreeRank(formula, XY, bestresponse=self.bestResponse, LeafRank=self.leafRank, nfcv=self.nfcv, varsplit=self.varSplit, growing=self.growing)
        if not self.printDebug:
            self.baseLib.sink()

    def evaluateCvOuter(self, X, Y, folds, leafRank, innerFolds=3):
        """
        Run model selection and output some ROC curves. In this case Y is a 1D array. 
        """
        Parameter.checkClass(X, numpy.ndarray)
        Parameter.checkClass(Y, numpy.ndarray)
        Parameter.checkInt(folds, 2, float('inf'))
        if Y.ndim != 1:
            raise ValueError("Expecting Y to be 1D")

        indexList = cross_val.StratifiedKFold(Y, folds)

        maxDepths = numpy.flipud(numpy.arange(1, 12, 1))
        if leafRank == self.getTreeRankLib().LRforest:
            varSplits = numpy.arange(0.6, 1.01, 0.2)
        else:
            varSplits = numpy.array([1])
        #According to Nicolas nfcv>1 doesn't help
        nfcvs = [1]
        #This is tied in with depth 
        mincrit = 0.00
        #If minsplit is too low sometimes get a node with no positive labels
        minSplits = numpy.array([50])

        self.setLeafRank(leafRank)

        bestParams = []
        bestTrainAUCs = numpy.zeros(folds)
        bestTrainROCs = []
        bestTestAUCs = numpy.zeros(folds)
        bestTestROCs = []
        bestMetaDicts = []
        i = 0 

        for trainInds, testInds in indexList:
            trainX, trainY = X[trainInds, :], Y[trainInds]
            testX, testY = X[testInds, :], Y[testInds]

            meanParamAUCs = []
            paramList = [] 

            logging.debug("Distribution of labels in train: " + str(numpy.bincount(trainY)))
            logging.debug("Distribution of labels in test: " + str(numpy.bincount(testY)))

            for varSplit in varSplits:
                for nfcv in nfcvs:
                    for minSplit in minSplits:

                        self.setMaxDepth(maxDepths[0])
                        self.setVarSplit(varSplit)
                        self.setNfcv(nfcv)
                        self.setMinSplit(minSplit)
                        logging.debug(self)
                        idx = cross_val.StratifiedKFold(trainY, innerFolds)

                        j = 0
                        metrics = numpy.zeros((len(idx), maxDepths.shape[0]))

                        for idxtr, idxts in idx:
                            Util.printIteration(j, 1, innerFolds)

                            innerTrainX, innerTestX = trainX[idxtr, :], trainX[idxts, :]
                            innerTrainY, innerTestY = trainY[idxtr], trainY[idxts]

                            self.learnModel(innerTrainX, innerTrainY)

                            for k in range(maxDepths.shape[0]):
                                maxDepth = maxDepths[k]

                                robjects.globalenv["maxDepth"] = maxDepth
                                robjects.globalenv["tree"] = self.tree
                                nodeList = robjects.r('tree$nodes[tree$depth>=maxDepth]')
                                self.tree = self.treeRankLib.subTreeRank(self.tree, nodeList)

                                predY = self.predict(innerTestX)
                                gc.collect()

                                metrics[j, k] = Evaluator.auc(predY, innerTestY)
                                
                            j += 1

                        meanAUC = numpy.mean(metrics, 0)
                        varAUC = numpy.var(metrics, 0)
                        logging.warn(self.baseLib.warnings())
                        logging.debug("Mean AUCs and variances at each depth " + str((meanAUC, varAUC)))

                        for k in range(maxDepths.shape[0]):
                            maxDepth = maxDepths[k]
                            meanParamAUCs.append(meanAUC[k])
                            paramList.append((maxDepth, varSplit, nfcv, minSplit))

                        #Try to get some memory back
                        gc.collect()
                        robjects.r('gc(verbose=TRUE)')
                        robjects.r('memory.profile()')

                        #print(self.hp.heap())

            #Now choose best params
            bestInd = numpy.argmax(numpy.array(meanParamAUCs))

            self.setMaxDepth(paramList[bestInd][0])
            self.setVarSplit(paramList[bestInd][1])
            self.setNfcv(paramList[bestInd][2])
            self.setMinSplit(paramList[bestInd][3])

            self.learnModel(trainX, trainY)
            predTrainY = self.predict(trainX)
            predTestY = self.predict(testX)
            bestTrainAUCs[i] = Evaluator.auc(predTrainY, trainY)
            bestTestAUCs[i] = Evaluator.auc(predTestY, testY)

            #Store the parameters and ROC curves
            bestParams.append(paramList[bestInd])
            bestTrainROCs.append(Evaluator.roc(trainY, predTrainY))
            bestTestROCs.append(Evaluator.roc(testY, predTestY))

            metaDict = {}
            metaDict["size"] = self.getTreeSize()
            metaDict["depth"] = self.getTreeDepth()
            bestMetaDicts.append(metaDict)

            i += 1

        allMetrics = [bestTrainAUCs, bestTrainROCs, bestTestAUCs, bestTestROCs]

        return (bestParams, allMetrics, bestMetaDicts)

    def getTreeSize(self):
        return len(self.tree[2])

    def getTreeDepth(self):
        return numpy.max(numpy.array(self.tree[14]))

    def __str__(self):
        #Just write out the parameters
        outStr = "TreeRank:"
        if self.leafRank == self.treeRankLib.LRCart:
            outStr += " LeafRank=CART"
        elif self.leafRank == self.treeRankLib.LRsvm:
            outStr += " LeafRank=SVM"
        elif self.leafRank == self.treeRankLib.LRforest:
            outStr += " LeafRank=Random Forests"
        outStr += " maxDepth=" + str(self.maxDepth)
        outStr += " varSplit=" + str(self.varSplit)
        outStr += " nfcv=" + str(self.nfcv)
        outStr += " minSplit=" + str(self.minSplit)
        return outStr

    def getModel(self):
        return self.tree



        