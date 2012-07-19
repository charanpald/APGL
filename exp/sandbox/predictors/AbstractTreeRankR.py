import gc 
import numpy
import logging
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects.numpy2ri import numpy2ri
from apgl.predictors.AbstractPredictor import AbstractPredictor
from apgl.util.Parameter import Parameter
from apgl.util.PathDefaults import PathDefaults
from apgl.util.ProfileUtils import ProfileUtils

"""
Abstract out some of the common functionality between TreeRankR and TreeRankForestR
"""

class AbstractTreeRankR(AbstractPredictor):
    def __init__(self):
        self.treeRankLib = importr('TreeRank')
        self.baseLib = importr('base')

        self.bestResponse = 1
        self.varSplit = 1.0
        self.maxDepth = 10
        self.leafRank = self.treeRankLib.LRCart
        self.nfcv = 0
        self.minSplit = 50
        self.growing = self.treeRankLib.growing_ctrl(minsplit=self.minSplit, maxdepth=self.maxDepth, mincrit=0)

        robjects.conversion.py2ri = numpy2ri
        self.printMemStats = False
        self.printDebug = False 
        self.tree = None 

    def setMaxDepth(self, maxDepth):
        self.maxDepth = maxDepth
        self.growing = self.treeRankLib.growing_ctrl(minsplit=self.minSplit, maxdepth=self.maxDepth, mincrit=0)

    def setMinSplit(self, minSplit):
        self.minSplit = minSplit
        self.growing = self.treeRankLib.growing_ctrl(minsplit=self.minSplit, maxdepth=self.maxDepth, mincrit=0)

    def setVarSplit(self, varSplit):
        self.varSplit = varSplit

    def setNfcv(self, nfcv):
        self.nfcv = nfcv

    def _getDataFrame(self, X, Y):
        """
        Create a DataFrame from numpy arrays X and Y
        """
        Parameter.checkClass(X, numpy.ndarray)
        Parameter.checkClass(Y, numpy.ndarray)

        X = self.baseLib.data_frame(robjects.vectors.Matrix(X))
        Y = self.baseLib.data_frame(robjects.vectors.Matrix(Y))

        XY = X.cbind(Y)
        XY.names[len(XY.names)-1] = "class"
        return XY

    def getTreeRankLib(self):
        return self.treeRankLib

    def getModel(self):
        Util.abstract()

    def predictScores(self, X):
        """
        Make predictions using the learnt tree. Returns the scores as a numpy array.
        """
        Parameter.checkClass(X, numpy.ndarray)

        predictFunc = robjects.r['predict']
        X = self.baseLib.data_frame(X)
        scores = self.baseLib.matrix(predictFunc(self.getModel(), X))
        return numpy.asarray(scores).ravel()

    def predict(self, X):
        """
        Basically, return the scores.
        """
        Parameter.checkClass(X, numpy.ndarray)

        scores = self.predictScores(X)
        return scores

    def setLeafRank(self, leafRank):
        """
        Set the leaf rank procedure using getTreeRankLib(self).LRCart, LRsvm
        or LRforest
        """
        self.leafRank = leafRank

    def aucFromROC(self, roc):
        """
        Get the AUC value from the ROC curve
        """
        return self.treeRankLib.auc(roc)[0]

    def getLsos(self):
        """
        Return a function to display R memory usage
        """
        fileName = PathDefaults.getSourceDir() + "/apgl/metabolomics/R/Util.R"
        robjects.r["source"](fileName)
        return robjects.r['lsos']

    def __loadLeafRanks(self):
        utilFileName = PathDefaults.getSourceDir() + "/apgl/metabolomics/R/Util.R"
        leafRanksFileName = PathDefaults.getSourceDir() + "/apgl/metabolomics/R/MSLeafRanks.R"
        robjects.r["source"](utilFileName)
        robjects.r["source"](leafRanksFileName)

    def getLrRbfSvm(self):
        """
        Return a version of the LRsvm leafrank algorithm which uses model
        selection to select C.
        """
        self.__loadLeafRanks()
        return robjects.r['LRsvm2']

    def getLrRbfSvmF(self):
        """
        Return a version of the LRsvm leafrank algorithm which uses model
        selection to select C.
        """
        self.__loadLeafRanks()
        return robjects.r['LRsvmF']

    def getLrLinearSvmPlain(self):
        """
        Return a linear SVM which does not do model selection. 
        """
        self.__loadLeafRanks()
        return robjects.r['LRsvmLinearPlain']

    def getLrLinearSvm(self):
        """
        Return a version of the LRsvm leafrank algorithm which uses model
        selection to select C.
        """
        self.__loadLeafRanks()
        return robjects.r['LRsvmLinear']

    def getLrCart(self):
        """
        Return a version of the LRsvm leafrank algorithm which uses model
        selection to select C.
        """
        self.__loadLeafRanks()
        return robjects.r['LRCart2']

    def getLrCartF(self):
        """
        Return a version of the LRsvm leafrank algorithm which uses model
        selection to select C.
        """
        self.__loadLeafRanks()
        return robjects.r['LRCartF']

    def getLrForest(self):
        """
        Return a version of the LRsvm leafrank algorithm which uses model
        selection to select C.
        """
        self.__loadLeafRanks()
        return robjects.r['LRforest2']

    def learnModel(self, X, Y):
        Parameter.checkClass(X, numpy.ndarray)
        Parameter.checkClass(Y, numpy.ndarray)
        Parameter.checkArray(X)
        Parameter.checkArray(Y)

        if numpy.unique(Y).shape[0] < 2:
            raise ValueError("Vector of labels must be binary, currently numpy.unique(Y) = " + str(numpy.unique(Y)))

        #If Y is 1D make it 2D
        if Y.ndim == 1:
            Y = numpy.array([Y]).T
        
        XY = self._getDataFrame(X, Y)
        formula = robjects.Formula('class ~ .')
        self.learnModelDataFrame(formula, XY)

        gc.collect()
        robjects.r('gc(verbose=TRUE)')
        robjects.r('memory.profile()')
        gc.collect()

        if self.printMemStats:
            logging.debug(self.getLsos()())
            logging.debug(ProfileUtils.memDisplay(locals()))

    def predictROC(self, X, Y):
        """
        Make predictions using the learnt tree. Returns the ROC curve as a numpy
        array
        """
        Parameter.checkClass(X, numpy.ndarray)
        Parameter.checkClass(Y, numpy.ndarray)

        XY = self._getDataFrame(X, Y)
        XYROC = self.treeRankLib.getROC(self.getModel(), XY)
        return numpy.array(XYROC)