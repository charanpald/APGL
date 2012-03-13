
from apgl.predictors.AbstractPredictor import AbstractPredictor
from apgl.predictors.AbstractPredictor import computeTestError, computeBootstrapError, computePenalisedError, computePenalty, computeIdealPenalty
from apgl.util.Parameter import Parameter
from apgl.util.Sampling import Sampling
from apgl.util.Evaluator import Evaluator
from apgl.util.Util import Util
import os 
import sys
import logging 
import numpy
import scipy
import scipy.sparse
import multiprocessing
import itertools


class LibSVM(AbstractPredictor):
    def __init__(self, kernel='linear', kernelParam=0.1, C=1.0, cost=1.0, type="C_SVC", processes=None, epsilon=0.001):
        try:
            from sklearn.svm import SVC 
        except ImportError:
            raise 
            
        super(LibSVM, self).__init__()

        self.C = C
        self.errorCost = cost
        self.epsilon = epsilon
        self.type = type
        self.tol = 0.001
        self.setKernel(kernel, kernelParam)
        self.setSvmType(type)

        self.processes = processes
        self.chunkSize = 10
        self.timeout = 5

        #Parameters used for model selection
        self.Cs = 2.0**numpy.arange(-10, 20, dtype=numpy.float)
        self.gammas = 2.0**numpy.arange(-10, 4, dtype=numpy.float)
        self.epsilons = 2.0**numpy.arange(-4, -2, dtype=numpy.float)

    def getCs(self):
        return self.Cs

    def getGammas(self):
        return self.gammas

    def getEpsilons(self):
        return self.epsilons

    def __updateParams(self):
        try:
            from sklearn.svm import SVC, SVR
        except:
            raise

        if self.type == "Epsilon_SVR":
            self.model = SVR(C=self.C, kernel=self.kernel, degree=self.kernelParam, gamma=self.kernelParam, epsilon=self.epsilon, tol=self.tol)
        elif self.type == "C_SVC":
            self.model = SVC(C=self.C, kernel=self.kernel, degree=self.kernelParam, gamma=self.kernelParam, tol=self.tol)
        else:
            raise ValueError("Invalid type : " + str(type))

    def setC(self, C):
        try:
            from sklearn.svm import SVC
        except:
            raise
        Parameter.checkFloat(C, 0.0, float('inf'))
        
        self.C = C
        self.__updateParams()


    def setKernel(self, kernel='linear', kernelParam=0.1):
        try:
            from sklearn.svm import SVC 
        except:
            raise 
        Parameter.checkString(kernel, ["linear", "gaussian", "rbf", "polynomial"])

        if kernel=="gaussian" or kernel=="rbf":
            self.kernel = "rbf"
        elif kernel=="polynomial":
            self.kernel = "poly"
        else:
            self.kernel = kernel

        self.kernelParam = kernelParam
        self.__updateParams()

    def setGamma(self, gamma):
        self.kernelParam = gamma
        self.__updateParams()

    def setSvmType(self, type):
        try:
            from sklearn.svm import SVC, SVR
        except:
            raise
        """
        Set the type as "Epsilon_SVR" or "C_SVC"
        """

        self.type = type
        self.__updateParams()

    def getMetricMethod(self):
        """

        Depending on the type "Epsilon_SVR" or "C_SVC" returns a way to measure
        the performance of the classifier.
        """
        if self.getType() == "C_SVC":
            return Evaluator.binaryError
        else:
            return Evaluator.rootMeanSqError
        

    def setErrorCost(self, errorCost):
        """
        The penalty on errors on positive labels. The penalty for negative labels
        is 1.
        """
        Parameter.checkFloat(errorCost, 0.0, float('inf'))
        self.errorCost = errorCost

    def setEpsilon(self, epsilon):
        """
        This is for backward compatibility only
        """
        self.epsilon = epsilon
        self.__updateParams()

    def setTermination(self, tol):
        Parameter.checkFloat(tol, 0.0, 1.0)
        self.tol = tol
        self.__updateParams()

    def learnModel(self, X, y):
        try:
            from sklearn.svm import SVC 
        except:
            raise 

        self.model.fit(X, y, class_weight={-1:1, 1:self.errorCost})

    def classify(self, X):
        try:
            from sklearn.svm import SVC 
        except:
            raise
        
        return self.predict(X, False)

    def predict(self, X, decisionValues=False): 
        """
        Classify a set of examples and return an array of predicted labels
        """
        try:
            from sklearn.svm import SVC 
        except:
            raise

        if decisionValues:
            return self.model.predict(X), self.model.decision_function(X)
        else:
            return self.model.predict(X)
        
    def getWeights(self):
        """
        A method to get the weight vector and bias b. The decision function is made
        using the sign of numpy.dot(X, weights) + b.
        """
        try:
            from sklearn.svm import SVC
        except:
            raise

        w = self.model.coef_
        b = self.model.intercept_ 

        return w, b

    def loadParams(self, fileName):
        """
        Load a set of SVM parameters from a matlab file. 
        """
        try:
            matDict = scipy.io.loadmat(fileName)
        except IOError:
            raise

        if "C" in list(matDict.keys()):
            self.setC(float(matDict["C"][0]))

        if "kernel" in list(matDict.keys()):
            self.setKernel(str(matDict["kernel"][0]), float(matDict["kernelParamVal"][0])) 

        if "errorCost" in list(matDict.keys()):
            self.setErrorCost(float(matDict["errorCost"][0]))

        if "epsilon" in list(matDict.keys()):
            self.setTermination(float(matDict["epsilon"][0]))

        if "svmType" in list(matDict.keys()):
            self.setSvmType(str(matDict["svmType"][0]))

    def saveParams(self, fileName):
        raise NotImplementedError("This is not implemented")

    def getC(self):
        return self.C

    def getErrorCost(self):
        return self.errorCost 

    def getKernel(self):
        return self.kernelStr

    def getEpsilon(self):
        return self.epsilon

    def getType(self):
        return self.type

    def getKernelParams(self):
        return self.kernelParam

    def getGamma(self):
        return self.kernelParam

    def __str__(self):
        return str(self.model)

    def getModel(self):
        return self.model

    def parallelVfcvRbf(self, X, y, idx, type="C_SVC"):
        """
        Perform parallel cross validation model selection using the RBF kernel
        and then pick the best one. Using the best set of parameters train using
        the whole dataset.

        :param X: The examples as rows
        :type X: :class:`numpy.ndarray`

        :param y: The binary -1/+1 labels 
        :type y: :class:`numpy.ndarray`

        :param idx: A list of train/test splits

        :params returnGrid: Whether to return the error grid
        :type returnGrid: :class:`bool`
        """
        Parameter.checkClass(X, numpy.ndarray)
        Parameter.checkClass(y, numpy.ndarray)
        folds = len(idx)

        self.setKernel("gaussian")

        if type=="C_SVC":
            paramDict = {} 
            paramDict["setC"] = self.getCs()
            paramDict["setGamma"] = self.getGammas()  
        else: 
            paramDict = {} 
            paramDict["setC"] = self.getCs()
            paramDict["setGamma"] = self.getGammas()  
            paramDict["setEpsilon"] = self.getEpsilons()  
                
        return self.parallelModelSelect(X, y, idx, paramDict)


    def copy(self): 
        """
        Return a new copied version of this object. 
        """
        svm = LibSVM(kernel=self.kernel, kernelParam=self.kernelParam, C=self.C, cost=self.errorCost, type=self.type, processes=self.processes, epsilon=self.epsilon)
        return svm 


    def parallelVfPenRbf(self, X, y, idx, Cvs, type="C_SVC"):
        """
        Perform v fold penalisation model selection using the RBF kernel
        and then pick the best one. Using the best set of parameters train using
        the whole dataset. Cv is the control on the amount of penalisation.

        :param X: The examples as rows
        :type X: :class:`numpy.ndarray`

        :param y: The binary -1/+1 labels
        :type y: :class:`numpy.ndarray`

        :param idx: A list of train/test splits
        """
        Parameter.checkClass(X, numpy.ndarray)
        Parameter.checkClass(y, numpy.ndarray)
        Parameter.checkClass(Cvs, numpy.ndarray)
        
        self.setKernel("gaussian")

        if type=="C_SVC":
            paramDict = {} 
            paramDict["setC"] = self.getCs()
            paramDict["setGamma"] = self.getGammas()  
        else: 
            paramDict = {} 
            paramDict["setC"] = self.getCs()
            paramDict["setGamma"] = self.getGammas()  
            paramDict["setEpsilon"] = self.getEpsilons()  
                
        return self.parallelPen(X, y, idx, paramDict, Cvs)

    def parallelPenaltyGridRbf(self, trainX, trainY, fullX, fullY, type="C_SVC"):
        """
        Find out the "ideal" penalty. 
        """
        Parameter.checkClass(trainX, numpy.ndarray)
        Parameter.checkClass(trainY, numpy.ndarray)
        if type=="C_SVC":
            paramDict = {} 
            paramDict["setC"] = self.getCs()
            paramDict["setGamma"] = self.getGammas()  
        else: 
            paramDict = {} 
            paramDict["setC"] = self.getCs()
            paramDict["setGamma"] = self.getGammas()  
            paramDict["setEpsilon"] = self.getEpsilons()  
                
        return self.parallelPenaltyGrid(trainX, trainY, fullX, fullY, paramDict)


        