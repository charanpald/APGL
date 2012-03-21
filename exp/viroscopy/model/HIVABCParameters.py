"""
This is just a class which represents a set of parameters for ABC model selection. 
"""
import numpy
import logging
import scipy.stats as stats
from apgl.util.Parameter import Parameter
from apgl.util.PathDefaults import PathDefaults
from apgl.util.Util import Util
from exp.viroscopy.model.HIVVertices import HIVVertices

class HIVABCParameters(object):
    def __init__(self, meanTheta):
        """
        Initialised this object with a mean value of theta 
        """

        self.paramFuncs = []
        self.priorDists = []
        self.priorDensities = []
        self.purtubationKernels = []
        self.purtubationKernelDensities = []

        purtScale = 0.2
        sigmaScale = 0.5

        #Now set up all the parameters
        ind = 0 
        min = meanTheta[ind]-15
        max = meanTheta[ind]+15
        priorDist = lambda: stats.randint.rvs(min, max)
        priorDensity = lambda x: stats.randint.pmf(x, min, max)
        purtubationKernel = lambda x: stats.randint.rvs(x-int(min*purtScale), x+int(min*purtScale))
        purtubationKernelDensity = lambda old, new: stats.randint.pmf(new, old-int(min*purtScale), old+int(max*purtScale))
        self.__addParameter(("graph", "setRandomInfected"), priorDist, priorDensity, purtubationKernel, purtubationKernelDensity)

        ind += 1
        mu = meanTheta[ind]
        sigma = mu*sigmaScale
        priorDist, priorDensity = self.createGammaParam(sigma, mu)
        purtubationKernel, purtubationKernelDensity = self.__createNormalPurt(sigma, purtScale)
        self.__addParameter(("rates", "setAlpha"), priorDist, priorDensity, purtubationKernel, purtubationKernelDensity)

        ind += 1
        mu = meanTheta[ind]
        sigma = mu*sigmaScale
        priorDist, priorDensity = self.createGammaParam(sigma, mu)
        purtubationKernel, purtubationKernelDensity = self.__createNormalPurt(sigma, purtScale)
        self.__addParameter(("rates", "setNewContactChance"), priorDist, priorDensity, purtubationKernel, purtubationKernelDensity)

        ind += 1
        mu = meanTheta[ind]
        sigma = mu*sigmaScale
        priorDist, priorDensity = self.createGammaParam(sigma, mu)
        purtubationKernel, purtubationKernelDensity = self.__createNormalPurt(sigma, purtScale)
        self.__addParameter(("rates", "setRandDetectRate"), priorDist, priorDensity, purtubationKernel, purtubationKernelDensity)

        ind += 1
        mu = meanTheta[ind]
        sigma = mu*sigmaScale
        priorDist, priorDensity = self.createGammaParam(sigma, mu)
        purtubationKernel, purtubationKernelDensity = self.__createNormalPurt(sigma, purtScale)
        self.__addParameter(("rates", "setCtRatePerPerson"), priorDist, priorDensity, purtubationKernel, purtubationKernelDensity)

        ind += 1
        mu = meanTheta[ind]
        sigma = mu*sigmaScale
        priorDist, priorDensity = self.createGammaParam(sigma, mu)
        purtubationKernel, purtubationKernelDensity = self.__createNormalPurt(sigma, purtScale)
        self.__addParameter(("rates", "setHeteroContactRate"), priorDist, priorDensity, purtubationKernel, purtubationKernelDensity)

        ind += 1
        mu = meanTheta[ind]
        sigma = mu*sigmaScale
        priorDist, priorDensity = self.createGammaParam(sigma, mu)
        purtubationKernel, purtubationKernelDensity = self.__createNormalPurt(sigma, purtScale)
        self.__addParameter(("rates", "setBiContactRate"), priorDist, priorDensity, purtubationKernel, purtubationKernelDensity)

        ind += 1
        mu = meanTheta[ind]
        sigma = mu*sigmaScale
        priorDist, priorDensity = self.createTruncNormParam(sigma, mu)
        purtubationKernel, purtubationKernelDensity = self.__createNormalPurt(sigma, purtScale)
        self.__addParameter(("rates", "setWomanManInfectProb"), priorDist, priorDensity, purtubationKernel, purtubationKernelDensity)

        ind += 1
        mu = meanTheta[ind]
        sigma = mu*sigmaScale
        priorDist, priorDensity = self.createTruncNormParam(sigma, mu)
        purtubationKernel, purtubationKernelDensity = self.__createNormalPurt(sigma, purtScale)
        self.__addParameter(("rates", "setManWomanInfectProb"), priorDist, priorDensity, purtubationKernel, purtubationKernelDensity)

        ind += 1
        mu = meanTheta[ind]
        sigma = mu*sigmaScale
        priorDist, priorDensity = self.createTruncNormParam(sigma, mu)
        purtubationKernel, purtubationKernelDensity = self.__createNormalPurt(sigma, purtScale)
        self.__addParameter(("rates", "setManBiInfectProb"), priorDist, priorDensity, purtubationKernel, purtubationKernelDensity)

    def createTruncNormParam(self, sigma, mode):
        """
        Truncated norm parameter between 0 and 1 
        """
        Parameter.checkFloat(sigma, 0.0, 1.0)
        Parameter.checkFloat(mode, 0.0, float('inf'))
        a = -mode/sigma
        b = (1-mode)/sigma
        priorDist = lambda: stats.truncnorm.rvs(a, b, loc=mode, scale=sigma)
        priorDensity = lambda x: stats.truncnorm.pdf(x, a, b, loc=mode, scale=sigma)
        return priorDist, priorDensity 

    def createGammaParam(self, sigma, mu):
        Parameter.checkFloat(sigma, 0.0, float('inf'))
        Parameter.checkFloat(mu, 0.0, float('inf'))

        if mu == 0.0:
            raise ValueError("Gamma distribution cannot have mean zero.")

        theta = sigma**2/mu
        k = mu/theta

        k = min(k, 1000)

        priorDist = lambda: stats.gamma.rvs(k, scale=theta)
        priorDensity = lambda x: stats.gamma.pdf(x, k, scale=theta)

        return priorDist, priorDensity 

    def __createNormalPurt(self, sigma, purtScale):
        Parameter.checkFloat(sigma, 0.0, float('inf'))
        Parameter.checkFloat(purtScale, 0.0, float('inf'))
        purtubationKernel = lambda x: stats.norm.rvs(x, sigma*purtScale)
        purtubationKernelDensity = lambda old, new: stats.norm.pdf(new, old, sigma*purtScale)
        return purtubationKernel, purtubationKernelDensity

    def __addParameter(self, paramFunc, priorDist, priorDensity, purtubationKernel, purtubationKernelDensity):
        self.paramFuncs.append(paramFunc)
        self.priorDists.append(priorDist)
        self.priorDensities.append(priorDensity)
        self.purtubationKernels.append(purtubationKernel)
        self.purtubationKernelDensities.append(purtubationKernelDensity)

    def getParamFuncs(self):
        return self.paramFuncs

    def sampleParams(self):
        theta = []

        for priorDist in self.priorDists:
            theta.append(priorDist())
            
        theta = numpy.array(theta)
        return theta

    def priorDensity(self, theta):
        density = 1

        for i in range(len(self.priorDensities)):
            priorDensityFunc = self.priorDensities[i]
            density *= priorDensityFunc(theta[i])

        return density

    def purtubationKernel(self, theta):
        newTheta = []

        for i in range(len(self.purtubationKernels)):
            newTheta.append(self.purtubationKernels[i](theta[i]))
        
        newTheta = numpy.array(newTheta)
        return newTheta

    def purtubationKernelDensity(self, oldTheta, theta):
        density = 1

        for i in range(len(self.purtubationKernelDensities)):
            purtubationKernelDensity = self.purtubationKernelDensities[i]
            density *= purtubationKernelDensity(oldTheta[i], theta[i])

        return density

