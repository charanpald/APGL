
import logging
import numpy.random
import scipy.io
import time
from exp.egograph.SvmEgoSimulator import SvmEgoSimulator
from exp.egograph.EgoUtils import EgoUtils
from apgl.util.PathDefaults import PathDefaults

"""
This class does parameter selection for the SVM in an ego simulation and then
runs the experiments and saves results.

It is a terrible class. 
"""

class SvmInfoExperiment:
    @staticmethod
    def trainSVM():
        """
        Train the support vector machine for the simulation. 
        """
        examplesFileName = SvmInfoExperiment.getExamplesFileName()
        svmParamsFile = SvmInfoExperiment.getSvmParamsFileName()
        sampleSize = SvmInfoExperiment.getNumSimulationExamples()

        simulator = SvmEgoSimulator(examplesFileName)
        CVal, kernel, kernelParamVal, errorCost = SvmInfoExperiment.loadSvmParams(svmParamsFile)

        startTime = time.time()
        simulator.trainClassifier(CVal, kernel, kernelParamVal, errorCost, sampleSize)
        endTime = time.time()
        logging.info("Time taken for SVM training is " + str((endTime-startTime)) + " seconds.")

        return simulator

    @staticmethod
    def runExperiment(graphType, p, k, infoProb, simulator):
        """
        Run an experiment to compare information spread over a random graph. The
        graphType is (SmallWorld|ErdosRenyi), p is rewire probability, k is the
        number of neighbours, and infoProb is the initial number of people with
        information.
        """
        startTime = time.time()

        egoFileName = "../../data/EgoData.csv"
        alterFileName = "../../data/AlterData.csv"
        numVertices =  SvmInfoExperiment.getNumVertices()
        maxIterations = 5

        simulationRepetitions = 5
        totalInfo = numpy.zeros((simulationRepetitions, maxIterations+1))
        averageHops = numpy.zeros(simulationRepetitions)
        receiversToSenders = numpy.zeros(simulationRepetitions)

        for i in range(0, simulationRepetitions):
            simulator.generateRandomGraph(egoFileName, alterFileName, numVertices, infoProb, graphType, p, k)
            (totalInfo[i, :], transmissions) = simulator.runSimulation(maxIterations)
            averageHops[i] = EgoUtils.averageHopDistance(transmissions)
            receiversToSenders[i] = EgoUtils.receiversPerSender(transmissions)

        outputFileName = SvmInfoExperiment.getOutputFileName(graphType, p, k, infoProb)

        endTime = time.time()
        logging.info("Total time taken is " + str((endTime-startTime)) + " seconds.")

        logging.info("totalInfo: "  + str(totalInfo))
        logging.info("averageHops: " + str(averageHops))
        logging.info("receiversToSenders: " + str(receiversToSenders))

        #Save variables
        matDict = {}
        matDict["totalInfo"] = totalInfo
        matDict["averageHops"] = averageHops
        matDict["receiversToSenders"] = receiversToSenders
        matDict["numVertices"] = numVertices
        matDict["p"] = numpy.array([p])
        matDict["k"] = numpy.array([k])
        matDict["q"] = numpy.array([infoProb])
        scipy.io.savemat(outputFileName, matDict)
        logging.info("Saved file as " + outputFileName)

    @staticmethod
    def getExamplesFileName():
        dataDir = PathDefaults.getDataDir() + "infoDiffusion/"
        return dataDir + "EgoAlterTransmissions.mat"
        #return "../../data/EgoAlterTransmissions2.csv"

    @staticmethod
    def getSvmParamsFileName():
        return PathDefaults.getOutputDir() + "diffusion/" + "svmParams.mat"

    def getLinearSvmParamsFileName():
        return PathDefaults.getOutputDir() + "diffusion/" + "svmParamsLinear.mat"

    @staticmethod
    def getNumVertices():
        return 10000

    @staticmethod
    def getNumCVExamples():
        return 15000

    @staticmethod
    def getNumSimulationExamples():
        return 20000

    @staticmethod
    def getOutputFileName(graphType, p, k, infoProb):
        outputDirectory = PathDefaults.getOutputDir()

        if graphType == "SmallWorld":
            outputFileName = outputDirectory + "SvmEgoOutput_type=" + graphType + "_p=" + str(p) + "_k=" + str(k) + "_q=" + str(infoProb)
        elif graphType == "ErdosRenyi":
            outputFileName = outputDirectory + "SvmEgoOutput_type=" + graphType + "_p=" + str(p) + "_q=" + str(infoProb)
        else:
            raise ValueError("Invalid graph type: " + graphType)

        return outputFileName


    @staticmethod
    def saveSvmParams(svmParamsFile):
        """
        This method runs model selection for the SVM and saves the parameters
        and errors. Only need to do this once for the data.
        """
        
        examplesFileName = SvmInfoExperiment.getExamplesFileName()
        
        folds = 4
        kernels = ["linear", "gaussian"]
        sampleSize = SvmInfoExperiment.getNumCVExamples()
        Cs = [4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0]
        errorCosts = [8.0, 16.0, 32.0]

        simulator = SvmEgoSimulator(examplesFileName)
        error = 1

        for i in range(0, len(kernels)):
            logging.info("Using " + kernels[i] + " kernel")
            kernel = kernels[i]

            if kernel == "linear":
                folds = 3
            else:
                folds = 3

            if kernel == "gaussian":
                Cs = [4.0, 16.0, 64.0, 256.0, 1024.0, 4096.0]
                kernelParams = [0.125, 0.25, 0.5, 1.0, 2.0]
            elif kernel == "polynomial":
                kernelParams = [2, 3, 4]
            elif kernel == "linear":
                kernelParams = [0.0]
            else:
                raise ValueError("Invalid kernel: " + kernel)

            CVal, kernelParamVal, errorCost, msError = simulator.modelSelection(Cs, kernel, kernelParams, errorCosts, folds, sampleSize)
            (means, vars) = simulator.evaluateClassifier(CVal, kernel, kernelParamVal, errorCost, folds, sampleSize)
        
            matDict = {}
            matDict["C"] = numpy.array([CVal])
            matDict["kernel"] = kernel
            matDict["kernelParamVal"] = numpy.array([kernelParamVal])
            matDict["errorCost"] = numpy.array([errorCost])
            matDict["sampleSize"] = numpy.array([sampleSize])
            matDict["msError"] = msError
            matDict["means"] = means
            matDict["vars"] = vars

            scipy.io.savemat(svmParamsFile + kernel[0].upper() + kernel[1:], matDict)
            logging.info("Saved file as " + svmParamsFile + kernel[0].upper() + kernel[1:])

            if msError < error:
                error = msError
                scipy.io.savemat(svmParamsFile, matDict)
                logging.info("Saved file as " + svmParamsFile)
            

    @staticmethod
    def loadSvmParams(svmParamsFile):
        try:
            matDict = scipy.io.loadmat(svmParamsFile)
        except IOError:
            raise 

        C = float(matDict["C"][0])
        kernel =  str(matDict["kernel"][0])
        kernelParamVal = float(matDict["kernelParamVal"][0])
        errorCost = float(matDict["errorCost"][0])

        return C, kernel, kernelParamVal, errorCost
