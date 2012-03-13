import logging

from apgl.predictors.LibSVM import LibSVM
from apgl.data.ExamplesList import ExamplesList
from apgl.data.Standardiser import Standardiser
from apgl.egograph import *
from apgl.graph import * 
from apgl.io.EgoCsvReader import EgoCsvReader
from apgl.util import * 
import numpy
 
class SvmEgoSimulator(AbstractDiffusionSimulator):
    """
    A class which combines SVM classification with the EgoSimulation. There are methods
    to run modelSelection, train the SVM and then run the simulation. The simulation itself
    is run using EgoSimulator. 
    """
    def __init__(self, examplesFileName):
        """
        Create the class by reading examples from a Matlab file. Instantiate the SVM
        and create a preprocesor to standarise examples to have zero mean and unit variance. 
        """
        self.examplesList = ExamplesList.readFromFile(examplesFileName)
        self.examplesList.setDefaultExamplesName("X")
        self.examplesList.setLabelsName("y")

        (freqs, items) = Util.histogram(self.examplesList.getSampledDataField("y").ravel())
        logging.info("Distribution of labels: " + str((freqs, items)))
        logging.info("The base error rate is " + str(float(min(freqs))/self.examplesList.getNumExamples()))
        
        self.classifier = LibSVM()
        self.errorMethod = Evaluator.balancedError

        self.preprocessor = Standardiser()
        X = self.preprocessor.standardiseArray(self.examplesList.getDataField(self.examplesList.getDefaultExamplesName()))
        self.examplesList.overwriteDataField(self.examplesList.getDefaultExamplesName(), X)

    def getPreprocessor(self):
        """
        Returns the preprocessor
        """
        return self.preprocessor

    def sampleExamples(self, sampleSize):
        """
        This function exists so that we can sample the same examples used in model
        selection and exclude them when running evaluateClassifier. 
        """
        self.examplesList.randomSubData(sampleSize)

    def modelSelection(self, Cs, kernel, kernelParams, errorCosts, folds, sampleSize):
        """
        Perform model selection using an SVM
        """
        Parameter.checkInt(sampleSize, 0, self.examplesList.getNumExamples())
        Parameter.checkInt(folds, 0, sampleSize)
        Parameter.checkString(kernel, ["linear", "polynomial", "gaussian"])
        Parameter.checkList(Cs, Parameter.checkFloat, [0.0, float("inf")])
        Parameter.checkList(errorCosts, Parameter.checkFloat, [0.0, float("inf")])

        #Perform model selection
        self.examplesList.randomSubData(sampleSize)
        (freqs, items) = Util.histogram(self.examplesList.getSampledDataField("y").ravel())
        logging.info("Using "  + str(sampleSize) + " examples for model selection")
        logging.info("Distribution of labels: " + str((freqs, items)))
        logging.info("List of Cs " + str(Cs))
        logging.info("List of kernels " + str(kernel))
        logging.info("List of kernelParams " + str(kernelParams))
        logging.info("List of errorCosts " + str(errorCosts))

        CVal, kernelParamVal, errorCost, error = self.classifier.cvModelSelection(self.examplesList, Cs, kernelParams, kernel, folds, errorCosts, self.errorMethod)
        logging.info("Model selection returned C = " + str(CVal) + " kernelParam = " + str(kernelParamVal) + " errorCost = " + str(errorCost)  + " with error " + str(error))
        return CVal, kernelParamVal, errorCost, error

    def evaluateClassifier(self, CVal, kernel, kernelParamVal, errorCost, folds, sampleSize, invert=True):
        """
        Evaluate the SVM with the given parameters. Often model selection is done before this step
        and in that case, invert=True uses a sample excluding those used for model selection. 
        """
        Parameter.checkFloat(CVal, 0.0, float('inf'))
        Parameter.checkFloat(errorCost, 0.0, float('inf'))
        Parameter.checkString(kernel, ["linear", "polynomial", "gaussian"])
        
        if kernel == "gaussian":
            Parameter.checkFloat(kernelParamVal, 0.0, float('inf'))
        elif kernel == "polynomial":
            Parameter.checkInt(kernelParamVal, 2, float('inf'))

        Parameter.checkInt(folds, 0, sampleSize)
        Parameter.checkInt(sampleSize, 0, self.examplesList.getNumExamples())

        if invert:
            allIndices = numpy.array(list(range(0, self.examplesList.getNumExamples())))
            testIndices = numpy.setdiff1d(allIndices, self.examplesList.getPermutationIndices())
            testIndices = numpy.random.permutation(testIndices)[0:sampleSize]
        else:
            testIndices = Util.sampleWithoutReplacement(sampleSize, self.examplesList.getNumExamples())

        logging.info("Using " + str(testIndices.shape[0]) + " examples for SVM evaluation")

        self.examplesList.setPermutationIndices(testIndices)
        self.classifier.setParams(C=CVal, kernel=kernel, kernelParam=kernelParamVal)
        self.classifier.setErrorCost(errorCost)
        
        (means, vars) = self.classifier.evaluateCv(self.examplesList, folds)

        logging.info("--- Classification evaluation ---")
        logging.info("Error on " + str(testIndices.shape[0]) + " examples is " + str(means[0]) + "(" + str(vars[0]) + ")")
        logging.info("Sensitivity (recall = TP/(TP+FN)): " + str(means[1])  + "(" + str(vars[1]) + ")")
        logging.info("Specificity (TN/TN+FP): "  + str(means[2])  + "(" + str(vars[2]) + ")")
        logging.info("Error on positives: "  + str(means[3])  + "(" + str(vars[3]) + ")")
        logging.info("Error on negatives: "  + str(means[4])  + "(" + str(vars[4]) + ")")
        logging.info("Balanced error: "  + str(means[5])  + "(" + str(vars[5]) + ")")

        return (means, vars)

    def trainClassifier(self, CVal, kernel, kernelParamVal, errorCost, sampleSize):
        Parameter.checkFloat(CVal, 0.0, float('inf'))
        Parameter.checkString(kernel, ["linear", "gaussian", "polynomial"])
        Parameter.checkFloat(kernelParamVal, 0.0, float('inf'))
        Parameter.checkFloat(errorCost, 0.0, float('inf'))
        Parameter.checkInt(sampleSize, 0, self.examplesList.getNumExamples())

        logging.info("Training SVM with C=" + str(CVal) + ", " + kernel + " kernel" + ", param=" + str(kernelParamVal) + ", sampleSize=" + str(sampleSize) + ", errorCost=" + str(errorCost))

        self.examplesList.randomSubData(sampleSize)
        self.classifier.setC(C=CVal)
        self.classifier.setKernel(kernel=kernel, kernelParam=kernelParamVal)
        self.classifier.setErrorCost(errorCost)

        X = self.examplesList.getSampledDataField(self.examplesList.getDefaultExamplesName())
        y = self.examplesList.getSampledDataField(self.examplesList.getLabelsName())
        y = y.ravel()
        self.classifier.learnModel(X, y)

        return self.classifier

    def getWeights(self):
        return self.classifier.getWeights()


    def runSimulation(self, maxIterations):
        Parameter.checkInt(maxIterations, 1, float('inf'))

        #Notice that the data is preprocessed in the same way as the survey data
        egoSimulator = EgoSimulator(self.graph, self.classifier, self.preprocessor)

        totalInfo = numpy.zeros(maxIterations+1)
        totalInfo[0] = EgoUtils.getTotalInformation(self.graph)
        logging.info("Total number of people with information: " + str(totalInfo[0]))

        logging.info("--- Simulation Started ---")

        for i in range(0, maxIterations):
            logging.info("--- Iteration " + str(i) + " ---")

            self.graph = egoSimulator.advanceGraph()
            totalInfo[i+1] = EgoUtils.getTotalInformation(self.graph)
            logging.info("Total number of people with information: " + str(totalInfo[i+1]))

            #Compute distribution of ages etc. in alters
            alterIndices = egoSimulator.getAlters(i)
            alterAges = numpy.zeros(len(alterIndices))
            alterGenders = numpy.zeros(len(alterIndices))

            for j in range(0, len(alterIndices)):
                currentVertex = self.graph.getVertex(alterIndices[j])
                alterAges[j] = currentVertex[self.egoQuestionIds.index(("Q5X", 0))]
                alterGenders[j] = currentVertex[self.egoQuestionIds.index(("Q4", 0))]

            (freqs, items) = Util.histogram(alterAges)
            logging.info("Distribution of ages " + str(freqs) + " " + str(items))
            (freqs, items) = Util.histogram(alterGenders)
            logging.info("Distribution of genders " + str(freqs) + " " + str(items))
            
        logging.info("--- Simulation Finished ---")

        return totalInfo, egoSimulator.getTransmissions()

    def getVertexFeatureDistribution(self, fIndex, vIndices=None):
        return self.graph.getVertexFeatureDistribution(fIndex, vIndices)

    def getPreProcessor(self):
        return self.preprocessor

    def getClassifier(self):
        return self.classifier 

    preprocessor = None
    examplesList = None
    classifier = None
    graph = None
    edgeWeight = 1 