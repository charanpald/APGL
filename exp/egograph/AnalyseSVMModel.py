import logging
import sys
import numpy
import random
from apgl.egograph.InfoExperiment import InfoExperiment
from apgl.egograph.SvmEgoSimulator import SvmEgoSimulator
from apgl.io.EgoCsvReader import EgoCsvReader

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

numpy.random.seed(21)
random.seed(21)

examplesFileName = SvmInfoExperiment.getExamplesFileName()
sampleSize = 86755

svmEgoSimulator = SvmEgoSimulator(examplesFileName)
preprocessor = svmEgoSimulator.getPreProcessor()
centerValues = preprocessor.getCentreVector()

svmParamsFileName = SvmInfoExperiment.getSvmParamsFileName() + "Linear.mat"
logging.info("Using SVM params from file " + svmParamsFileName)

C, kernel, kernelParamVal, errorCost = SvmInfoExperiment.loadSvmParams(svmParamsFileName)
svmEgoSimulator.trainClassifier(C, kernel, kernelParamVal, errorCost, sampleSize)

weights, b  = svmEgoSimulator.getWeights()

numpy.set_printoptions(precision=3)

#Print the weights then their sorted values by indices and then value
sortedWeightsInds = numpy.flipud(numpy.argsort(abs(weights)))
sortedWeights = numpy.flipud(weights[numpy.argsort(abs(weights))])

egoCsvReader = EgoCsvReader()
questionIds = egoCsvReader.getEgoQuestionIds()
questionIds.extend(egoCsvReader.getAlterQuestionIds())

print(weights)
numRankedItems = 20

for i in range(0,numRankedItems):
    print((str(centerValues[sortedWeightsInds[i]]) + " & " + questionIds[sortedWeightsInds[i]][0] + " & " + str("%.3f" % sortedWeights[i]) + "\\\\"))
print(b)