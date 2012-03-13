
import logging
import sys
import numpy
import random
from apgl.egograph.InfoExperiment import InfoExperiment
from apgl.egograph.SvmEgoSimulator import SvmEgoSimulator

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

numpy.random.seed(21)
random.seed(21)

examplesFileName = SvmInfoExperiment.getExamplesFileName()
folds = 5
cvSampleSize = SvmInfoExperiment.getNumCVExamples()
sampleSize = 75000

svmEgoSimulator = SvmEgoSimulator(examplesFileName)
classifier = svmEgoSimulator.getClassifier()
#classifier.setTermination(0.1)

C, kernel, kernelParamVal, errorCost = SvmInfoExperiment.loadSvmParams(SvmInfoExperiment.getSvmParamsFileName())

logging.info("Training SVM with C=" + str(C) + ", " + kernel + " kernel" + ", param=" + str(kernelParamVal) + ", sampleSize=" + str(sampleSize) + ", errorCost="+str(errorCost))

svmEgoSimulator.sampleExamples(cvSampleSize)
svmEgoSimulator.evaluateClassifier(C, kernel, kernelParamVal, errorCost, folds, sampleSize, True)
