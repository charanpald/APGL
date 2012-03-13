
from apgl.graph import *
from apgl.util import *
from apgl.data import *
from apgl.predictors import *
import numpy
import logging
import sys
import pickle 

import matplotlib
import matplotlib.pyplot as plt

"""
Process the results from the synthetic dataset. 
"""

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
ks = list(range(10, 310, 10))
noises = [0.05, 0.1, 0.15]
numVertices = 500 

matplotlib.rcParams['ps.useafm'] = True

outputDir = PathDefaults.getOutputDir() + "influence/"
svmMeanInfErrors = numpy.zeros((len(ks), len(noises)))
svmStdInfErrors = numpy.zeros((len(ks), len(noises)))
egoMeanInfErrors = numpy.zeros((len(ks), len(noises)))
egoStdInfErrors = numpy.zeros((len(ks), len(noises)))

svmMeanErrors = numpy.zeros(len(noises))
svmStdErrors = numpy.zeros(len(noises))

egoMeanErrors = numpy.zeros(len(noises))
egoStdErrors = numpy.zeros(len(noises))

#Print all the parameters 
for i in range(len(noises)):
    noise = noises[i]

    paramsFile = outputDir + "SvmParamsLinear_n=" + str(noise)
    file = open(paramsFile, "rb")
    paramsList = pickle.load(file)
    print(paramsList)

    paramsFile = outputDir + "EgoParamsLinear_n=" + str(noise)
    file = open(paramsFile, "rb")
    paramsList = pickle.load(file)
    print(paramsList)

print("\n")

for i in range(len(noises)):
    noise = noises[i]
    fileName = outputDir + "influenceErrorsSvm_n=" + str(noise) + ".npz"
    errorDict = numpy.load(fileName)
    svmMeanInfErrors[:, i] = errorDict["arr_0"]
    svmStdInfErrors[:, i] = errorDict["arr_1"]

    fileName = outputDir + "influenceErrorsEgo_n=" + str(noise) + ".npz"
    errorDict = numpy.load(fileName)
    egoMeanInfErrors[:, i] = errorDict["arr_0"]
    egoStdInfErrors[:, i] = errorDict["arr_1"]

    svmResultsFile = outputDir + "SvmResults_n=" + str(noise) + ".npz"
    errorDict = numpy.load(svmResultsFile)
    svmMeanErrors[i] = errorDict["arr_0"]
    svmStdErrors[i] = errorDict["arr_1"]

    egoResultsFile = outputDir + "EgoResults_n=" + str(noise) + ".npz"
    errorDict = numpy.load(egoResultsFile)
    egoMeanErrors[i] = errorDict["arr_0"]
    egoStdErrors[i] = errorDict["arr_1"]

randErrors = numpy.zeros(len(ks))

for i in range(len(ks)):
    k = ks[i]
    randInds = numpy.random.permutation(500)[0:k]
    realInds = numpy.random.permutation(500)
    randErrors[i] = numpy.setdiff1d(randInds[0:k], realInds[0:k]).shape[0]/float(k)

print("SVM infuence errors:")
for i in range(len(ks)):
    print((Latex.array2DsToRows(svmMeanInfErrors[i, :], svmStdInfErrors[i, :]) + "\\"))

print("Ego-centric learner infuence errors:")
for i in range(len(ks)):
    print((Latex.array2DsToRows(egoMeanInfErrors[i, :], egoStdInfErrors[i, :]) + "\\"))

#Also print errors of SVM
print("\n")
print("SVM prediction errors")
print((Latex.array2DsToRows(svmMeanErrors, svmStdErrors) + "\\"))

print("\n")
print("Ego-centric prediction errors")
print((Latex.array2DsToRows(egoMeanErrors, egoStdErrors) + "\\"))

 
plt.figure(1)
plt.plot(ks, svmMeanInfErrors[:, 0],  'k.-')
plt.plot(ks, svmMeanInfErrors[:, 1],  'k.--')
plt.plot(ks, svmMeanInfErrors[:, 2],  'k.:')
#plt.plot(ks, svmMeanInfErrors[:, 3],  'k.-.')

plt.plot(ks, egoMeanInfErrors[:, 0],  'r.-')
plt.plot(ks, egoMeanInfErrors[:, 1],  'r.--')
plt.plot(ks, egoMeanInfErrors[:, 2],  'r.:')
#plt.plot(ks, egoMeanInfErrors[:, 3],  'r.-.')

#This plot is as bad as noise = 1 
#plt.plot(ks, randErrors,  'r.-.')
plt.legend( (r'SVM $\sigma=0.05$', r'SVM $\sigma=0.1$', r'SVM $\sigma=0.15$', r'EC $\sigma=0.05$', r'EC $\sigma=0.1$', r'EC $\sigma=0.15$'))
plt.xlabel("k")
plt.ylabel("Error")
plt.show()