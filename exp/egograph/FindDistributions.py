import logging
import random
import sys
import time
import numpy
import numpy.random
from apgl.egograph.SvmInfoExperiment import SvmInfoExperiment
from apgl.egograph.SvmEgoSimulator import SvmEgoSimulator
from apgl.io.EgoCsvReader import EgoCsvReader
from apgl.util.Latex import Latex
from apgl.generator import * 

def getVertexFeatureDistribution(index, vertexIndices):
    #In the 1st case we assume the vector of indices represents a binary variable
    if type(index) == numpy.ndarray:
        freqs = numpy.zeros(len(index))
        items = numpy.array(list(range(len(index))))+1
        for i in range(len(index)):
            (fs, its) = simulator.getVertexFeatureDistribution(index[i], vertexIndices)
            if numpy.nonzero(its == 1)[0].size != 0:
                freqs[i] = fs[numpy.nonzero(its == 1)]

        #items = numpy.r_[0, items]
        #freqs = numpy.r_[len(vertexIndices)-sum(freqs), freqs]
    else:
        (freqs, items) = simulator.getVertexFeatureDistribution(index, vertexIndices)

    return (freqs, items)

def getDistributions(index, numVertices, egos, alters, numItems):
    (freqs, items) = getVertexFeatureDistribution(index, list(range(0, numVertices)))
    histVector = numpy.zeros(numItems)
    histVector[items-1] = freqs/sum(freqs)

    distTable = numpy.array(list(range(0, numItems)))
    distTable = numpy.r_[numpy.array([distTable]), numpy.array([histVector])]

    (freqs, items) = getVertexFeatureDistribution(index, egos.tolist())
    histVector = numpy.zeros(numItems)
    histVector[items-1] = freqs/sum(freqs)
    distTable = numpy.r_[distTable, numpy.array([histVector])]

    for i in range(len(alters)):
        (freqs, items) = getVertexFeatureDistribution(index, alters[i].tolist())
        histVector = numpy.zeros(numItems)
        histVector[items-1] = freqs/sum(freqs)
        distTable = numpy.r_[distTable, numpy.array([histVector])]

    return distTable 

def printDistributions(distTable):
    print((" & " + Latex.array1DToRow(distTable[0, :]) + " &  \\\\"))
    print("\hline")
    print(("All & " + Latex.array1DToRow(distTable[1, :]) +  "\\\\"))
    print(("Egos & " + Latex.array1DToRow(distTable[2, :]) +  "\\\\"))

    for i in range(distTable.shape[0]-3):
        print(("Alters " + str(i+1) + " & " + Latex.array1DToRow(distTable[3+i, :])  + "\\\\"))

    print("\n")

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

random.seed(21)
numpy.random.seed(21)
startTime = time.time()

examplesFileName = SvmInfoExperiment.getExamplesFileName()
egoFileName = "../../data/EgoData.csv"
alterFileName = "../../data/AlterData.csv"
#This is double what we use for the diffusion results
numVertices = 20000
#numVertices = 5000

infoProb = 0.1
"""
p = 0.1
k = 15
generator = SmallWorldGenerator(p, k)
"""

#A second set of parameters 
p = float(30)/numVertices
generator = ErdosRenyiGenerator(p)

simulationRepetitions = 5 
maxIterations = 3
sampleSize = SvmInfoExperiment.getNumSimulationExamples()
#sampleSize = 1000
svmParamsFile = SvmInfoExperiment.getSvmParamsFileName()
CVal, kernel, kernelParamVal, errorCost = SvmInfoExperiment.loadSvmParams(svmParamsFile)

simulator = SvmEgoSimulator(examplesFileName)
simulator.trainClassifier(CVal, kernel, kernelParamVal, errorCost, sampleSize)

egoCsvReader = EgoCsvReader()

genderIndex = egoCsvReader.genderIndex
ageIndex = egoCsvReader.ageIndex
incomeIndex = egoCsvReader.incomeIndex
townSizeIndex = egoCsvReader.townSizeIndex
foodRiskIndex = egoCsvReader.foodRiskIndex
experienceIndex = egoCsvReader.experienceIndex
internetFreqIndex = egoCsvReader.internetFreqIndex
peopleAtWorkIndex = egoCsvReader.peopleAtWorkIndex
educationIndex = egoCsvReader.educationIndex

professionIndices = numpy.zeros(egoCsvReader.numProfessions)

for i in range(1, egoCsvReader.numProfessions+1):
    egoQuestionIds = egoCsvReader.getEgoQuestionIds()
    professionIndices[i-1] = egoQuestionIds.index(("Q7_" + str(i), 1))

#Number of each propery
numGenders = 2
numAges = 12
numEducations = 6
numProfessions = 8
numFoodRisks = 7
numExperiences = 2
numInternets = 4

"""
For professions there are 8 binary variables. In the survey data the user must pick
a value, however in our data generation it is possible some have all zero values.
We will ignore those in the distribution of professions
"""

#List of the numbers we want to store at each simulation
quantitiesArray = numpy.zeros((simulationRepetitions, maxIterations+1))
genderDistributions = numpy.zeros((simulationRepetitions, maxIterations+3, numGenders))
ageDistributions = numpy.zeros((simulationRepetitions, maxIterations+3, numAges))
educationDistributions = numpy.zeros((simulationRepetitions, maxIterations+3, numEducations))
professionDistributions = numpy.zeros((simulationRepetitions, maxIterations+3, numProfessions))
foodRiskDistributions = numpy.zeros((simulationRepetitions, maxIterations+3, numFoodRisks))
experienceDistributions = numpy.zeros((simulationRepetitions, maxIterations+3, numExperiences))
internetFreqDistributions = numpy.zeros((simulationRepetitions, maxIterations+3, numInternets))

for i in range(0, simulationRepetitions):
    vList = VertexList(numVertices, 0)
    graph = SparseGraph(vList)
    graph = generator.generate(graph)
    simulator.generateRandomGraph(egoFileName, alterFileName, infoProb, graph)
    (totalInfo, transmissions) = simulator.runSimulation(maxIterations)

    alters = []
    egos = numpy.unique(transmissions[0][:, 0])
    quantitiesArray[i, 0] = egos.shape[0]

    for j in range(len(transmissions)):
        alters.append(numpy.unique(transmissions[j][:, 1]))
        quantitiesArray[i, 1+j] = alters[j].shape[0]

    """
    The commented features have very large ranges - we need to bin them appropriately.
    Either that, or ignore them.
    """

    genderDistributions[i, :, :] = getDistributions(genderIndex, numVertices, egos, alters, numGenders)
    ageDistributions[i, :, :] = getDistributions(ageIndex, numVertices, egos, alters, numAges)
    educationDistributions[i, :, :] = getDistributions(educationIndex, numVertices, egos, alters, numEducations)
    professionDistributions[i, :, :] = getDistributions(professionIndices, numVertices, egos, alters, numProfessions)
    foodRiskDistributions[i, :, :] = getDistributions(foodRiskIndex, numVertices, egos, alters, numFoodRisks)
    experienceDistributions[i, :, :] = getDistributions(experienceIndex, numVertices, egos, alters, numExperiences)
    internetFreqDistributions[i, :, :] = getDistributions(internetFreqIndex, numVertices, egos, alters, numInternets)

print((numpy.mean(quantitiesArray, 0)))

printDistributions(numpy.mean(genderDistributions, 0))
printDistributions(numpy.mean(ageDistributions, 0))
printDistributions(numpy.mean(educationDistributions, 0))
printDistributions(numpy.mean(professionDistributions, 0))
printDistributions(numpy.mean(foodRiskDistributions, 0))
printDistributions(numpy.mean(experienceDistributions, 0))
printDistributions(numpy.mean(internetFreqDistributions, 0))


print("Standard deviations")
print((numpy.std(quantitiesArray, 0)))

printDistributions(numpy.std(genderDistributions, 0))
printDistributions(numpy.std(ageDistributions, 0))
printDistributions(numpy.std(educationDistributions, 0))
printDistributions(numpy.std(professionDistributions, 0))
printDistributions(numpy.std(foodRiskDistributions, 0))
printDistributions(numpy.std(experienceDistributions, 0))
printDistributions(numpy.std(internetFreqDistributions, 0))