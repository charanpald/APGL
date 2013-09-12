
import numpy 
from exp.influence2.ArnetMinerDataset import ArnetMinerDataset
from exp.influence2.GraphRanker import GraphRanker
from exp.influence2.RankAggregator import RankAggregator
from apgl.util.Latex import Latex 
from apgl.util.Util import Util 
from apgl.util.Evaluator import Evaluator 

ranLSI = True
numpy.set_printoptions(suppress=True, precision=3, linewidth=100)
dataset = ArnetMinerDataset(runLSI=ranLSI)
#dataset.fields = ["Intelligent Agents"]

ns = numpy.arange(5, 55, 5)
bestaverageTestPrecisions = numpy.zeros(len(dataset.fields))

computeInfluence = True
graphRanker = GraphRanker(k=100, numRuns=100, computeInfluence=computeInfluence, p=0.05, inputRanking=[1, 2])
methodNames = graphRanker.getNames()
methodNames.append("MC2")

numMethods = len(methodNames) 
averageTrainPrecisions = numpy.zeros((len(dataset.fields), len(ns), numMethods))
averageTestPrecisions = numpy.zeros((len(dataset.fields), len(ns), numMethods))

coverages = numpy.load(dataset.coverageFilename)
print("==== Coverages ====")
print(coverages)

for s, field in enumerate(dataset.fields): 
    if ranLSI: 
        outputFilename = dataset.getOutputFieldDir(field) + "outputListsLSI.npz"
    else: 
        outputFilename = dataset.getOutputFieldDir(field) + "outputListsLDA.npz"
        
    try: 
        print(field)  
        outputLists, trainExpertMatchesInds, testExpertMatchesInds = Util.loadPickle(outputFilename)
        graph, authorIndexer = Util.loadPickle(dataset.getCoauthorsFilename(field))
        
        trainPrecisions = numpy.zeros((len(ns), numMethods))
        testPrecisions = numpy.zeros((len(ns), numMethods))
        
        #Remove training experts from the output lists 
        testOutputLists = [] 
        for outputList in outputLists:
            newOutputList = []
            for item in outputList: 
                if item not in trainExpertMatchesInds: 
                    newOutputList.append(item)
            testOutputLists.append(newOutputList)
        
        for i, n in enumerate(ns):     
            for j, outputList in enumerate(outputLists): 
                testOutputList = testOutputLists[j]                
                
                trainPrecisions[i, j] = Evaluator.precisionFromIndLists(trainExpertMatchesInds, outputList[0:n]) 
                testPrecisions[i, j] = Evaluator.precisionFromIndLists(testExpertMatchesInds, testOutputList[0:n]) 
                averageTrainPrecisions[s, i, j] = Evaluator.averagePrecisionFromLists(trainExpertMatchesInds, outputList[0:n], n)
                averageTestPrecisions[s, i, j] = Evaluator.averagePrecisionFromLists(testExpertMatchesInds, testOutputList[0:n], n) 

        #Now look at rank aggregations
        relevantItems = set([])
        for outputList in testOutputLists: 
            relevantItems = relevantItems.union(outputList)
        relevantItems = list(relevantItems)

        newOutputList = []
        newOutputList.append(testOutputLists[0])
        newOutputList.append(testOutputLists[1])
        newOutputList.append(testOutputLists[2])
        #newOutputList.append(testOutputLists[3])
        #newOutputList.append(testOutputLists[4])
        #newOutputList.append(testOutputLists[5])
        #newOutputList.append(testOutputLists[6])

        rankAggregate = RankAggregator.MC2(newOutputList, relevantItems)[0]
        j = len(outputLists)
        
        for i, n in enumerate(ns):
            testPrecisions[i, j] = Evaluator.precisionFromIndLists(testExpertMatchesInds, rankAggregate) 
            averageTestPrecisions[s, i, j] = Evaluator.averagePrecisionFromLists(testExpertMatchesInds, rankAggregate, n) 
        
        print(authorIndexer.reverseTranslate(trainExpertMatchesInds))
        print(authorIndexer.reverseTranslate(testExpertMatchesInds))
        print(authorIndexer.reverseTranslate(testOutputLists[0][0:10]))
        
        print(trainPrecisions)
        print(averageTrainPrecisions[s, :, :])
        print(testPrecisions)
        print(averageTestPrecisions[s, :, :])
    except IOError as e: 
        print(e)

meanAverageTrainPrecisions = numpy.mean(averageTrainPrecisions, 0)
meanAverageTrainPrecisions = numpy.c_[numpy.array(ns), meanAverageTrainPrecisions]

meanAverageTestPrecisions = numpy.mean(averageTestPrecisions, 0)
meanAverageTestPrecisions = numpy.c_[numpy.array(ns), meanAverageTestPrecisions]

print("==== Summary ====")
print(Latex.listToRow(methodNames))
print(Latex.array2DToRows(meanAverageTrainPrecisions))

print(Latex.listToRow(methodNames))
print(Latex.array2DToRows(meanAverageTestPrecisions))

