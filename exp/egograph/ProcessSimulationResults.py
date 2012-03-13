
import scipy.io
import numpy 
from apgl.egograph.InfoExperiment import InfoExperiment
from apgl.util.Latex import Latex
import matplotlib.pyplot as pyplot

outputDir = SvmInfoExperiment.getOutputDirectoryName()

graphType = "SmallWorld"
ps = [0.01, 0.02, 0.05, 0.1]
ks = [10, 15]
infoProbs = [0.1, 0.2, 0.5]

numpy.set_printoptions(2)

maxIters = 4

for p in ps:
    for k in ks:
        for infoProb in infoProbs:
            outputFileName = SvmInfoExperiment.getOutputFileName(graphType, p, k, infoProb)

            try:
                outputDict = scipy.io.loadmat(outputFileName + ".mat")
                averageHops = outputDict["averageHops"]
                totalInfo = outputDict["totalInfo"]
                numVertices = outputDict["numVertices"]

                if "receiversToSenders" in outputDict:
                    receiversToSenders = outputDict["receiversToSenders"]
                    meanReceiversToSenders = numpy.mean(receiversToSenders)
                    stdReceiversToSenders = numpy.std(receiversToSenders)
                else:
                    meanReceiversToSenders = 0
                    stdReceiversToSenders = 0

                meanTotalInfo = numpy.mean(totalInfo, 0)/numVertices
                stdTotalInfo = numpy.std(totalInfo, 0)/numVertices
                meanAverageHops = numpy.mean(averageHops)
                stdAverageHops = numpy.std(averageHops)

                meanTotalInfo = meanTotalInfo.ravel()
                stdTotalInfo = stdTotalInfo.ravel()
                meanTotalInfo = meanTotalInfo[1:maxIters]-meanTotalInfo[0:maxIters-1]
                stdTotalInfo = stdTotalInfo[0:maxIters-1]+stdTotalInfo[1:maxIters]
                if infoProb == 0.5:
                    pyplot.plot(meanTotalInfo.ravel(), label= "p="+str(p)+",k="+str(k))

                print((str(p) + " & " + str(k) + " & " + str(infoProb) + " & %.3f" % meanAverageHops), end=' ')
                print(("(%.3f) & " % stdAverageHops), end=' ')
                print(("%.3f (%.3f) & " % (meanReceiversToSenders, stdReceiversToSenders)), end=' ')
                print((Latex.array2DsToRows(meanTotalInfo.ravel()[0:maxIters], stdTotalInfo.ravel()[0:maxIters]) + "\\\\"))
            except IOError:
                print(("File not found : " + outputFileName))

pyplot.xlabel('Total information')
pyplot.ylabel('Iteration')
pyplot.legend(loc=10, ncol=3, shadow=True)

#pyplot.show()

print("")
print("")

graphType = "ErdosRenyi"
ps = [0.001, 0.002, 0.003, 0.004]

for p in ps:
    for infoProb in infoProbs:

        outputFileName = SvmInfoExperiment.getOutputFileName(graphType, p, k, infoProb)

        try:
            outputDict = scipy.io.loadmat(outputFileName + ".mat")
            averageHops = outputDict["averageHops"]
            totalInfo = outputDict["totalInfo"]
            numVertices = outputDict["numVertices"]

            if "receiversToSenders" in outputDict:
                receiversToSenders = outputDict["receiversToSenders"]
                meanReceiversToSenders = numpy.mean(receiversToSenders)
                stdReceiversToSenders = numpy.std(receiversToSenders)
            else:
                meanReceiversToSenders = 0
                stdReceiversToSenders = 0

            meanTotalInfo = numpy.mean(totalInfo, 0)/numVertices
            stdTotalInfo = numpy.std(totalInfo, 0)/numVertices
            meanAverageHops = numpy.mean(averageHops)
            stdAverageHops = numpy.std(averageHops)

            meanTotalInfo = meanTotalInfo[1:maxIters]-meanTotalInfo[0:maxIters-1]
            stdTotalInfo = stdTotalInfo[0:maxIters-1]+stdTotalInfo[1:maxIters]

            print((str(p) + " & "  + str(infoProb) + " & %.3f" % meanAverageHops), end=' ')
            print(("(%.3f) & " % stdAverageHops), end=' ')
            print(("%.3f (%.3f) & " % (meanReceiversToSenders, stdReceiversToSenders)), end=' ')
            print((Latex.array2DsToRows(meanTotalInfo.ravel()[0:maxIters], stdTotalInfo.ravel()[0:maxIters]) + "\\\\"))
        except IOError:
            print(("File not found : " + outputFileName))