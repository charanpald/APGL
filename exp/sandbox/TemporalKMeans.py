
import numpy
from apgl.util.Parameter import Parameter 
"""
An implementation of the temporal k-means method.
"""

class TemporalKMeans(object):
    def __init__(self):
        pass

    def cluster(self, XList, k, tau):
        """
        Take a set of zero mean and unit variance examples in the rows of X (the
        entries of XList), and find clusters. Each matrix X must have the same
        number of rows, but can have differing numbers of columns. 
        """
        Parameter.checkInt(k, 1, float('inf'))
        Parameter.checkFloat(tau, 0.0, 1.0)

        n = XList[0].shape[0]
        m = len(XList)

        muList = []

        #Randomly assign initial means
        for i in range(m):
            numFeatures = XList[i].shape[1]
            mu = numpy.random.randn(k, numFeatures)
            muList.append(mu)

        #Each column represents class membership of all examples at a time point
        #Each row is the class membership of an example for all times 
        C = numpy.zeros((n, m), numpy.int)
        CLast = C+1

        while((C != CLast).any()):
            CLast = C

            #Need centered class membership 
            for i in range(m):
                for j in range(n):
                    dists = numpy.zeros(k)
                    for s in range(k):
                        dists[s] = (1-tau)*numpy.linalg.norm(XList[i][j, :] - muList[i][s, :])

                        tempCRow = C[j, :]
                        tempCRow[i] = s
                        dists[s] += tau*numpy.var(tempCRow)

                    #print(dists)
                    C[j, i] = numpy.argmin(dists)

            #Update means
            for i in range(m):
                for s in range(k):
                    muList[i][s, :] = numpy.mean(XList[i][C[:, i]==s, :], 0)

        return C, muList