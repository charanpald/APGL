"""
We generate a set of toy datasets.
"""

import numpy
import logging
import sys 
import scipy.stats
import matplotlib.pyplot as plt
from apgl.util.PathDefaults import PathDefaults

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.random.seed(21)

numFeatures = 2
numCentres = 10
numPositives = 5

mean = numpy.zeros(numFeatures)
vars = numpy.ones(numFeatures)

centres = numpy.zeros((numCentres, numFeatures)) 
labels = numpy.ones(numCentres)
labels[numPositives:] = -1

for i in range(numCentres):
    for j in range(numFeatures):
        centres[i, j] = scipy.stats.norm.rvs(mean[j], vars[j])
        
print(centres)

numExamples = 5000
X = numpy.zeros((numExamples, numFeatures))
y = numpy.zeros(numExamples)

for i in range(numExamples):
    ind = numpy.random.randint(numCentres)
    centre = centres[ind, :]
    label = labels[ind]

    for j in range(numFeatures):
        X[i, j] = scipy.stats.norm.rvs(centre[j], vars[j]/5.0)
    y[i] = label

#Plot results
plt.figure(0)
plt.scatter(X[y==1, 0], X[y==1, 1], c='r' ,label="-1")
plt.scatter(X[y==-1, 0], X[y==-1, 1], c='b',label="+1")
plt.legend()

#We need the probability P(x) and p(y|x) 
print(numpy.min(X, 0))
print(numpy.max(X, 0))

numGridPoints = 200
gridPoints = numpy.linspace(-3, 3, numGridPoints)
pdfX = numpy.zeros((numGridPoints, numGridPoints))
pdfY1X = numpy.zeros((numGridPoints, numGridPoints))
pdfYminus1X = numpy.zeros((numGridPoints, numGridPoints))

for i in range(numGridPoints):
    print(i)
    for j in range(numGridPoints):
        pdfX[i, j] = 0
        pdfY1X[i, j] = 0
        pdfYminus1X[i, j] = 0
        for k in range(numCentres):
            pdfX[i, j] += scipy.stats.norm.pdf(gridPoints[i], centres[k, 0], vars[0]/5.0)*scipy.stats.norm.pdf(gridPoints[j], centres[k, 1], vars[1]/5.0)

            if labels[k] == 1:
                pdfY1X[i, j] += scipy.stats.norm.pdf(gridPoints[i], centres[k, 0], vars[0]/5.0)*scipy.stats.norm.pdf(gridPoints[j], centres[k, 1], vars[1]/5.0)
            else:
                pdfYminus1X[i, j] += scipy.stats.norm.pdf(gridPoints[i], centres[k, 0], vars[0]/5.0)*scipy.stats.norm.pdf(gridPoints[j], centres[k, 1], vars[1]/5.0)

        pdfX[i, j] /= numCentres
        pdfY1X[i, j] /= numCentres
        pdfYminus1X[i, j] /= numCentres

        pdfY1X[i, j] /= pdfX[i, j]
        pdfYminus1X[i, j] /= pdfX[i, j]

plt.figure(1)
plt.title('p(x)')
plt.contourf(gridPoints, gridPoints, pdfX.T, 100, antialiased=True)

plt.figure(2)
plt.title('p(y=1|x)')
plt.contourf(gridPoints, gridPoints, pdfY1X.T, 100, antialiased=True)

plt.figure(3)
plt.title('p(y=-1|x)')
plt.contourf(gridPoints, gridPoints, pdfYminus1X.T, 100, antialiased=True)

#Save the pdfs
dataDir = PathDefaults.getDataDir() + "modelPenalisation/toy/"
fileName = dataDir + "toyData.npz"

numpy.savez(fileName, gridPoints, X, y, pdfX, pdfY1X, pdfYminus1X)
logging.info('Saved results into file ' + fileName)

plt.show()
