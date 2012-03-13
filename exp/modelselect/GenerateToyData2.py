"""
We generate a toy regression dataset 
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
    y[i] =  scipy.stats.norm.pdf(X[i, 0], centre[0], vars[0]/5.0)*scipy.stats.norm.pdf(X[i, 1], centre[1], vars[1]/5.0)

#Plot results
plt.figure(0)
plt.scatter(X[:, 0], X[:, 1], s=y*5)

#We need the probability P(x) and p(y|x) 
print(numpy.min(X, 0))
print(numpy.max(X, 0))

print(numpy.min(y))
print(numpy.max(y))

numGridPoints = 200
gridPoints = numpy.linspace(-3, 3, numGridPoints)
pdfX = numpy.zeros((numGridPoints, numGridPoints))
pdfYX = numpy.zeros((numGridPoints, numGridPoints))

for i in range(numGridPoints):
    print(i)
    for j in range(numGridPoints):
        pdfX[i, j] = 0
        pdfYX[i, j] = 0

        for k in range(numCentres):
            pdfX[i, j] += scipy.stats.norm.pdf(gridPoints[i], centres[k, 0], vars[0]/5.0)*scipy.stats.norm.pdf(gridPoints[j], centres[k, 1], vars[1]/5.0)
            pdfYX[i, j] += scipy.stats.norm.pdf(gridPoints[i], centres[k, 0], vars[0]/5.0)*scipy.stats.norm.pdf(gridPoints[j], centres[k, 1], vars[1]/5.0)


        pdfX[i, j] /= numCentres
        pdfYX[i, j] /= numCentres
        pdfYX[i, j] /= pdfX[i, j]

plt.figure(1)
plt.title('p(x)')
plt.contourf(gridPoints, gridPoints, pdfX.T, 100, antialiased=True)

plt.figure(2)
plt.title('p(y|x)')
plt.contourf(gridPoints, gridPoints, pdfYX.T, 100, antialiased=True)

#Save the pdfs
dataDir = PathDefaults.getDataDir() + "modelPenalisation/toy/"
fileName = dataDir + "toyDataReg.npz"

numpy.savez(fileName, gridPoints, X, y, pdfX, pdfYX)
logging.info('Saved results into file ' + fileName)

plt.show()
