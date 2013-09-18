

import numpy
import matplotlib 
matplotlib.use("GTK3Agg")
import matplotlib.pyplot as plt 

numpy.random.seed(21)

outputFilename = "/home/charanpal/Documents/Postdoc/Documents/Statistics/Lecture1/Figures/DiscreteHist.eps" 

X = numpy.random.randint(1, 7, 100)
plt.figure(0)
n, bins, patches = plt.hist(X, bins=6, rwidth=0.5)
plt.ylabel("Frequency")
plt.xlabel("x")
plt.savefig(outputFilename)

print(numpy.bincount(X))
print(n, bins, patches)




#Continuous histogram 
outputFilename = "/home/charanpal/Documents/Postdoc/Documents/Statistics/Lecture1/Figures/ContinuousHist.eps" 

X = numpy.random.randn(100)
plt.figure(2)
n, bins, patches = plt.hist(X, bins=10)
plt.ylabel("Frequency")
plt.xlabel("x")
plt.savefig(outputFilename)

print(n, bins, patches)

outputFilename = "/home/charanpal/Documents/Postdoc/Documents/Statistics/Lecture1/Figures/ContinuousCDF.eps" 
plt.figure(1)
plt.plot(bins[1:], numpy.cumsum(n)/numpy.sum(n))
plt.ylabel("F(x)")
plt.xlabel("x")
plt.savefig(outputFilename)

plt.show()