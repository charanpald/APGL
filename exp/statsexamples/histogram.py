

import numpy
import matplotlib 
matplotlib.use("GTK3Agg")
import matplotlib.pyplot as plt 

numpy.random.seed(21)
X = numpy.random.randint(1, 7, 100)

print(X)

n, bins, patches = plt.hist(X, bins=6, rwidth=0.5)

print(numpy.bincount(X))
print(n, bins, patches)

outputFilename = "/home/charanpal/Documents/Postdoc/Documents/Statistics/Lecture1/Figures/DiscreteHist.eps" 

plt.ylabel("Frequency")
plt.xlabel("x")
plt.savefig(outputFilename)
plt.show()

