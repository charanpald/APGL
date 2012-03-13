import numpy
import matplotlib
import matplotlib.pyplot as plt
from apgl.util import *

outputDirectory = PathDefaults.getOutputDir()
outputDir = outputDirectory + "influence/"
influenceArraySW1 = numpy.load(outputDir + "InfluenceArraySW1.npy")
influenceArrayER = numpy.load(outputDir + "InfluenceArrayER.npy")

print((influenceArraySW1.shape))
print((influenceArrayER.shape))

matplotlib.rcParams['ps.useafm'] = True

iks = list(range(10, 510, 10))
iks.insert(0, 1)
#iks = range(50, 550, 50)


for i in range(0, influenceArraySW1.shape[0]):
    print((Latex.array1DToRow(influenceArraySW1[i, :]) + "\\\\"))
print("\n\n")

plt.figure(1)
plt.plot(iks, influenceArraySW1[:, 0],  'k.-', iks, influenceArraySW1[:, 1], 'k.--', iks, influenceArraySW1[:, 2], 'k.:')
plt.axis([0, 500, 0, 1000])
plt.legend( ('General Inf.', 'Standard Inf.', 'Random'), loc=4)
plt.xlabel("k")
plt.ylabel("Total Activation")

plt.figure(2)
plt.plot(iks, influenceArrayER[:, 0],  'k.-', iks, influenceArrayER[:, 1], 'k.--', iks, influenceArrayER[:, 2], 'k.:')
plt.axis([0, 500, 0, 1000])
plt.legend( ('General Inf.', 'Standard Inf.', 'Random'), loc=4)
plt.xlabel("k")
plt.ylabel("Total Activation")

#Now finally do it for the dense graph
influenceArraySW2 = numpy.load(outputDir + "influenceArraySW2.npy")

print("d=rand")
for i in range(0, influenceArraySW2.shape[0]):
    print((Latex.array1DToRow(influenceArraySW2[i, :]) + "\\\\"))
print("\n\n")

plt.figure(3)
plt.plot(iks, influenceArraySW2[:, 0],  'k.-', iks, influenceArraySW2[:, 1], 'k.--', iks, influenceArraySW2[:, 2], 'k.:')
plt.axis([0, 500, 0, 1000])
plt.legend( ('General Inf.', 'Standard Inf.', 'Random'), loc=4)
plt.xlabel("k")
plt.ylabel("Total Activation")
plt.show()

