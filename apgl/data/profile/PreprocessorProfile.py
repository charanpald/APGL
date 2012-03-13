import numpy
from apgl.data.Preprocessor import Preprocessor
from apgl.data.ExamplesList import ExamplesList

numExamples = 200000
numFeatures = 500

preprocessor = Standardiser()

#Test an everyday matrix
X = numpy.random.rand(numExamples, numFeatures)
print("Created random matrix.")
Xn = preprocessor.normaliseArray(X)

print("All done!")