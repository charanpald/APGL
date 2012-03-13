PrimalPCA
=========

An implementation of Principal Components Analysis (PCA) algorithm. The code snippet below shows an example usage of the class. The matrix U is contains the PCA projection directions as its columns and lmbdas is a vector of eigenvalues for the PCA problem. 

::
	
	import numpy 
	from apgl.features.PrimalPCA import PrimalPCA 
	
        numExamples = 10
        numFeatures = 5
        X = numpy.random.randn(numExamples, numFeatures)

        k = 4
        pca = PrimalPCA(k)
        U, lmbdas = pca.learnModel(X)

Methods 
-------
.. autoclass:: apgl.features.PrimalPCA
   :members:
   :inherited-members:
   :undoc-members:
