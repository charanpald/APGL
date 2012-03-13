KernelCCA
=========

An implementation of the regularised Kernel Canonical Correlation Analysis (KCCA) algorithm. The following code generate two sets of examples sampled from the uniform distribution and then performs KCCA in the linear kernel space. The resulting dual directions for X and Y and given by alpha and beta respectively. See John Shawe-Taylor and Nello Cristianini, Kernel methods for pattern analysis, Cambridge University Press, 2004 for details. 

:: 

	import numpy 
	from apgl.features.KernelCCA import KernelCCA
	from apgl.kernel.LinearKernel import LinearKernel

        numExamples = 5
        numXFeatures = 10
        numYFeatures = 15
        X = numpy.random.rand(numExamples, numXFeatures)
        Y = numpy.random.rand(numExamples, numYFeatures)

        tau = 0.0
        kernel = LinearKernel()
        cca = KernelCCA(kernel, kernel, tau)
        alpha, beta, lmbdas = cca.learnModel(X, Y)

Methods 
-------
.. autoclass:: apgl.features.KernelCCA
   :members:
   :inherited-members:
   :undoc-members:
