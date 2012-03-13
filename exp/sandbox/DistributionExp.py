"""
We want to better understand the parameters of the probability distrbutions.
"""
import scipy.stats as stats
import numpy
import matplotlib.pyplot as pyplot

sigma = 0.1
mode = 1
theta = (-mode + numpy.sqrt(mode**2+4*sigma**2))/2
k = mode/theta + 1

mean = 0.3
sigma = 0.2
a = -mean/sigma
b = (1-mean)/sigma
rv = stats.truncnorm(a, b, loc=mean, scale=sigma)
x = numpy.linspace(0.1, numpy.minimum(rv.dist.b, 3))
h = pyplot.plot(x, rv.pdf(x))

pyplot.show()