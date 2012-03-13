from cvxopt import matrix, solvers
from apgl.data.Standardiser import Standardiser
import numpy 
"""
Let's test the massively complicated bound on the clustering error
"""
numpy.set_printoptions(suppress=True, linewidth=150)


numC1Examples = 50
numC2Examples = 50 
d = 3
numpy.random.seed(21)

center1 = numpy.array([-1, -1, -1])
center2 = numpy.array([1, 1, 1])


V1 = numpy.random.randn(numC1Examples, d)+center1
V2 = numpy.random.randn(numC2Examples, d)+center2
V = numpy.r_[V1, V2]

#Normalise V
V = Standardiser().normaliseArray(V.T).T
V1 = V[0:numC1Examples, :]
V2 = V[numC1Examples:, :]

delta = 0.5

q = delta/2 - numC1Examples - numC1Examples
muC1 = numpy.mean(V1, 0)
muC2 = numpy.mean(V2, 0)

zero1 = numpy.zeros(d)
zero2 = numpy.zeros((d, d))
zero3 = numpy.zeros((2*d+2, 2*d+2))
zero4 = numpy.zeros(d*2+2)
ones1 = numpy.ones(d)

f = numpy.r_[zero1, zero1, -1, -1]
g = numpy.r_[muC1*numC1Examples, muC2*numC1Examples, 0, 0]
h = numpy.r_[zero1, zero1, -1/numC1Examples, -1/numC2Examples]
Q1 = numpy.diag(numpy.r_[ones1, zero1, 0, 0])
Q2 = numpy.diag(numpy.r_[zero1, ones1, 0, 0])

P1 = numpy.c_[zero2, zero2, muC1, -muC2]
P2 = numpy.c_[zero2, zero2, -muC1, muC2]
P3 = numpy.c_[numpy.array([muC1]), -numpy.array([muC1]), 0, 0]
P4 = numpy.c_[-numpy.array([muC2]), numpy.array([muC2]), 0, 0]

P = numpy.r_[P1, P2, P3, P4]

R1 = numpy.c_[0, 0.5 * numpy.array([f])]
R2 = numpy.c_[0.5 * numpy.array([f]).T, zero3]
R = numpy.r_[R1, R2]

S1 = numpy.r_[numpy.c_[-q, -0.5 *numpy.array([g])], numpy.c_[-0.5*numpy.array([g]).T, P]]
S2 = numpy.r_[numpy.c_[-1, numpy.array([zero4])], numpy.c_[numpy.array([zero4]).T, Q1]]
S3 = numpy.r_[numpy.c_[-1, numpy.array([zero4])], numpy.c_[numpy.array([zero4]).T, Q2]]
S4 = numpy.r_[numpy.c_[-1, -0.5 * numpy.array([h])], -0.5 * numpy.c_[numpy.array([h]).T, zero3]]


print(S1)

cvxc = matrix(R.flatten())
cvxG = [matrix(S1.flatten()).T]
#cvxG += [matrix(S2.flatten()).T]
#cvxG += [matrix(S3.flatten()).T]
#cvxG += [matrix(S4.flatten()).T]

cvxh = [matrix([0.0])]
#cvxh += [matrix([0.0])]
#cvxh += [matrix([0.0])]
#cvxh += [matrix([0.0])]


sol = solvers.sdp(cvxc, Gs=cvxG, hs=cvxh)