"""
This is a linear programming solution to the k-means problem.
"""
import numpy
from cvxopt import solvers
from cvxopt import matrix
from cvxopt import spmatrix


class LPKMeans(object):
    def __init__(self):
        pass

    def cluster(self, X, k):
        """
        Cluster a set of examples X in k clusters. This algorithm is very slow
        because there are ell^2 number of variables in the optimisation. 
        """

        #First we have to formulate the LP problem in standard form
        ell = X.shape[0]
        numVars = ell**2+ell
        print(("numVars =" + str(numVars)))

        K = numpy.dot(X, X.T)
        Kd = numpy.diag(K)
        
        j = numpy.ones(ell)
        j2 = numpy.ones(numVars)
        
        D = numpy.outer(Kd, j) + numpy.outer(j, Kd) - 2*K

        d = D.ravel()
        c = numpy.r_[d, numpy.zeros(ell)]

        I = spmatrix(1.0, list(range(ell)), list(range(ell)))
        I2 = spmatrix(1.0, list(range(ell**2+ell)), list(range(numVars)))
        Q = spmatrix(0, [], [], size=(ell**2, numVars))

        for i in range(ell):
            Q[i*ell:(i+1)*ell,i*ell:(i+1)*ell] = I
            Q[i*ell:(i+1)*ell, ell**2:] = -I

        r = spmatrix(0, [], [], size=(ell**2+ell, 1))
        r[ell**2:] = 1

        G = spmatrix(0, [], [], size=(3*ell**2 + 2*ell + 1, numVars))
        G[0:ell**2,:] = Q
        G[ell**2:2*(ell**2)+ell,:] = I2
        G[2*(ell**2)+ell:3*(ell**2)+2*ell:,:] = -I2
        G[3*(ell**2)+2*ell, :] = r.T        

        h = numpy.zeros(ell**2)
        h = numpy.r_[h, j2]
        h = numpy.r_[h, numpy.zeros(ell**2+ell)]
        h = numpy.append(h, [k])

        #Equality constraints
        A = spmatrix(0, [], [], size=(ell, numVars))
        for i in range(ell):
            A[i, i*ell:(i+1)*ell] = 1

        b = numpy.ones(ell)
        c = matrix(c)
        h = matrix(h)
        b = matrix(b)
        sol = solvers.lp(c, G, h, A, b, solver="glpk")

        return sol 

