
from apgl.graph import *
from apgl.util import *
import numpy


class SpectralEdgePredictor(object):
    def __init__(self, degree):
        self.degree = degree


    def predictEdges(self, graphList1, graphList2):
        """
        Make a prediction over a set of graphs. All graphs must be same size.
        Try just for the path counting kernel. 
        """

        

    def predictEdge(self, graph1, graph2):
        """
        We need this function to predict over a set of pairs of graphs.
        """ 

        numVertices = graph1.getNumVertices()

        A = graph1.getWeightMatrix()
        L = graph1.laplacianMatrix()
        B = graph2.getWeightMatrix()

        (x1, U1) = numpy.linalg.eig(A)
        (x2, U2) = numpy.linalg.eig(L)
        (s1, _) = numpy.linalg.eig(B)

        print(x1)

        Y1 = Util.mdot(U1.T, B, U1)
        y1 = numpy.diag(Y1)

        Y2 = Util.mdot(U2.T, B, U2)
        y2 = numpy.diag(Y2)

        numMethods = 4
        f = numpy.zeros((numVertices, numMethods))
        phi = numpy.zeros(numMethods)

        #Now we find the function f of eigenvalues x which is closest to y
        #For the moment we leave out the exponential ones since they give NaNs
        f[:, 0] = self.pathFunction(x1, y1, self.degree)
        #f[:, 1] = self.expFunction(x1, y1)
        f[:, 1] = self.neumannFunction(x1, y1)
        f[:, 2] = self.combFunction(x2)
        f[:, 3] = self.neumannFunction(x2, y2)
        #f[:, 5] = self.expFunction(x2, y2) #Check this

        for i in range(2):
            phi[i] = numpy.linalg.norm(f[:, i] - y1)

        for i in range(2, numMethods):
            phi[i] = numpy.linalg.norm(f[:, i] - y2)

        print(phi)

        return numpy.argmin(phi)
    


    def pathFunction(self, x, y, d):
        """
        Compute the path function kernel using degree d and given alpha values.
        Alpha is  be a vector of size d and computed automatically. 
        """

        numVertices = x.shape[0]
        f = numpy.zeros(numVertices)
        Q = numpy.zeros((numVertices, d+1))

        for i in range(0, d+1):
            Q[:, i] = x**i

        alpha = Util.mdot(numpy.linalg.inv(numpy.dot(Q.T, Q)), Q.T, y)

        for i in range(0, d+1):
            f = f + alpha[i]*x**i

        return f

    def expFunction(self, x, y):
        """
        Work out the exponential function kernel for a given scalar alpha
        """

        alpha = numpy.sum(numpy.log(y*x) - numpy.log(x))/numpy.sum(x)

        f = numpy.exp(alpha*x)
        return f

    def neumannFunction(self, x, y):
        """
        Compute the Neumann kernel for a vector.
        """
        alpha = -(numpy.sum(x) - numpy.dot(x.T, y))/numpy.dot(x**2, y)

        f = 1/(1- alpha * x)

        return f


    def combFunction(self, x):
        """
        Compute the combination kernel on the Laplacian matrix
        """

        x = (x + numpy.abs(x))/2
        x = x + x==0
        f = 1/x

        return f

    