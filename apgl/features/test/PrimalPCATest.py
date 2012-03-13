# To change this template, choose Tools | Templates
# and open the template in the editor.

import unittest
import numpy
import scipy.linalg
from apgl.features.PrimalPCA import PrimalPCA 


class  PrimalPCATest(unittest.TestCase):
    def setUp(self):
        self.numExamples = 10
        self.numFeatures = 5

        self.X = numpy.random.randn(self.numExamples, self.numFeatures)

    def testLearnModel(self):
        k = 4
        tol = 10**-6 
        pca = PrimalPCA(k)
        U, lmbdas = pca.learnModel(self.X)

        #Compute eignvalues manually
        C = numpy.dot(self.X.T, self.X)
        lmbdas2, U2 = scipy.linalg.eig(C)
        inds = numpy.flipud(numpy.argsort(lmbdas2))

        lmbdas2 = lmbdas2[inds]
        U2 = U2[:, inds]

        self.assertTrue( (lmbdas >= 0).all())
        self.assertEquals(lmbdas.shape[0], self.numFeatures )
        self.assertEquals(U.shape[1], self.numFeatures )

        self.assertTrue( numpy.linalg.norm(lmbdas2- lmbdas) <= tol )
        self.assertTrue( numpy.linalg.norm(U2- U) <= tol )

        newX = pca.project(self.X)
        newX2 = numpy.dot(self.X, U2[:, 0:k])

        self.assertTrue( numpy.linalg.norm(newX2 - newX) <= tol )

        #Test orthogonality of U
        self.assertTrue( numpy.linalg.norm(numpy.dot(U.T, U) - numpy.eye(self.numFeatures)) <= tol )

        #Test projecting just k directions
        k = 2
        pca.setK(k)
        newX = pca.project(self.X)
        self.assertTrue( numpy.linalg.norm(newX2[:, 0:k] - newX) <= tol)

    def testProject(self):
        k = 2
        pca = PrimalPCA(k)
        U, lmbdas = pca.learnModel(self.X)
      
        newX = pca.project(self.X)
        self.assertEquals(newX.shape[0], self.X.shape[0])
        self.assertEquals(newX.shape[1], k)

if __name__ == '__main__':
    unittest.main()

