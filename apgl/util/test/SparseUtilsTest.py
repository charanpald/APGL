import unittest
import numpy
import apgl
import scipy.sparse
from apgl.util.SparseUtils import SparseUtils

class  SparseUtilsTest(unittest.TestCase):
    def testEquals(self):
        A = numpy.array([[4, 2, 1], [6, 3, 9], [3, 6, 0]])
        B = numpy.array([[4, 2, 1], [6, 3, 9], [3, 6, 0]])

        A = scipy.sparse.csr_matrix(A)
        B = scipy.sparse.csr_matrix(B)

        self.assertTrue(SparseUtils.equals(A, B))

        A[0, 1] = 5
        self.assertFalse(SparseUtils.equals(A, B))

        A[0, 1] = 2
        B[0, 1] = 5
        self.assertFalse(SparseUtils.equals(A, B))

        A[2, 2] = -1
        self.assertFalse(SparseUtils.equals(A, B))

        #Test two empty graphs
        A = scipy.sparse.csr_matrix((5, 5)) 
        B = scipy.sparse.csr_matrix((5, 5))

        self.assertTrue(SparseUtils.equals(A, B))

    def testNorm(self):
        numRows = 10
        numCols = 10

        for k in range(10):
            A = scipy.sparse.rand(numRows, numCols, 0.1, "csr")

            norm = SparseUtils.norm(A)

            norm2 = 0
            for i in range(numRows):
                for j in range(numCols):
                    norm2 += A[i, j]**2

            norm2 = numpy.sqrt(norm2)
            norm3 = numpy.linalg.norm(numpy.array(A.todense()))
            self.assertAlmostEquals(norm, norm2)
            self.assertAlmostEquals(norm, norm3)


if __name__ == '__main__':
    unittest.main()


