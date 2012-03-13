import unittest
import numpy
import apgl


@apgl.skipIf(not apgl.checkImport('pysparse'), 'No module pysparse')
class  PySparseUtilsTest(unittest.TestCase):
    def testSum(self):
        try:
            from pysparse import spmatrix
            from apgl.util.PySparseUtils import PySparseUtils
        except ImportError as error:
            return

        n = 10
        X = spmatrix.ll_mat(n, n)

        self.assertEquals(PySparseUtils.sum(X), 0.0)

        X[1, 1] = 5
        X[2, 4] = 6.1
        X[3, 1] = 2.5

        self.assertEquals(PySparseUtils.sum(X), 13.6)

    def testNonzero(self):
        try:
            from pysparse import spmatrix
            from apgl.util.PySparseUtils import PySparseUtils
        except ImportError as error:
            return

        n = 10
        X = spmatrix.ll_mat(n, n)

        self.assertTrue((PySparseUtils.nonzero(X)[0]==numpy.array([], numpy.int)).all())
        self.assertTrue((PySparseUtils.nonzero(X)[0]==numpy.array([], numpy.int)).all())

        X[1, 1] = 5
        X[2, 4] = 6.1
        X[3, 1] = 2.5

        self.assertTrue((PySparseUtils.nonzero(X)[0]==numpy.array([1,2,3], numpy.int)).all())
        self.assertTrue((PySparseUtils.nonzero(X)[1]==numpy.array([1,4,1], numpy.int)).all())


if __name__ == '__main__':
    unittest.main()

    
