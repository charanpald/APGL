"""
We extend the pysparse lil_matrix with some useful methods. 
"""

import numpy

class PySparseUtils(object):
    @staticmethod 
    def sum(M):
        """
        Sum all of the elements of the matrix M.
        """
        (rows, cols) = PySparseUtils.nonzero(M)
        elements = numpy.zeros(len(rows))
        M.take(elements, rows, cols)

        return numpy.sum(elements)

    @staticmethod
    def nonzero(M):
        """
        Compute the nonzero entries of the matrix M, and return a tuple of two
        arrays - the first containing row indices and the second containing
        columns ones. 
        """
        rows = numpy.zeros(M.nnz, numpy.int)
        cols = numpy.zeros(M.nnz, numpy.int)

        kys = list(M.keys())

        for i in range(M.nnz):
            rows[i] = kys[i][0]
            cols[i] = kys[i][1]

        return rows, cols
    