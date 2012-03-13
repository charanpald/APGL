import logging
import unittest
import numpy
import scipy.sparse 
from apgl.util.ProfileUtils import ProfileUtils

class  ProfileUtilsTest(unittest.TestCase):
    def testMemDisplay(self):
        A = numpy.random.rand(10, 10)
        B = numpy.random.rand(100, 100)

        C = scipy.sparse.rand(1000, 1000, 0.5)

        #ProfileUtils.memDisplay(locals())

    def testMemory(self):
        logging.info(ProfileUtils.memory())

if __name__ == '__main__':
    unittest.main()

