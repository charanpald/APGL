
import logging
import unittest
from apgl.util.PathDefaults import PathDefaults 


class  PathDefaultsTestCase(unittest.TestCase):

    def testGetProjectDir(self):
        logging.debug((PathDefaults.getSourceDir()))

if __name__ == '__main__':
    unittest.main()

