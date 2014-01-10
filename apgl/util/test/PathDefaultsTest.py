
import unittest
from apgl.util.PathDefaults import PathDefaults 

class  PathDefaultsTestCase(unittest.TestCase):
    def testGetProjectDir(self):
        print((PathDefaults.getSourceDir()))
        
    def testGetDataDir(self):
        print((PathDefaults.getDataDir()))
        
    def testGetOutputDir(self):
        print((PathDefaults.getOutputDir()))


if __name__ == '__main__':
    unittest.main()

