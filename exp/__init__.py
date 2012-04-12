def test():
    """
    A function which uses the unittest library to find all tests (those files
    matching "*Test.py"), and run those tests. 
    """
    try:
        import traceback
        import sys
        import os
        import unittest
        import logging

        logging.disable(logging.WARNING)
        logging.disable(logging.INFO)
        logging.disable(logging.DEBUG)
        
        sourceDir = os.path.abspath( __file__ )
        sourceDir, head = os.path.split(sourceDir)        
        
        print("Running tests from " + sourceDir)

        overallTestSuite = unittest.TestSuite()
        overallTestSuite.addTest(unittest.defaultTestLoader.discover(sourceDir, pattern='*Test.py'))
        unittest.TextTestRunner(verbosity=1).run(overallTestSuite)
    except ImportError as error:
        traceback.print_exc(file=sys.stdout)