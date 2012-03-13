
from apgl.version import __version__

def checkImport(name):
    """
    A function to check if the given module exists by importing it.
    """
    try:
        import imp
        imp.find_module(name)
        return True
    except ImportError as error:
        return False 

def getPythonVersion():
    """
    Get the python version number as a floating point value.
    """
    import sys
    version = sys.version_info
    version = version[0] + version[1]/10.0 + version[2]/100.0
    return version

def skipIf(condition, reason):
    """
    A docorator for test skipping.
    """
    version = getPythonVersion()
    if version >= 2.7:
        import unittest
        return unittest.skipIf(condition, reason)
    else:
        import unittest2
        return unittest2.skipIf(condition, reason)

def test():
    """
    A function which uses the unittest library to find all tests within apgl (those files
    matching "*Test.py"), and run those tests. In python 2.7 and above the unittest framework
    is used otherwise one needs unittest2 for python 2.3-2.6.
    """
    try:
        import traceback
        import sys
        import os
        import unittest
        import logging
        from apgl.util.PathDefaults import PathDefaults

        logging.disable(logging.WARNING)
        dir = PathDefaults.getSourceDir()
        print("Running tests from " + dir)
        version = getPythonVersion()

        if version >= 2.7:
            testModules = unittest.defaultTestLoader.discover(dir, pattern='*Test.py')
        else:
            import unittest2
            testModules = unittest2.defaultTestLoader.discover(dir, pattern='*Test.py')
            
        overallTestSuite = unittest.TestSuite()

        for testSuite in testModules:
            overallTestSuite.addTests(testSuite)

        unittest.TextTestRunner(verbosity=1).run(overallTestSuite)

        logging.disable(logging.INFO)
    except ImportError as error:
        traceback.print_exc(file=sys.stdout)

