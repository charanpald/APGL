


import numpy
import unittest
import apgl
from apgl.util.FileLock import FileLock
from apgl.util.PathDefaults import PathDefaults


class  FileLockTest(unittest.TestCase):
    def setUp(self):
        tempDir = PathDefaults.getTempDir()
        self.fileName = tempDir + "abc"

    def testInit(self):
        fileLock = FileLock(self.fileName)

    def testLock(self):
        fileLock = FileLock(self.fileName)
        fileLock.lock()

    def testUnlock(self):
        fileLock = FileLock(self.fileName)
        fileLock.lock()

        self.assertTrue(fileLock.isLocked())
        fileLock.unlock()
        self.assertTrue(not fileLock.isLocked())