

"""
Some functions to make profiling code a bit simpler.
"""

import logging
import numpy
import scipy.sparse
from apgl.util.PathDefaults import PathDefaults
import os

class ProfileUtils(object):
    def __init__(self):
        pass

    @staticmethod
    def profile(command, globalVars, localVars, numStats=30):
        """
        Just profile the given command with the global and local variables
        and print out the cumulative and function times. 
        """
        try:
            import pstats
            import cProfile
        except ImportError:
            raise ImportError("profile() requires pstats and cProfile")

        outputDirectory = PathDefaults.getOutputDir()
        directory = outputDirectory + "test/"
        profileFileName = directory + "profile.cprof"

        logging.info("Starting to profile ...")
        cProfile.runctx(command, globalVars, localVars, profileFileName)
        logging.info("Done")
        stats = pstats.Stats(profileFileName)
        stats.strip_dirs().sort_stats("cumulative").print_stats(numStats)
        stats.strip_dirs().sort_stats("time").print_stats(numStats)

    @staticmethod
    def memDisplay(localDict):
        """
        Try to display the memory usage of numpy and scipy arrays. The input
        is the local namespace dict, found using memDisplay(locals())
        """

        #Store the array name and size in bytes
        arrayList = []

        for item in localDict.keys():
            if type(localDict[item])==numpy.ndarray:
                bytes = localDict[item].nbytes
                arrayList.append((item, localDict[item].shape, bytes))
            elif scipy.sparse.issparse(localDict[item]):
                bytes = localDict[item].getnnz()*localDict[item].dtype.itemsize
                arrayList.append((item, localDict[item].shape, bytes))

        #Now do the same for globals
        for item in globals().keys():
            if type(globals()[item])==numpy.ndarray:
                bytes = globals()[item].nbytes
                arrayList.append((item, globals()[item].shape, bytes))
            elif scipy.sparse.issparse(globals()[item]):
                bytes = globals()[item].getnnz()*globals()[item].dtype.itemsize
                arrayList.append((item, globals()[item].shape, bytes))
            
        #Now sort list
        arrayList.sort(key= lambda s: s[2])
        arrayList.reverse()

        #Now print results
        logging.debug("---------------------------------------")
        for item in arrayList:
            logging.debug(str(item[0]) +  ": " + str(item[1]) + " - " + str(float(item[2])/10**6) + " MB")
        logging.debug("---------------------------------------")

    #The following code is from the Python cookbook http://code.activestate.com/recipes/286222/:
    @staticmethod
    def _VmB(VmKey):
        '''Private.
        '''
        _proc_status = '/proc/%d/status' % os.getpid()

        _scale = {'kB': 1024.0, 'mB': 1024.0*1024.0,
                  'KB': 1024.0, 'MB': 1024.0*1024.0}
         # get pseudo file  /proc/<pid>/status
        try:
            t = open(_proc_status)
            v = t.read()
            t.close()
        except:
            return 0.0  # non-Linux?
         # get VmKey line e.g. 'VmRSS:  9999  kB\n ...'
        i = v.index(VmKey)
        v = v[i:].split(None, 3)  # whitespace
        if len(v) < 3:
            return 0.0  # invalid format?
         # convert Vm value to bytes
        return float(v[1]) * _scale[v[2]]

    @staticmethod
    def memory(since=0.0):
        '''Return memory usage in bytes.
        '''
        return ProfileUtils._VmB('VmSize:') - since


    @staticmethod
    def resident(since=0.0):
        '''Return resident memory usage in bytes.
        '''
        return ProfileUtils._VmB('VmRSS:') - since

    @staticmethod
    def stacksize(since=0.0):
        '''Return stack size in bytes.
        '''
        return ProfileUtils._VmB('VmStk:') - since