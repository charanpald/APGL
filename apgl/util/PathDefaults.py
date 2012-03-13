import apgl
import os
import tempfile

#TODO: Read from a configuration file
class PathDefaults(object):
    """
    This class stores some useful global default paths. 
    """
    def __init__(self):
        pass

    @staticmethod
    def getProjectDir():
        dir =  apgl.__path__[0].split("/")

        try:
            ind = dir.index('APGL')+1

            projectDir = ""
            for i in range(0, ind):
                projectDir +=  dir[i] + "/"
        except ValueError:
            projectDir = os.path.abspath( __file__ )
            projectDir, head = os.path.split(projectDir)
            projectDir, head = os.path.split(projectDir)
            projectDir, head = os.path.split(projectDir)
            projectDir, head = os.path.split(projectDir)
            projectDir += "/"
        return projectDir 

    @staticmethod
    def getSourceDir():
        dir = os.path.abspath( __file__ )
        dir, head = os.path.split(dir)
        dir, head = os.path.split(dir)
        dir, head = os.path.split(dir)
        return dir 
        


    @staticmethod
    def getDataDir():
        return PathDefaults.getProjectDir() + "data/"

    @staticmethod
    def getTempDir():
        return tempfile.gettempdir() + "/"
        #return PathDefaults.getProjectDir() + "data/temp/"

    @staticmethod
    def getOutputDir():
        return PathDefaults.getProjectDir() + "output/"
