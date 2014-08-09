import os
import tempfile 
try: 
    from ConfigParser import SafeConfigParser
except ImportError: 
    from configparser import SafeConfigParser  
from os.path import expanduser

class PathDefaults(object):
    """
    This class stores some useful global default paths. 
    """
    @staticmethod 
    def readField(field):
        configFileName = expanduser("~") + os.sep + ".apglrc"

        if not os.path.exists(configFileName): 
            print("Creating missing config file: " + configFileName)
            defaultConfig = "[paths]\n" 
            defaultConfig += "data = " + expanduser("~") + os.sep + "data" + os.sep + "\n"
            defaultConfig += "output = " + expanduser("~") + os.sep + "output" + os.sep + "\n"
            configFile = open(configFileName, "w")
            configFile.write(defaultConfig)
            configFile.close()
            
        parser = SafeConfigParser()
        parser.read(configFileName)
        return parser.get('paths', field)

    @staticmethod
    def getSourceDir():
        """
        Root directory of source code for APGL. 
        """
        dir = os.path.abspath( __file__ )
        dir, head = os.path.split(dir)
        dir, head = os.path.split(dir)
        return dir 
        
    @staticmethod
    def getDataDir():
        """
        Location of data files. 
        """
        return PathDefaults.readField("data")

    @staticmethod
    def getOutputDir():
        """
        Location of output files. 
        """
        return PathDefaults.readField("output")
        
    @staticmethod
    def getTempDir():
        return tempfile.gettempdir() + os.sep
