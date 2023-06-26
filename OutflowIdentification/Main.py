# -*- coding: iso-8859-1 -*-
#@author: Phillip Kahl

## Import existing packages
from datetime import datetime                                                               ## import datetime package
from  distutils.util import strtobool                                                       ## import distutils package
import multiprocessing as mp                                                                ## import multiprocessing package
import numpy as np                                                                          ## import numpy package
import os                                                                                   ## import os package
from pathlib import Path                                                                    ## import pathlib package
import platform                                                                             ## import platform package
import shutil                                                                               ## import shutil package
import sys                                                                                  ## import sys package
import time                                                                                 ## import time package
import xml.etree.ElementTree as ET                                                          ## import xml package

## Import self written modules
from ConcentrateFV import ConcentrateTrainFV as ctfv
from Create_Feature_Vector import create_feature_vector as cfv
from Create_Feature_Vector import check_copy_database as ccd
from SVM_halv import TrainSupportVectorMachine as TSVM
from SVM_halv import ApplySupportVectorMachine as ASVM
from SVM_halv import GenerateDataSets as GDS
from Plot_Results import PlotResults as plr

    
def Call_Main(MasterFile):

    tcm = time.time()

    ## Outflow Master File
    MasterFileDir = str(Path(__file__).parent.resolve()) + "/"

    ## Read the master tree
    try:
        MasterTree = ET.parse(os.path.join(MasterFileDir+MasterFile))

    except Exception as e:
        exception_type, exception_object, exception_traceback = sys.exc_info()
        filename = exception_traceback.tb_frame.f_code.co_filename
        line_number = exception_traceback.tb_lineno
        
        print("\nReading the data from the %s file failed." %(MasterFile))
        print("Its directory was indicated as %s" %(MasterFileDir))
        print("Exception type:\t%s" %(exception_type))
        print("File name:\t%s" %(filename))
        print("Line number:\t%s" %(line_number))
        print("The error itself:\n%s\n\n" %(e))        
        print("\n\nPlease indicate a valid data file and directory.")

        sys.exit(1)


    ## Get local name
    namelocal = MasterTree.find("namelocal").text
    
    ## Give the sub roots for a local machine and a server
    if platform.node() == namelocal:
        flagroot = MasterTree.find("flags")
        pathroot = MasterTree.find("pathslocal")
        fileroot = MasterTree.find("fileslocal")
        pararoot = MasterTree.find("parameters")

        #print(pathroot)

    else:
        flagroot = MasterTree.find("flags")
        pathroot = MasterTree.find("pathsserver")
        fileroot = MasterTree.find("filesserver")
        pararoot = MasterTree.find("parameters")

    ##----------------------------------------------------------------------------------------------------------------------------------------------------
    ## initialize dictionary for model parameters
    MyParameters = {}

    ## Get the given flags
    MyParameters['MASK'] = strtobool(flagroot.find("MASK").text)
    MyParameters['CREATE_FV'] = strtobool(flagroot.find("CREATE_FV").text)
    MyParameters['TRAIN_SVM'] = strtobool(flagroot.find("TRAIN_SVM").text)
    MyParameters['HALV'] = strtobool(flagroot.find("HALV").text)
    MyParameters['Plot_GS'] = strtobool(flagroot.find("Plot_GS").text)
    MyParameters['Grid_SVM'] = strtobool(flagroot.find("Grid_SVM").text)
    MyParameters['create_fits'] = strtobool(flagroot.find("create_fits").text)
    MyParameters['CHECK_FV'] = strtobool(flagroot.find("CHECK_FV").text)
    MyParameters['CHECK_FV2D'] = strtobool(flagroot.find("CHECK_FV2D").text)
    MyParameters['CHECK_FI'] = strtobool(flagroot.find("CHECK_FI").text)
    MyParameters['Plot_CM'] = strtobool(flagroot.find("Plot_CM").text)
    MyParameters['FV_CUBES'] = strtobool(flagroot.find("FV_CUBES").text)
    MyParameters['FixRandom'] = strtobool(flagroot.find("FixRandom").text)
    MyParameters['DatabaseStatistic'] = strtobool(flagroot.find("DatabaseStatistic").text)
    MyParameters['DatabaseAutomatic'] = strtobool(flagroot.find("DatabaseAutomatic").text)
    MyParameters['DatabaseManual'] = strtobool(flagroot.find("DatabaseManual").text)
    MyParameters['ProductionRun'] = strtobool(flagroot.find("ProductionRun").text)
    MyParameters['ThesisMode'] = strtobool(flagroot.find("ThesisMode").text)


    ## Read in other relevant stuff from the XML file
    ## read the paths from the Master File
    MyParameters['SavePath'] = pathroot.find("SavePath").text                                   ## Path to save in the results
    MyParameters['ReadPath'] = pathroot.find("ReadPath").text                                   ## Path to read in most files
    MyParameters['DatabasePath'] = pathroot.find("DatabasePath").text                           ## Path to read in the database
    MyParameters['ReadFV'] = pathroot.find("Read_FV_Path").text                                 ## Path to read in the feature vectors
    MyParameters['ReadSVM'] = pathroot.find("Read_SVM_Path").text                               ## Path to read in the SVM

    ## read in the files names from the Master File
    MyParameters['Read_SVM_name'] = fileroot.find("SVM_file").text                              ## Name of the saved SVM
    MyParameters['database_file'] = fileroot.find("database_file").text                         ## Name of the database file
    MyParameters['database_name'] = ".".join(MyParameters['database_file'].split(".")[:-1])     ## Name of the database
    
    ## Subcube size
    MyParameters['SigmaSC'] = float(pararoot.find("SigmaSC").text)                              ## Spartial sub cube extend
    MyParameters['VelocitySC'] = float(pararoot.find("VelocitySC").text)                        ## Velocity sub cube extend

    ## read in the training feature values
    MyParameters['kernel'] = pararoot.find("kernel").text                                       ## the used kernel
    MyParameters['train_size'] = float(pararoot.find("train_size").text)                        ## define size of the test data
    MyParameters['SigmaLevel'] = pararoot.find("SigmaLevel").text                               ## define sigma noise level, outdated but could be relevant to only select certain data
    if MyParameters["SigmaLevel"] == "None" or MyParameters["SigmaLevel"] == "none":            ## Turn the SigmaLevel to None or a float
        MyParameters["SigmaLevel"] = None
    else:
        MyParameters['SigmaLevel'] = float(MyParameters["SigmaLevel"])

    ## Grid search parameters
    MyParameters['grid_scoring'] = pararoot.find("grid_scoring").text                           ## Define grid search scoring function
    MyParameters['gammaspacing'] = pararoot.find("gammaspacing").text                           ## spacing of gamma grid
    MyParameters['gamman'] = int(pararoot.find("gamman").text)                                  ## number of gamma grid components
    MyParameters['gammamin'] = float(pararoot.find("gammamin").text)                            ## minimal gamma grid value
    MyParameters['gammamax'] = float(pararoot.find("gammamax").text)                            ## maximal gamma grid value
    MyParameters['cspacing'] = pararoot.find("cspacing").text                                   ## spacing of c grid
    MyParameters['cn'] = int(pararoot.find("cn").text)                                          ## number of c grid components
    MyParameters['cmin'] = float(pararoot.find("cmin").text)                                    ## minimal c grid value
    MyParameters['cmax'] = float(pararoot.find("cmax").text)                                    ## maximal c grid value

    ## Other parameters
    MyParameters['fdmp'] = float(pararoot.find("fdmp").text)                                    ## feature distribution plot ranges
    MyParameters['fi_scoring'] = pararoot.find("fi_scoring").text                               ## feature importance score function

    ## Get the number of processes
    MyParameters['npro'] = int(pararoot.find("npro").text)                                      ## define number of used processes
    MyParameters['npro'] = min(MyParameters['npro'], int(np.floor(mp.cpu_count()*.8)))          ## ensure that the number is smaller than 90% of the computer cores
    
    ## Set the random seed if requested
    if MyParameters["FixRandom"] == True:
        np.random.seed(42)


    ## create current and all used paths
    ## Time-stamp for parent dir to create unique dirs
    currenttime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    ## Name all other files
    MyParameters['NameDir'] = "%s_%s/" %(currenttime, MyParameters["database_file"][:-4])
    MyParameters['OutputPath'] = MyParameters['SavePath'] + "Output/"
    MyParameters['OutputPath'] = MyParameters['SavePath']
    MyParameters['DatePath'] = MyParameters['OutputPath'] + MyParameters['NameDir']
    MyParameters['FVPath'] = MyParameters['DatePath'] + "FV/"
    MyParameters['SVMPath'] = MyParameters['DatePath'] + "SVM/"
    MyParameters['CubesPath'] = MyParameters['DatePath'] + "Cubes/"
    MyParameters['PlotPath'] = MyParameters['DatePath'] + "Plots/"
    MyParameters['OtherFilesPath'] = MyParameters['DatePath'] + "Others/"
    MyParameters['TempPath'] = MyParameters['DatePath'] + "Temp/"

    ## Creating all directories
    ## create a new directory for the output if not existing
    if not os.path.isdir(MyParameters['OutputPath']):
        Path(MyParameters['OutputPath']).mkdir(parents=True)

    ## create a new parent directory if not existing
    if not os.path.isdir(MyParameters['DatePath']):
        Path(MyParameters['DatePath']).mkdir(parents=True)

    ## create a new directory for the FV if not existing
    if not os.path.isdir(MyParameters['FVPath']):
        Path(MyParameters['FVPath']).mkdir(parents=True)

    ## create a new directory for the SVM if not existing
    if not os.path.isdir(MyParameters['SVMPath']):
        Path(MyParameters['SVMPath']).mkdir(parents=True)

    ## create a new directory for the cubes if not existing
    if not os.path.isdir(MyParameters['CubesPath']):
        Path(MyParameters['CubesPath']).mkdir(parents=True)

    ## create a new directory for the plots if not existing
    if not os.path.isdir(MyParameters['PlotPath']):
        Path(MyParameters['PlotPath']).mkdir(parents=True)

    ## create a new directory for other files if not existing
    if not os.path.isdir(MyParameters['OtherFilesPath']):
        Path(MyParameters['OtherFilesPath']).mkdir(parents=True)

    ## create a new directory for temporary files if not existing
    if not os.path.isdir(MyParameters['TempPath']):
        Path(MyParameters['TempPath']).mkdir(parents=True)

    ## Used files names
    MyParameters['SVM_name'] = "%s_SVM.pkl" %(MyParameters['database_name'])

    ## SVM database
    MyParameters['SVM_database'] = "%sSVM_database.csv" %(MyParameters["SVMPath"])

    ## Copy database to dir
    shutil.copyfile("%s%s" %(MyParameters["DatabasePath"], MyParameters["database_file"]),
                    "%s%s" %(MyParameters["DatePath"], MyParameters["database_file"]))

    ## Shortened database name, contains only unique values and file names
    MyParameters["database_short"] = "Database_short.csv"

    ## Ensure the database names are not identical
    if MyParameters["database_file"] == MyParameters["database_short"]:
        MyParameters["database_short"] = "Database_short_1.csv"

    ## Check the database format, the MASK flag and save a copy of the database to the DataPath
    print("Start Check and Copy the database!")
    ts = time.time()
    MyParameters["MASK"] = ccd(MyParameters)
    if time.time()-ts <= 50:
        print("ccd took %.3f s.\n" %(time.time()-ts))
    else:
        print("ccd took %s.\n" %(time.strftime('%Hh:%Mm:%Ss', time.gmtime(time.time()-ts))))

    ## Generate the automatic and manual datasets
    if MyParameters["MASK"] == True:
        print("Start Generate Data Sets!")
        ts = time.time()
        MyParameters = GDS(MyParameters)
        if time.time()-ts <= 50:
            print("GDS took %.3f s.\n" %(time.time()-ts))
        else:
            print("GDS took %s.\n" %(time.strftime('%Hh:%Mm:%Ss', time.gmtime(time.time()-ts))))

    ## Calculate feature vectors if requested
    if MyParameters['CREATE_FV'] == True:
        ## Changing the FV read path to the FV path as one wants to use out the just created FV
        MyParameters['ReadFV'] = MyParameters['FVPath']

        print("Start Calculating Feature Vectors!")
        ts = time.time()
        cfv(MyParameters)
        if time.time()-ts <= 50:
            print("cfv took %.3f s.\n" %(time.time()-ts))
        else:
            print("cfv took %s.\n" %(time.strftime('%Hh:%Mm:%Ss', time.gmtime(time.time()-ts))))


    ## Train SVM and test it afterwards
    if MyParameters['TRAIN_SVM'] == True and MyParameters['MASK'] == True:

        ## Concentrating the Feature Vectors
        print("Start Concentrating the Featrue Vectors!")
        ts = time.time()
        ctfv(MyParameters)
        if time.time()-ts <= 50:
            print("ctfv took %.3f s.\n" %(time.time()-ts))
        else:
            print("ctfv took %s.\n" %(time.strftime('%Hh:%Mm:%Ss', time.gmtime(time.time()-ts))))

        ## Changing the SVM read path to the SVM path as one wants to use out the just created SVM
        MyParameters['ReadSVM'] = MyParameters['SVMPath']

        print("Start Training a SVM!")
        ts = time.time()
        MyParameters = TSVM(MyParameters)

        if time.time()-ts <= 50:
            print("TSVM took %.3f s.\n" %(time.time()-ts))
        else:
            print("TSVM took %s.\n" %(time.strftime('%Hh:%Mm:%Ss', time.gmtime(time.time()-ts))))

    ## Apply the SVM to full cubes
    print("Start Applying the SVM!")
    ts = time.time()
    ASVM(MyParameters, mode="full")
    if time.time()-ts <= 50:
        print("ASVM_1 took %.3f s.\n" %(time.time()-ts))
    else:
        print("ASVM_1 took %s.\n" %(time.strftime('%Hh:%Mm:%Ss', time.gmtime(time.time()-ts))))

    ## Generate accuracy plots
    print("Start Plot Results!")
    ts = time.time()
    plr(MyParameters, mode="full")
    if time.time()-ts <= 50:
        print("plr_1 took %.3f s.\n" %(time.time()-ts))
    else:
        print("plr_1 took %s.\n" %(time.strftime('%Hh:%Mm:%Ss', time.gmtime(time.time()-ts))))

    ## Remove the temp directory
    print("Start removing the unwanted files!")
    if os.path.isdir(MyParameters['TempPath']):
        shutil.rmtree(MyParameters['TempPath'])

    if MyParameters["ProductionRun"] == True:
        # Remove FV dir
        shutil.rmtree(MyParameters['FVPath'])

        # Remove Others dir
        shutil.rmtree(MyParameters['OtherFilesPath'])

    print("\n\nAll done.\nThe whole run took %s." %(time.strftime('%Hh:%Mm:%Ss', time.gmtime(time.time()-tcm))))

    return MyParameters['ReadSVM']

if __name__ == '__main__':
    MasterFile = "SVMMaster.xml"
    Call_Main(MasterFile)