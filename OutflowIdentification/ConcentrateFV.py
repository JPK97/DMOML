# -*- coding: iso-8859-1 -*-
#@author: Phillip Kahl

## Import existing packages
import os
import numpy as np
import pandas as pd
import pickle as pkl
import psutil
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def ConcentrateTrainFV(CalcParameters):

    print("Start concentrating training data.")

    ## Read the dataset and create a copy of it
    data_pd = pd.read_csv("%s%s" %(CalcParameters["DatePath"], CalcParameters["database_short"]), index_col=0)
    data_pd_test = data_pd
    data_pd_test["Parent dir"] = CalcParameters["TempPath"]

    ### Create an empyt lists to buffer in the concentrated FV names and their number of data points
    cFVnsheader = ["Concentrated name", "Number of vectors"]
    cFVns = list()

    ## Concentrated FV file index
    cfvi = 1

    ## Initiate the concentrated arrays
    train_con = pd.DataFrame()

    ## Initiate the standard scaler
    sc = StandardScaler()

    ## Train a SVM for each FV set
    for index, row in data_pd.iterrows():

        if not index+1 == len(data_pd.index):
            print("Concentrating the FV-set Nr. %i out of %i." %(index+1, len(data_pd.index)), end="\r")
        else:
            print("Concentrating the FV-set Nr. %i out of %i." %(index+1, len(data_pd.index)))
    
        ## Set the rng TBD
        rng = np.random.RandomState(0)

        ## Get the name of the feature vectors
        Cube_Name = row["Cube name"]

        ## Get the file size and avilible memory in bytes
        FV_memory = os.stat("%s%s_FV.pkl" %(CalcParameters["ReadFV"], ".".join(Cube_Name.split(".")[:-1]))).st_size
        free_memory = psutil.virtual_memory().available

        ## If the free_memory gets short, save the file to disc and reset the arrays
        if FV_memory*3 >= free_memory:

            print("The availible memory is about to be exceeded. Save the file for the %i-th time." %(cfvi))

            cFVns = np.append((cFVns, ("Combined_Training_Train_Set_%i.csv" %(cfvi), len(train_con.index))))

            train_con.index = np.linspace(1, len(train_con.index), len(train_con.index), dtype=int)

            ## save the array
            train_con.to_csv("%sCombined_Training_Train_Set_%i.csv" %(CalcParameters["TempPath"], cfvi))

            ## Increasing count and reseting data frame
            cfvi += 1
            train_con = pd.DataFrame()

        ## Reading the FV (and get its header in 1st run)
        if index == data_pd.first_valid_index():
            FV_dset = pd.read_pickle("%s%s_FV.pkl" %(CalcParameters["ReadFV"], ".".join(Cube_Name.split(".")[:-1])))
            FV_header = FV_dset.columns[4:]
            FV_dset = FV_dset.to_numpy()

        else:
            FV_dset = pd.read_pickle("%s%s_FV.pkl" %(CalcParameters["ReadFV"], ".".join(Cube_Name.split(".")[:-1]))).to_numpy()

        ## Drop all rows containing nans
        FV_dset = FV_dset[~np.isnan(FV_dset).any(axis=1)]

        if not CalcParameters["SigmaLevel"] == None:
            FV_dset = FV_dset[np.greater(FV_dset.T[3], CalcParameters["SigmaLevel"])]

        ## Gain the features
        X = FV_dset[:,5:]
        y = np.array(FV_dset[:,4], dtype=int)

        ## Splitting the data into test and training set and free memory space
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=CalcParameters["train_size"], random_state=rng, stratify=y)

        X, y = 0, 0

        ## Update the standard scaler
        sc.partial_fit(X_train)

        ## Save the test data to file, free memory space, and update the test data base
        test_data = pd.DataFrame(data=np.concatenate((y_test.reshape(-1,1), X_test), axis=1), columns=FV_header)
        X_test, y_test = 0, 0

        test_data.to_csv("%sTraining_Test_Set_%s.csv" %(CalcParameters["TempPath"], index))

        test_data = 0

        data_pd_test.loc[index, "Cube name"] = "Training_Test_Set_%s.csv" %(index)

        ## Concentrate the training data and free memory space
        train_data = pd.DataFrame(data=np.concatenate((y_train.reshape(-1,1), X_train), axis=1), columns=FV_header)
        X_train, y_train = 0, 0

        if train_con.empty:
            train_con = train_data

        else:
            train_con = pd.concat([train_con, train_data])
    
        train_data = 0

    ## Update file names and save results to file
    cFVns = np.append(cFVns, ("Combined_Training_Train_Set_%i.csv" %(cfvi), len(train_con.index)))
    cFVns = pd.DataFrame(data=cFVns.reshape(cfvi,2), columns=cFVnsheader)
    cFVns.to_csv("%sConcentrated_FV_Names.csv" %(CalcParameters["TempPath"]))

    ## Save the name list to file
    train_con.index = np.linspace(1, len(train_con.index), len(train_con.index), dtype=int)
    train_con.to_csv("%sCombined_Training_Train_Set_%i.csv" %(CalcParameters["TempPath"], cfvi))

    ## Save the test data base to the temporary data base
    data_pd_test.to_csv("%sReduced_Test_DataBase.csv" %(CalcParameters["TempPath"]))

    ## Save the standard scaler
    pkl.dump(sc, open('%sscaler.pkl' %(CalcParameters['SVMPath']),'wb'))

    ## Return nothing
    return
