# -*- coding: iso-8859-1 -*-

"""
Created on Thu Jun 17 16:39:25 2021

@author: Phillip
"""

##############################################################################
### Include some packages

import numpy as np                                                                          ## import numpy package
import matplotlib.pyplot as plt                                                             ## import matplotlib package
import pandas as pd                                                                         ## import pandas package
from datetime import datetime as datetime                                                   ## import datetime package

def CalcStat(x, y, nbins=10):
    ## Function to bin the data and calculate the std in x and y direction
    ## Determine counts and position of each bin
    n, pos = np.histogram(x, bins=nbins)

    ## Remove empty bins
    no = n
    nt = np.append(n, 1)
    pos = pos[nt != 0]
    n = n[n != 0]

    ## Calculate the weighted counts to determine the mean and std
    sx, _ = np.histogram(x, bins=pos, weights=x)
    sx2, _ = np.histogram(x, bins=pos, weights=x*x)

    ## Calculte the mean and std of each bin
    xdata = sx / n
    xerr = np.sqrt(np.maximum(sx2/n - xdata*xdata, 0))

    ## Same for y
    sy, _ = np.histogram(x, bins=nbins, weights=y)
    sy2, _ = np.histogram(x, bins=nbins, weights=y*y)
    sy = sy[no != 0]
    sy2 = sy2[no != 0]

    ydata = sy / n
    yerr = np.sqrt(np.maximum(sy2/n - ydata*ydata, 0))

    ## If an error is 0 set it to None
    if np.all(xerr == 0):
        xerr = None

    if np.all(yerr == 0):
        yerr = None

    ## Return the results
    return xdata, ydata, xerr, yerr


def PlotDatabase(database, yaxis=None, xerr=None, yerr=None, split_set=False, ThesisMode=False, savename=None, savepath=None):

    ## Define color and shape list
    colors = ["b", "g", "r", "c", "m", "y", "k"]
    shapes = ["P", "s", "D", "*", "o", "^", ">", "X"]

    ## Get x information
    xlabel = database.columns[0]
    xdata = database[xlabel]

    ## Number of datapoints
    ndp = xdata.nunique()
        
    ## Get number of bins
    ## Check for integer divisor between 5 and 15
    lod = []
    for i in range(5,15):
        if ndp%i == 0:
            lod.append(i)

    ## List --> array
    lod = np.array(lod)

    ## If there is no result set the number of bins to 10 or the max number of unique datapoints
    if len(lod) == 0:
        nbins = min(10, ndp)

    ## If there is just one result take it
    elif len(lod) == 1:
        nbins = lod[0]

    ## If there are more candinates, find the closest to 10
    else:
        idx = (np.abs(lod - 10)).argmin()
        nbins = lod[idx]

    ## Initiate figure
    if ThesisMode == False:
        fig, ax = plt.subplots(1,figsize=(12, 12))
    else:
        fig, ax = plt.subplots(1,figsize=(7.47, 7.47))

    ## Split database or not
    if split_set == False:
        ## Plot all columns against the 1st
        for i in range(1,len(database.columns)):

            ## Get data, (errors) and label of column
            label = database.columns[i]
            ydata = database[label]
            if xerr is not None:
                xerri = xerr[i-1].T
            if yerr is not None:
                yerri = yerr[i-1].T

            ## If there are just a few (<20) data points, just plot them
            if len(xdata) < 20:
                ## Scatter the data points or plot errorbars
                if xerr is not None and yerr is not None:
                    ax.errorbar(xdata, ydata, xerr=xerri, yerr=yerri, fmt="%ss" %(colors[(i-1)%len(colors)]), alpha=0.5, label="%s" %(label), linewidth=3, elinewidth=.5, ecolor="%s" %(colors[(i-1)%len(colors)]), capsize=5, capthick=.5)
                elif xerr is not None and yerr is None:
                    ax.errorbar(xdata, ydata, xerr=xerri, fmt="%ss" %(colors[(i-1)%len(colors)]), alpha=0.5, label="%s" %(label), linewidth=3, elinewidth=.5, ecolor="%s" %(colors[(i-1)%len(colors)]), capsize=5, capthick=.5)
                elif xerr is None and yerr is not None:
                    #print("Prior errorbar if.")
                    ax.errorbar(xdata, ydata, yerr=yerri, fmt="%ss" %(colors[(i-1)%len(colors)]), alpha=0.5, label="%s" %(label), linewidth=3, elinewidth=1, ecolor="%s" %(colors[(i-1)%len(colors)]), capsize=5, capthick=2.5)
                else:
                    ax.scatter(xdata, ydata, c="%s" %(colors[(i-1)%len(colors)]), alpha=0.5, label="%s" %(label))

            else:
                ## Bin data and get the standart derivative
                xdb, ydb, xds, yds = CalcStat(xdata, ydata, nbins)

                ## Scatter the data points or plot errorbars
                if xerr is not None and yerr is not None:
                    ax.errorbar(xdata, ydata, xerr=xerri, yerr=yerri, fmt="%ss" %(colors[(i-1)%len(colors)]), alpha=0.15, label="%s - full data" %(label), linewidth=3, elinewidth=.5, ecolor="%s" %(colors[(i-1)%len(colors)]), capsize=5, capthick=.5)
                elif xerr is not None and yerr is None:
                    ax.errorbar(xdata, ydata, xerr=xerri, fmt="%ss" %(colors[(i-1)%len(colors)]), alpha=0.15, label="%s - full data" %(label), linewidth=3, elinewidth=.5, ecolor="%s" %(colors[(i-1)%len(colors)]), capsize=5, capthick=.5)
                elif xerr is None and yerr is not None:
                    #print("Prior errorbar else.")
                    ax.errorbar(xdata, ydata, yerr=yerri, fmt="%ss" %(colors[(i-1)%len(colors)]), alpha=0.15, label="%s - full data" %(label), linewidth=3, elinewidth=.5, ecolor="%s" %(colors[(i-1)%len(colors)]), capsize=5, capthick=.5)
                else:
                    ax.scatter(xdata, ydata, c="%s" %(colors[(i-1)%len(colors)]), alpha=0.15, label="%s - full data" %(label))

                ## Plot the binned data points
                ax.errorbar(xdb, ydb, xerr=xds, yerr=yds, fmt="%s%s" %(colors[(i-1)%len(colors)], shapes[int(np.floor((i-1)/len(colors)))]),
                            linewidth=3, elinewidth=.5, ecolor="%s" %(colors[(i-1)%len(colors)]), capsize=5, capthick=.5, label="%s - binned" %(label))

    else:
        ## Scatter the data points
        ydata = database[database.columns[1]]
        ax.scatter(xdata, ydata, c="%s" %(colors[0]), alpha=0.15, label="All data")
        
        ## Plot all columns against the 1st
        for i in range(2,len(database.columns)):

            ## Get data and label of column
            label = database.columns[i]
            cdata = database[label]

            ## Get unique class data
            ucdata = cdata.unique()

            ## Iterate over unique class data
            for ii, cd in enumerate(ucdata):

                ## Find all matching values
                cdi = cdata.index[cdata == cd].tolist()

                ## Bin data and get the standart derivative
                xdb, ydb, xds, yds = CalcStat(xdata[cdi], ydata[cdi], nbins)

                ## Plot the binned data points
                ax.errorbar(xdb, ydb, xerr=xds, yerr=yds, fmt="%s%s" %(colors[(i-1)%len(colors)], shapes[ii%len(shapes)]),
                            linewidth=3, elinewidth=.5, ecolor='k', capsize=5, capthick=.5, label="%s - %s" %(label, cd))

    ## Name axes and title
    ax.set_xlabel(r"%s" %(xlabel))

    ## Use special labels for the different modes
    if yaxis == "accuracy":
        
        scoremin = database[database.columns[1:]].min().min()
        scoremax = database[database.columns[1:]].max().max()
        
        pymi, pyma = ax.get_ylim()
        if pyma - pymi <= 20:
            pass
        elif pymi <= 40 and pyma >= 60:
            ax.set_ylim(-5, 105)
        elif pymi <= 40:
            ax.set_ylim(0 - abs(scoremax-pyma), pyma)
        elif pyma >= 60:
            ax.set_ylim(pymi, 100 + abs(scoremin-pymi))
        ax.set_ylabel(r"Accuracy [%]")
        ax.set_title("SVM Accuracy")

    elif yaxis == "FPI":
        ax.axhline(y=0, color="gray", linestyle="--", lw=2)
        ax.set_ylabel(r"Decrease in accuracy score")
        ax.set_title("SVM Feature Importance")

    ## Set grid and legend
    ax.grid(color='black', ls='solid', lw=0.1)
    ax.legend()

    ## Save figure if path and name are provided
    if savename != None and savepath != None:
        plt.savefig("%s%s" %(savepath, savename.replace("/", "-")), dpi='figure', format="pdf", metadata=None, bbox_inches="tight", 
                    pad_inches=0.1, backend=None)
        plt.close()

    else:
        plt.show()

def PlotResults(CalcParameters, mode):

    ## Read in data frames
    if CalcParameters["DatabaseAutomatic"] == True or CalcParameters["DatabaseManual"] == True:
        ## Read the score data frame
        if mode == "full":
            data_pd = pd.read_csv("%s%s" %(CalcParameters["DatePath"], CalcParameters["database_short"]), index_col=0)

        elif mode == "test":
            data_pd = pd.read_csv("%sTest-Mode_%s" %(CalcParameters["DatePath"], CalcParameters["database_short"]), index_col=0)

    
        ## Read the feature importance data frame 
        if CalcParameters["CHECK_FI"] and "Mask name" in data_pd.columns and CalcParameters["MASK"] == True:

            ## Feature importance database
            data_pd_fi = pd.read_csv("%sFeatureImportance.csv" %(CalcParameters["DatePath"]))

            ## "Whisker" data even though the model has used violins
            whiskers_data = data_pd_fi.drop([col for col in data_pd_fi.columns if "whiskers length" not in col], axis=1).to_numpy()
            
            ## "Whiskers" data ranges
            whsikers_length = np.zeros(shape=(int(len(whiskers_data[0])/2),len(whiskers_data),2))
            for i in range(int(len(whiskers_data[0])/2)):
                whsikers_length[i] = whiskers_data[:,[2*i,2*i+1]]
            
            ## Buffer to data frame
            data_pd_fi = data_pd_fi.drop([col for col in data_pd_fi.columns if "whiskers length" in col], axis=1)

            ## Get column names from header            
            feature_cols = [col for col in data_pd_fi.columns if "Feature" in col]


    ## Analyze automatic set (and feature importance set)
    if CalcParameters['DatabaseAutomatic'] == True:

        Automatic_Set = CalcParameters['AutomaticSet'] 

        ## Iterate over sets
        for column in Automatic_Set:

            ## Get sub data set
            dataset = data_pd[[column, "Overall score", "Balanced score", "Outflow score", "Non-outflow score"]]
            
            ## Get a time stamp to ensure the plot names are unique
            time_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            ## Generate the plots
            if mode == "full":
                PlotDatabase(dataset, yaxis="accuracy", savename="%s-Automatic-%s.pdf" %(column, time_stamp), ThesisMode=CalcParameters["ThesisMode"], savepath=CalcParameters["PlotPath"]) 
            elif mode == "test":
                PlotDatabase(dataset, yaxis="accuracy", savename="Test-Mode_%s-Automatic-%s.pdf" %(column, time_stamp), ThesisMode=CalcParameters["ThesisMode"], savepath=CalcParameters["PlotPath"]) 

            ## Same for FPI if requested
            if CalcParameters["CHECK_FI"] and "Mask name" in data_pd.columns and CalcParameters["MASK"] == True:

                ## Get sub data set
                dataset = data_pd_fi[np.append(column, feature_cols)]

                ## Generate the plots
                if mode == "full":
                    PlotDatabase(dataset, yaxis="FPI", yerr=whsikers_length, ThesisMode=CalcParameters["ThesisMode"], savename="%s-Automatic_FPI-%s.pdf" %(column, time_stamp), savepath=CalcParameters["PlotPath"]) 
                elif mode == "test":
                    PlotDatabase(dataset, yaxis="FPI", yerr=whsikers_length, ThesisMode=CalcParameters["ThesisMode"], savename="Test-Mode_%s-Automatic_FPI-%s.pdf" %(column, time_stamp), savepath=CalcParameters["PlotPath"]) 



    ## Analyze manual set
    if CalcParameters['DatabaseManual'] == True:

        Manual_Set_Main = CalcParameters['ManualSetMain']
        Manual_Set_Sub = CalcParameters['ManualSetSub']

        ## Iterate over sets
        for Main_Set, Sub_Set in zip(Manual_Set_Main, Manual_Set_Sub):

            ## Create sub data sets to plot them
            Main_Column = data_pd[Main_Set]
            Score_Column = data_pd["Overall score"]
            Sub_Column = data_pd[Sub_Set]

            dataset = pd.concat([Main_Column, Score_Column, Sub_Column], axis=1)
            
            ## Get a time stamp to ensure the plot names are unique
            time_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            ## Generate the plots
            if mode == "full":
                PlotDatabase(dataset, ThesisMode=CalcParameters["ThesisMode"], split_set = True, savename="%s-Manual-%s.pdf" %(column, time_stamp), savepath=CalcParameters["PlotPath"])
                
            elif mode == "test":
                PlotDatabase(dataset, ThesisMode=CalcParameters["ThesisMode"], split_set = True, savename="Test-Mode_%s-Manual-%s.pdf" %(column, time_stamp), savepath=CalcParameters["PlotPath"])

    return