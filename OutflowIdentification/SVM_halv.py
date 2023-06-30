# -*- coding: iso-8859-1 -*-
"""
Created on Mon May 17 11:05:13 2021

@author: Phillip
"""

## Machine Learning Packages
from mlxtend.classifier import EnsembleVoteClassifier

from sklearn.ensemble import BaggingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KernelDensity
from sklearn.svm import SVC

try:
    from sklearn.experimental import enable_halving_search_cv  # noqa
    from sklearn.model_selection import HalvingGridSearchCV
    hgs = True
    
except:
    hgs = False
    
## General Packages
from astropy.io import fits
import itertools
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import os
import pandas as pd
import pickle as pkl
import psutil
import shutil
import sys
import time as ti

## Self-Written Modules
from Create_Feature_Vector import Feature_Extractor as cfv

def GenerateDataSets(CalcParameters):

    ## Read the database
    data_pd = pd.read_csv("%s%s" %(CalcParameters["DatePath"], CalcParameters["database_short"]), index_col=0)

    ## Get all column names that contain non constant vales
    header_full = list(data_pd.loc[:, (data_pd != data_pd.iloc[0]).any()].columns)

    ## ... and are metric
    header_metric = list(data_pd.loc[:, (data_pd != data_pd.iloc[0]).any()].select_dtypes(include=["number"]).columns)

    ## If the database contains less than 1 metric column, one can't use the the automatic and manual option
    ## Turn these options off
    if CalcParameters['DatabaseAutomatic'] == True and len(header_metric) < 1:
        print("WARNNIG: There are no metric columns in the dataset to generate the automatic database. The corresponding flag is set to False.")
        CalcParameters['DatabaseAutomatic'] = False

    if CalcParameters['DatabaseManual'] == True and len(header_metric) < 1:
        print("WARNNIG: There are no metric columns in the dataset to generate the manual database. The corresponding flag is set to False.")
        CalcParameters['DatabaseManual'] = False

    ## If the database contains less than 4 columns, one can't use the manual option
    ## Turn this option off
    if CalcParameters['DatabaseManual'] == True and len(header_full) < 4:
        print("WARNNIG: There are not enought columns in the dataset to generate the manual database. The corresponding flag is set to False.")
        CalcParameters['DatabaseManual'] = False

    ## Set the automatic database if requested
    if CalcParameters['DatabaseAutomatic'] == True and len(header_metric) >= 1:
        ## Get all non-constant, metric datasets from the database
        Automatic_Set = list(data_pd.loc[:, (data_pd != data_pd.iloc[0]).any()].select_dtypes(include=["number"]).columns)

    ## else return an empty set
    else:
        Automatic_Set = []

    ## Buffer the automatic set to the parameter set
    CalcParameters['AutomaticSet'] = Automatic_Set

    ## Set the manual database if requested
    if CalcParameters['DatabaseManual'] == True:
        ## Create empty datasets
        Manual_Set_Main = []
        Manual_Set_Sub = []

        ## Generate several main and sub datasets
        while 1:
            ## Create/empty sub sub set entry
            Manual_Set_Sub_Sub = []

            ## Request input
            print("\nThe non-constant metric column data are:\n%s\n\nPlease select one or type [done] to stop adding, [database] to display the current datasets or [remove] to remove a dataset:" %(header_metric))
            set_main_input = input()
        
            ## Add no entry
            if set_main_input == "done":
                break

            ## Display the dataset
            elif set_main_input == "database":
                if not Manual_Set_Main:
                    print("The database is empty.")

                else:
                    for i, main_entry in enumerate(Manual_Set_Main):
                        print("Dataset %i, main keyword %s:\n%s\n" %(i, main_entry, Manual_Set_Sub[i]))

            ## Remove a dataset
            elif set_main_input == "remove":

                ## Check if the main data set isn't empty
                if not Manual_Set_Main:
                    print("Cannot remove an entry as the list is empty.")

                elif Manual_Set_Main:
                    while 1:
                        ## Request input
                        print("The main database contains the following entries:\n%s\n\nPlease indicate which set you want to remove or type [quit] if you don't want to remove one:" %(Manual_Set_Main))
                        rm_input = input()

                        ## Remove no entry
                        if rm_input == "quit":
                            break

                        ## Check if the input is in the main data set
                        elif rm_input in Manual_Set_Main:
                            ## Check for doubles
                            if Manual_Set_Main.count(rm_input) == 1:
                    
                                ## Get the index and remove it
                                rm_index = Manual_Set_Main.index(rm_input)
                                Manual_Set_Main.pop(rm_index)
                                Manual_Set_Sub.pop(rm_index)

                            ## Indicate which of the doubles one wants to remove
                            else:
                                print("\n\nThere is more than one matching entry in the list. Possible data sets are:\n")
                                rm_set_numbers = []
                                for i, main_entry in enumerate(Manual_Set_Main):
                                    if main_entry == rm_input:
                                        print("Dataset %i, main keyword %s:\n%s\n" %(i, main_entry, Manual_Set_Sub[i]))
                                        rm_set_numbers.append(str(i))

                                while 1:
                                    ## Request input
                                    print("Please indicate which set number %s you want to remove or type [quit] to stop removing an entry:" %(rm_set_numbers))
                                    rm_input_number = input()

                                    ## Remove no entry
                                    if rm_input_number == "quit":
                                        break

                                    ## Remove a dataset
                                    elif rm_input_number in rm_set_numbers:
                                        Manual_Set_Main.pop(int(rm_input_number))
                                        Manual_Set_Sub.pop(int(rm_input_number))
                                        break
                                  

                                    ## Invalid entry
                                    else:
                                        print("The entry is none of the optional ones. Please check your spelling.\n\n")

                                ## Break after the while loop as the removing part is done
                                break

                        ## Invalid entry
                        else:
                            print("The entry is none of the optional ones. Please check your spelling.\n\n")        

                ## Invalid entry
                else:
                    print("The entry is none of the optional ones. Please check your spelling.\n\n")        

            ## Add sub set
            elif set_main_input in header_metric:
                ## Copy the subset of the full, non-constant collums and remove the main entry
                sub_set_options = header_full.copy()
                print(sub_set_options)
                sub_set_options.remove(set_main_input)
                sub_set_options.remove('Parent dir')
                sub_set_options.remove('Cube name')
                sub_set_options.remove('Mask name')

                print("\nHint: It is recommented to generate sub-classes as little as possible!\n")

                while 1:
                    ## Request input and avoid emtpy options
                    if not sub_set_options:
                        print("\nThere are no possible sub sets left.\nPlease type [done] to stop adding, [database] to display the current datasets or [quit] to add none:")
                    else:
                        print("\nPossible sub set entries are:\n%s\n\nPlease select one or type [done] to stop adding, [database] to display the current datasets or [quit] to add none:" %(sub_set_options))
                    
                    set_sub_input = input()

                    ## Stop adding entries
                    if set_sub_input == "done":
                        ## Ensure the sub set is not empty
                        if not Manual_Set_Sub_Sub:
                            print("The subset would be empty; so no data will be added to the main or sub set.")
                            break

                        else:
                            ## Add the main and sub set to the lists
                            print("Add the entries.")
                            Manual_Set_Main.append(set_main_input)
                            Manual_Set_Sub.append((Manual_Set_Sub_Sub))
                            print(Manual_Set_Main, Manual_Set_Sub)
                            break

                    ## Display the dataset
                    elif set_sub_input == "database":
                        print("Main keyword %s; current sub keywords:\n%s\n" %(set_main_input, Manual_Set_Sub_Sub))

                    ## Add none
                    elif set_sub_input == "quit":
                        break

                    ## Add the entry to the sub dataset
                    elif set_sub_input in sub_set_options:
                        sub_set_options.remove(set_sub_input)
                        Manual_Set_Sub_Sub.append(set_sub_input)

                    ## Invalid entry
                    else:
                        print("The entry is none of the optional ones. Please check your spelling.\n\n")

            ## Invalid entry
            else:
                print("The entry is none of the optional ones. Please check your spelling.\n\n")


    ## Else return empty sets
    else:
        Manual_Set_Main = []
        Manual_Set_Sub = []

    ## Buffer the manual main and sub set to the parameters
    CalcParameters['ManualSetMain'] = Manual_Set_Main
    CalcParameters['ManualSetSub'] = Manual_Set_Sub

    ## Return the updated parameters
    return CalcParameters           

def get_scores(gs, CalcParameters=False, is_sh=False, Plot_GS=False):
    """
    A function to analyze and plot the Grid Search results.
    """
    ## Read out the important lines; in case of having, only consider last iteration step
    if is_sh == True:
        results = pd.DataFrame(gs.cv_results_)[["param_C", "param_gamma", "mean_test_score", "iter", "mean_fit_time"]] # .astype(np.float64)
        results = results.drop(results[results["iter"] < results["iter"].max()].index)

    else:
        results = pd.DataFrame(gs.cv_results_)[["param_C", "param_gamma", "mean_test_score", "mean_fit_time"]] # .astype(np.float64)

    ## Get the index with the highest score, if multiple highest scores select the one with the fastest fit time
    imax = results.sort_values(by=["mean_test_score", "mean_fit_time"], ascending=[False, True]).iloc[0]

    ## Read out the max score with its c and gamma configuration
    smax = imax["mean_test_score"]
    cmax = imax["param_C"]
    gmax = imax["param_gamma"]


    if Plot_GS == True:
        results = pd.DataFrame(gs.cv_results_)
        results[["param_C", "param_gamma"]] = results[["param_C", "param_gamma"]].astype(
            np.float64
        )
        if is_sh:
            # SH dataframe: get mean_test_score values for the highest iter
            scores_matrix = results.sort_values("iter").pivot_table(
                index="param_gamma",
                columns="param_C",
                values="mean_test_score",
                aggfunc="last",
            )
        else:
            scores_matrix = results.pivot(
                index="param_gamma", columns="param_C", values="mean_test_score"
            )

        ## Trasform the scores matrix to numpy and flip it
        scores_matrix = np.flip(np.array(scores_matrix), 0)

        ## Create a figure
        if CalcParameters["ThesisMode"] == False:
            fig, ax = plt.subplots(figsize=(8,7))
            fig_log, ax_log = plt.subplots(figsize=(8,7))
        else:
            fig, ax = plt.subplots(figsize=(7.47, 6.54))
            fig_log, ax_log = plt.subplots(figsize=(7.47, 6.54))


        ## Adjust the x (C) and y (gamma) axis
        if CalcParameters["cspacing"] == "linear":
            ds = (CalcParameters["cmax"] - CalcParameters["cmin"]) / (CalcParameters["cn"]-1)
            cmine = CalcParameters["cmin"] - ds/2
            cmaxe = CalcParameters["cmax"] + ds/2
            cmi = cmax - ds/2
            cma = cmax + ds/2
            ax.set_xlabel(r"$C$-Value")
            ax_log.set_xlabel(r"$C$-Value")
        elif CalcParameters["cspacing"] == "log":
            df = (CalcParameters["cmax"]/CalcParameters["cmin"])**(1/(CalcParameters["cn"]-1))
            cmine = np.log10(CalcParameters["cmin"] / np.sqrt(df))
            cmaxe = np.log10(CalcParameters["cmax"] * np.sqrt(df))
            cmi = np.log10(cmax / np.sqrt(df))
            cma = np.log10(cmax * np.sqrt(df))
            ax.set_xlabel(r"log$_{10}(C)$-Value")
            ax_log.set_xlabel(r"log$_{10}(C)$-Value")

        if CalcParameters["gammaspacing"] == "linear":
            ds = (CalcParameters["gammamax"] - CalcParameters["gammamin"]) / (CalcParameters["gamman"]-1)
            gammamine = CalcParameters["gammamin"] - ds/2
            gammamaxe = CalcParameters["gammamax"] + ds/2
            ax.set_ylabel(r"$\gamma$-Value")
            ax_log.set_ylabel(r"$\gamma$-Value")
            gammami = gmax - ds/2
            gammama = gmax + ds/2
        elif CalcParameters["gammaspacing"] == "log":
            df = (CalcParameters["gammamax"]/CalcParameters["gammamin"])**(1/(CalcParameters["gamman"]-1))
            gammamine = np.log10(CalcParameters["gammamin"] / np.sqrt(df))
            gammamaxe = np.log10(CalcParameters["gammamax"] * np.sqrt(df))
            gammami = np.log10(gmax / np.sqrt(df))
            gammama = np.log10(gmax * np.sqrt(df))
            ax.set_ylabel(r"log$_{10}(\gamma)$-Value")
            ax_log.set_ylabel(r"log$_{10}(\gamma)$-Value")

        ## Plot the data
        im = ax.imshow(scores_matrix, extent=[cmine, cmaxe, gammamine, gammamaxe], cmap="viridis")

        ## Highlight the best combination
        ax.plot([cma, cma, cmi, cmi, cma], [gammama, gammami, gammami, gammama, gammama], c="k")
        ax.grid()
        fig.tight_layout()

        ### Add colorbar     
        im_ratio = scores_matrix.shape[0]/scores_matrix.data.shape[1]
        cbar = plt.colorbar(im, fraction=0.047*im_ratio)
        cbar.set_label("Classification Score\nScoring = %s" %(CalcParameters["grid_scoring"]))

        ## Save figure
        fig.savefig("%sParameterGrid_gamma-%.3e_c-%.3e_score-%.3e.pdf" %(CalcParameters['PlotPath'], gmax, cmax, smax), dpi='figure', format="pdf",
                    metadata=None, bbox_inches="tight", pad_inches=0.1, backend=None)

        ## Try to plot the data in logscale
        try:
            im_log = ax_log.imshow(1-scores_matrix+5e-6, extent=[cmine, cmaxe, gammamine, gammamaxe], norm=colors.LogNorm(vmin=max(1e-5, (1-scores_matrix).min())), cmap="viridis_r")
            ax_log.plot([cma, cma, cmi, cmi, cma], [gammama, gammami, gammami, gammama, gammama], c="k")
            ax_log.grid()
            fig_log.tight_layout()

            ### Add colorbar     
            cbar_log = plt.colorbar(im_log, fraction=0.047*im_ratio)
            cbar_log.set_label("Misclassification Score\nScoring = %s" %(CalcParameters["grid_scoring"]))

            fig_log.savefig("%sParameterGrid_LogScale_gamma-%.3e_c-%.3e_score-%.3e.pdf" %(CalcParameters['PlotPath'], gmax, cmax, smax), dpi='figure', format="pdf",
                        metadata=None, bbox_inches="tight", pad_inches=0.1, backend=None)
        except Exception as e:
            exception_type, exception_object, exception_traceback = sys.exc_info()
            filename = exception_traceback.tb_frame.f_code.co_filename
            line_number = exception_traceback.tb_lineno
        
            print("\nPlotting the Grid Search results in log scale failed.")
            print("Exception type:\t%s" %(exception_type))
            print("File name:\t%s" %(filename))
            print("Line number:\t%s" %(line_number))
            print("The error itself:\n%s\n\n" %(e))

        if is_sh:
            print("WARNING: iter is not printed onto imshow plot. Check the code for a first but incomplete draft.")
        #    iterations = results.pivot_table(
        #        index="param_gamma", columns="param_C", values="iter", aggfunc="max"
        #    ).values
        #    for i in range(len(gammas)):
        #        for j in range(len(Cs)):
        #            ax.text(
        #                j,
        #                i,
        #                iterations[i, j],
        #                ha="center",
        #                va="center",
        #                color="w",
        #                fontsize=20,
        #            )

        #fig.subplots_adjust(right=0.8)
        #ax.set_xscale(CalcParameters["cspacing"])
        #ax.set_yscale(CalcParameters["gammaspacing"])
        #ax.set_yscale("log")

        #cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        #fig.colorbar(im, cax=cbar_ax)
        #cbar_ax.set_ylabel("Scoring = %s" %(CalcParameters["grid_scoring"]), rotation=90, va="bottom")
        #cbar.set_label("Scoring = %s" %(CalcParameters["grid_scoring"]))

        plt.close("all")

    return smax, cmax, gmax

## Function to predict the results
def Model_Predict(model, FV):
    
    ## Try to predict the FV
    try:
        return model.predict(FV)
   
    ## If unsuccessful, build sub samples
    except:
        print("Predicting the whole FV set at once failed. Building 3 subsamples.")

        ## Determine new sample size
        order_FV = np.floor(np.log10(len(FV)))
        
        sss  = int(np.floor(len(FV) / (2**order_FV)) * 2**(order_FV-1))

        print("The subsamples have the size of %i data points." %(sss))

        ## Try to predict the results until a certain threshold is undercut
        while sss >= 1e5:

            ## Try to predict the FV
            try:

                ## Define/reset parameters
                FV_pred = np.zeros(len(FV), dtype=int)
                ind0 = 0
                ind1 = sss
                pri = 0

                ## Try to predict all subsets but the last
                while ind1 < len(FV):
          
                    try:

                        FV_pred[ind0:ind1] = model.predict(FV[ind0:ind1])
        
                    except Exception as e:
                        print("Something went wrong...")
                        print("The error:\n" , e)

        
                    ind0 = ind1
                    ind1 += sss

                    if pri == 0:
                        print("Still calculating/", end="\r")
                        pri += 1

                    elif pri == 1:
                        print("Still calculating\\", end="\r")
                        pri += 1

                    else:
                        print("Still calculating-", end="\r")
                        pri -= 2


                ind1 = len(FV)

                ## Predict the remaining set
                try:
                    FV_pred[ind0:ind1] = model.predict(FV[ind0:ind1])
        
                except Exception as e:
                    print("Something went wrong...")
                    print("The error:\n" , e)

                sss = 0

                return FV_pred

            ## Else limit the number of feature vectors to be analyzed at once
            except:

                sss = int(sss/2)

                print("The subsamples were still too large. Reducing their size by 2 to %i data points." %(sss))

    ## If the function failed, return 0 as a score
    print("Warning! Calculating the scoers failed. Returning 0 as a score.")
    return np.array([0])


## A function to plot the confusion matrix
def Plot_Confusion_Matrix(cm, percent_mode=False, ThesisMode=False, class_label = [], pname = "", ppath = "./"):

    if percent_mode == True:
        cm *= 100

    ## Calculate the Confusion Matrix in relative numbers (to actual classification)
    cmr = cm/cm.sum(axis=1).reshape(-1,1) * 100     ## Relative matrix

    ## Get dimension of sqare matrix
    cmfd = np.shape(cm)[0]+2

    ## Calculate class labels if not given
    if not class_label:
        for i in range(cmfd-2):
            class_label = np.append(class_label, "Class %i" %(i+1))

    ## Enlarge matrix to plot class labels
    if percent_mode == False:
        cmf = np.zeros((cmfd, cmfd), dtype=int)
    else:
        cmf = np.zeros((cmfd, cmfd), dtype=float)
    cmf[2:, 2:] = cm
    cmf[1] += cmf.sum(0)
    cmf[:,1] += cmf.sum(1)

    ## Same for relative matrix
    cmfrmask = np.ones((cmfd, cmfd),dtype=bool)
    cmfrmask[2:, 2:] = 0
    cmfr = np.zeros((cmfd, cmfd))
    cmfr[2:, 2:] = cmr
    cmfr = np.ma.masked_array(cmfr, mask=cmfrmask)

    ## Class label space
    cmflmask = np.ones((cmfd, cmfd),dtype=bool)
    cmflmask[2:, 0:2] = 0
    cmfl = np.ones((cmfd, cmfd))
    cmfl = np.ma.masked_array(cmfl, mask=cmflmask)

    ## Initiate figure
    if ThesisMode == False:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig, ax = plt.subplots(figsize=(7.47, 7.47))

    ## Plot a "grid"
    for xi in np.arange(0.5, cmfd, 1):
        plt.plot([xi, xi], [0.5, cmfd-0.5], c="k", ls=":")
    for yi in np.arange(0.5, cmfd, 1):
        plt.plot([0.5, cmfd-0.5], [yi, yi], c="k", ls=":")

    ## Plot the labels
    if percent_mode == False:
        ax.text(1, 1, "Total Population\n%i" %(cmf[1, 1]), color='black', verticalalignment='center', horizontalalignment='center')
        ax.text((cmfd+1)/2, 0.25, "Predicted Classification", color='black', rotation="horizontal", verticalalignment='center', horizontalalignment='center')
        ax.text(0.25, (cmfd+1)/2, "Actual Classification", color='black', rotation="vertical", verticalalignment='center', horizontalalignment='center')

        ## Plot the absolute classification numbers ...
        ## ... in case of binary classes
        if cmfd == 4:
            ax.text(2, 1, "%s (PP)\n%i\n=%.3f %%" %(class_label[0], cmf[1, 2], cmf[1, 2]/cmf[1,1]*100), color='black', verticalalignment='center', horizontalalignment='center')
            ax.text(3, 1, "%s (PN)\n%i\n=%.3f %%" %(class_label[1], cmf[1, 3], cmf[1, 3]/cmf[1,1]*100), color='black', verticalalignment='center', horizontalalignment='center')
            ax.text(1, 2, "%s (P)\n%i\n=%.3f %%" %(class_label[0], cmf[2, 1], cmf[2, 1]/cmf[1,1]*100), color='black', verticalalignment='center', horizontalalignment='center')
            ax.text(1, 3, "%s (N)\n%i\n=%.3f %%" %(class_label[1], cmf[3, 1], cmf[3, 1]/cmf[1,1]*100), color='black', verticalalignment='center', horizontalalignment='center')

            if cmfr[2,2] >= 20:
                ax.text(2, 2, "TP\n%i\n=%.3f %%" %(cmf[2, 2], cmf[2, 2]/cmf[1,1]*100), color='black', verticalalignment='center', horizontalalignment='center')
            else:
                ax.text(2, 2, "TP\n%i\n=%.3f %%" %(cmf[2, 2], cmf[2, 2]/cmf[1,1]*100), color='white', verticalalignment='center', horizontalalignment='center')
        
            if cmfr[2,3] >= 20:
                ax.text(3, 2, "FN\n%i\n=%.3f %%" %(cmf[2, 3], cmf[2, 3]/cmf[1,1]*100), color='black', verticalalignment='center', horizontalalignment='center')
            else:
                ax.text(3, 2, "FN\n%i\n=%.3f %%" %(cmf[2, 3], cmf[2, 3]/cmf[1,1]*100), color='white', verticalalignment='center', horizontalalignment='center')

            if cmfr[3,2] >= 20:
                ax.text(2, 3, "FP\n%i\n=%.3f %%" %(cmf[3, 2], cmf[3, 2]/cmf[1,1]*100), color='black', verticalalignment='center', horizontalalignment='center')
            else:
                ax.text(2, 3, "FP\n%i\n=%.3f %%" %(cmf[3, 2], cmf[3, 2]/cmf[1,1]*100), color='white', verticalalignment='center', horizontalalignment='center')

            if cmfr[3,3] >= 20:
                ax.text(3, 3, "TN\n%i\n=%.3f %%" %(cmf[3, 3], cmf[3, 3]/cmf[1,1]*100), color='black', verticalalignment='center', horizontalalignment='center')
            else:
                ax.text(3, 3, "TN\n%i\n=%.3f %%" %(cmf[3, 3], cmf[3, 3]/cmf[1,1]*100), color='white', verticalalignment='center', horizontalalignment='center')

        ## .. for more than two classes
        else:
            for i in range(cmfd - 2):
                ax.text(i+2, 1, "%s\n%i\n=%.3f %%" %(class_label[i], cmf[1, i+2], cmf[1, i+2]/cmf[1,1]*100), color='black', verticalalignment='center', horizontalalignment='center')
                ax.text(1, i+2, "%s\n%i\n=%.3f %%" %(class_label[i], cmf[i+2, 1], cmf[i+2, 1]/cmf[1,1]*100), color='black', verticalalignment='center', horizontalalignment='center')

            for xi, yi in list(itertools.product(range(cmfd-2), range(cmfd-2))):
                if cmfr[xi+2, yi+2] >= 20:
                    ax.text(xi+2, yi+2, "%i\n=%.3f %%" %(cmf[yi+2, xi+2], cmf[yi+2, xi+2]/cmf[1,1]*100), color='black', verticalalignment='center', horizontalalignment='center')
                else:
                    ax.text(xi+2, yi+2, "%i\n=%.3f %%" %(cmf[yi+2, xi+2], cmf[yi+2, xi+2]/cmf[1,1]*100), color='white', verticalalignment='center', horizontalalignment='center')
        
    elif percent_mode == True:
        ax.text(1, 1, "Total Population\n%.3f %%" %(cmf[1, 1]), color='black', verticalalignment='center', horizontalalignment='center')
        ax.text((cmfd+1)/2, 0.25, "Predicted Classification", color='black', rotation="horizontal", verticalalignment='center', horizontalalignment='center')
        ax.text(0.25, (cmfd+1)/2, "Actual Classification", color='black', rotation="vertical", verticalalignment='center', horizontalalignment='center')

        ## Plot the absolute classification numbers ...
        ## ... in case of binary classes
        if cmfd == 4:
            ax.text(2, 1, "%s (PP)\n%.3f %%" %(class_label[0], cmf[1, 2]), color='black', verticalalignment='center', horizontalalignment='center')
            ax.text(3, 1, "%s (PN)\n%.3f %%" %(class_label[1], cmf[1, 3]), color='black', verticalalignment='center', horizontalalignment='center')
            ax.text(1, 2, "%s (P)\n%.3f %%" %(class_label[0], cmf[2, 1]), color='black', verticalalignment='center', horizontalalignment='center')
            ax.text(1, 3, "%s (N)\n%.3f %%" %(class_label[1], cmf[3, 1]), color='black', verticalalignment='center', horizontalalignment='center')

            if cmfr[2,2] >= 20:
                ax.text(2, 2, "TP\n%.3f %%" %(cmf[2, 2]), color='black', verticalalignment='center', horizontalalignment='center')
            else:
                ax.text(2, 2, "TP\n%.3f %%" %(cmf[2, 2]), color='white', verticalalignment='center', horizontalalignment='center')
        
            if cmfr[2,3] >= 20:
                ax.text(3, 2, "FN\n%.3f %%" %(cmf[2, 3]), color='black', verticalalignment='center', horizontalalignment='center')
            else:
                ax.text(3, 2, "FN\n%.3f %%" %(cmf[2, 3]), color='white', verticalalignment='center', horizontalalignment='center')

            if cmfr[3,2] >= 20:
                ax.text(2, 3, "FP\n%.3f %%" %(cmf[3, 2]), color='black', verticalalignment='center', horizontalalignment='center')
            else:
                ax.text(2, 3, "FP\n%.3f %%" %(cmf[3, 2]), color='white', verticalalignment='center', horizontalalignment='center')

            if cmfr[3,3] >= 20:
                ax.text(3, 3, "TN\n%.3f %%" %(cmf[3, 3]), color='black', verticalalignment='center', horizontalalignment='center')
            else:
                ax.text(3, 3, "TN\n%.3f %%" %(cmf[3, 3]), color='white', verticalalignment='center', horizontalalignment='center')

        ## .. for more than two classes
        else:
            for i in range(cmfd - 2):
                ax.text(i+2, 1, "%s\n%.3f %%" %(class_label[i], cmf[1, i+2]), color='black', verticalalignment='center', horizontalalignment='center')
                ax.text(1, i+2, "%s\n%.3f %%" %(class_label[i], cmf[i+2, 1]), color='black', verticalalignment='center', horizontalalignment='center')

            for xi, yi in list(itertools.product(range(cmfd-2), range(cmfd-2))):
                if cmfr[xi+2, yi+2] >= 20:
                    ax.text(xi+2, yi+2, "%.3f %%" %(cmf[yi+2, xi+2]), color='black', verticalalignment='center', horizontalalignment='center')
                else:
                    ax.text(xi+2, yi+2, "%.3f %%" %(cmf[yi+2, xi+2]), color='white', verticalalignment='center', horizontalalignment='center')
 
    ## Plot the label area and the relative matrix (color-coded)
    ax.imshow(cmfl, vmin=0, vmax=6, cmap="Oranges")
    ax.imshow(cmfl.T, vmin=0, vmax=6, cmap="Purples")
    cmfrp = ax.imshow(cmfr, vmin=0, vmax=100, cmap="viridis")   

    ### Add colorbar     
    im_ratio = cmfr.shape[0]/cmfr.shape[1]
    cbar = plt.colorbar(cmfrp, fraction=0.047*im_ratio)
    cbar.set_label('Accuracy rate [%]\nRelative to Actual Classification')

    ## Adjust plot ranges and turn axis off
    ax.set_xlim(0, cmfd-0.5)
    ax.set_ylim(cmfd-0.5, 0)
    plt.axis('off')

    ## Title
    plt.title("Confusion Matrix")

    ## Save the Confusion matrix if a name is given or plot it
    if pname != "":
        fig.savefig("%s%s" %(ppath, pname), dpi='figure', format="pdf",
                    metadata=None, bbox_inches="tight", pad_inches=0.1, backend=None)
        plt.close("all")

    else:
        plt.show()

    return cm

## True Negative Rate
def TNR(est, X, yacc):
    ypre = est.predict(X)
    tn, fp, fn, tp = confusion_matrix(yacc, ypre).ravel()
    return tn / (tn+fp)

## True Positive Rate
def TPR(est, X, yacc):
    ypre = est.predict(X)
    tn, fp, fn, tp = confusion_matrix(yacc, ypre).ravel()
    return tp / (tp+fn)

## True Positive Positive Negative Rate
def TPPNR(est, X, yacc):
    ypre = est.predict(X)
    tn, fp, fn, tp = confusion_matrix(yacc, ypre).ravel()
    return (2*tp / (tp+fn) + tn / (tn+fp)) / 3

## True Positive Negative Negative Rate
def TPNNR(est, X, yacc):
    ypre = est.predict(X)
    tn, fp, fn, tp = confusion_matrix(yacc, ypre).ravel()
    return (tp / (tp+fn) + 2*tn / (tn+fp)) / 3
       

def TrainSupportVectorMachine(CalcParameters):
    """
    A function to train a Support Vector Machine on the generated FV sets.
    """

    print("Start training.")

    ## Read the dataset
    data_pd = pd.read_csv("%sConcentrated_FV_Names.csv" %(CalcParameters["TempPath"]), index_col=0)

    ## Create an empyt list to buffer in the SVMs/classifiers, number of training data points and feature Percentiles
    ## if there is more than one set of FV
    if len(data_pd.index) > 1:
        clfs = list()
        ndps = list()
        pers = list()

    ## Train a SVM for each FV set
    for index, row in data_pd.iterrows():

        print("Training a SVM for FV-set Nr. %i out of %i." %(index+1, len(data_pd.index)))
    
        ## Set the rng TBD
        rng = np.random.RandomState(0)

        ## Get the directory name of the concentrated FVs and the number of included vectors
        CFVN = row["Concentrated name"]
        NoFV = row["Number of vectors"]

        ## Reading the FV and gain the training data set
        FV_dset = pd.read_csv("%s%s" %(CalcParameters["TempPath"], CFVN), index_col=0)  #.to_numpy()

        FV_dset = FV_dset.to_numpy()

        X_train = FV_dset[:,1:]
        y_train = np.array(FV_dset[:,0], dtype=int)

        ## Remove invalid voxel (class -1) from set
        if -1 in y_train:
            X_train = X_train[y_train != -1]
            y_train = y_train[y_train != -1]


        ## Standarize the data
        sc = pkl.load(open('%sscaler.pkl' %(CalcParameters['SVMPath']),'rb'))
        X_train = sc.transform(X_train)

        ## Number of features
        nf = len(X_train[0])

        ## Get the 95% percentiles for each feature of both classes and take the outer ones
        peup_0 = np.nanpercentile(X_train[y_train==0],95,axis=0)
        pedo_0 = np.nanpercentile(X_train[y_train==0],5,axis=0)
        peup_1 = np.nanpercentile(X_train[y_train==1],95,axis=0)
        pedo_1 = np.nanpercentile(X_train[y_train==1],5,axis=0)
        peup = np.max((peup_0, peup_1), axis=0)
        pedo = np.min((pedo_0, pedo_1), axis=0)
        peud = np.array((peup,pedo))


        ## Max number of training points to estimate good gamma and c as well as a bagging sample
        param_lim = 1e5
    
        ## If the trainings set is larger, shorten it to param_lim to avoid long calculation times
        if len(X_train) > param_lim:
        
            X_param, _, y_param, _ = train_test_split(
                X_train, y_train, train_size=int(param_lim), random_state=rng, stratify=y_train)

        ## Else take the whole data set
        else:
        
            X_param = X_train
            y_param = y_train


        ## Check if both gamma and c are fixed by their numbers (=1)
        if CalcParameters["gamman"] == 1 and CalcParameters["cn"] == 1:
            ## If true use the average value
            gammabest = (CalcParameters["gammamax"] + CalcParameters["gammamin"]) / 2
            cbest = (CalcParameters["cmax"] + CalcParameters["cmin"]) / 2

        ## elif check if both are fixed by their minimal and maximal value
        elif CalcParameters["gammamax"] == CalcParameters["gammamin"] and CalcParameters["cmax"] == CalcParameters["cmin"]:
            ## If true use these values
            gammabest = CalcParameters["gammamax"]
            cbest = CalcParameters["cmax"]
            
            ## Set to the number of gammas/cs to 1 as they don't differ
            CalcParameters["gamman"] = 1
            CalcParameters["cn"] = 1
    
        ## else find the best combination of gamma and c
        else:

            tigss = ti.time()
            print("Find the best combination of gamma and c.")

            ## Generate the gamma space in lin or logspace, but avoid the case of gammamax == gammamin
            if CalcParameters["gammamax"] == CalcParameters["gammamin"]:
                gammas = np.array([CalcParameters["gammamax"]])
                ## Set to the number of gammas/cs to 1 as they don't differ
                CalcParameters["gamman"] = 1

            else:
                if CalcParameters["gammaspacing"] == "log":
                    gammas = np.logspace(np.log10(CalcParameters["gammamin"]), np.log10(CalcParameters["gammamax"]), CalcParameters["gamman"])
                elif CalcParameters["gammaspacing"] == "linear":
                    gammas = np.linspace(CalcParameters["gammamin"], CalcParameters["gammamax"], CalcParameters["gamman"])
            
            ## Same for c
            if CalcParameters["cmax"] == CalcParameters["cmin"]:
                cs = np.array([CalcParameters["cmax"]])
                CalcParameters["cn"] = 1

            else:
                if CalcParameters["cspacing"] == "log":
                    cs = np.logspace(np.log10(CalcParameters["cmin"]), np.log10(CalcParameters["cmax"]), CalcParameters["cn"])
                elif CalcParameters["cspacing"] == "linear":
                    cs = np.linspace(CalcParameters["cmin"], CalcParameters["cmax"], CalcParameters["cn"])

            ## Create the grid                
            param_grid = {'gamma': gammas, 'C': cs}

            ## Allocate 90% of the avilible memory
            mem = psutil.virtual_memory().available*7.5e-7

            print("The memory used for the GS is %.3e GB" %(mem/1e3))
            ## Initiallize the SVM for Grid Search
            svm = SVC(random_state=rng, kernel=CalcParameters["kernel"], class_weight="balanced",
                    cache_size=mem/(CalcParameters["npro"]+1))

            ## Enable self-written scoring functions
            if CalcParameters["grid_scoring"] == "TPR" or CalcParameters["grid_scoring"] == "tpr":
                grid_scoring = TPR
            elif CalcParameters["grid_scoring"] == "TNR" or CalcParameters["grid_scoring"] == "tnr":
                grid_scoring = TNR
            elif CalcParameters["grid_scoring"] == "TPPNR" or CalcParameters["grid_scoring"] == "tppnr":
                grid_scoring = TPPNR
            elif CalcParameters["grid_scoring"] == "TPNNR" or CalcParameters["grid_scoring"] == "tpnnr":
                grid_scoring = TPNNR
            else:
                grid_scoring = CalcParameters["grid_scoring"]

            ## If possible and asked, try to find the parameters via halving grid search
            if CalcParameters['HALV'] == True and hgs == True:

                try:
                    gs = HalvingGridSearchCV(estimator=svm, param_grid=param_grid, factor=4,
                                                random_state=rng, n_jobs=CalcParameters["npro"], 
                                                cv=8, scoring=grid_scoring)
                    gs.fit(X_param, y_param)

                ## In case it fails, use GridSearchCV
                except Exception as e:
                    CalcParameters['HALV'] = False
                    print("WARNING: Failed to apply the HalvingGridSearchCV. Set 'HALV'-flag to False and apply GridSearchCV.")
                    print("The error:", e)

            ## Else apply the normal grid search
            if CalcParameters['HALV'] == False or hgs == False:
                gs = GridSearchCV(estimator=svm, param_grid=param_grid, n_jobs=CalcParameters["npro"],
                                    cv=8, scoring=grid_scoring)
                gs.fit(X_param, y_param)

            ## Read out the best score and parameters
            scorebest, cbest, gammabest = get_scores(gs, CalcParameters=CalcParameters, is_sh=np.logical_and(CalcParameters['HALV'], hgs), Plot_GS=CalcParameters["Plot_GS"])

            print("With a score of %.2f%%, the best parameters are:\n\
                    C = %.2e\tgamma = %.2e\n\n\n" %(scorebest*100, cbest, gammabest))
            print("The Grid Search took %.3f s." %(ti.time()-tigss))


        ## Apply bagging classifier
        n_bagging_ideal = 30

        ## Calculate the failure rate of the bagging algorithm
        bicytr = np.bincount(y_train)
        prbf = max((bicytr / bicytr.sum())**(bicytr.sum()/n_bagging_ideal) * (1 - (bicytr  / bicytr.sum())**(bicytr.sum()/n_bagging_ideal))**(n_bagging_ideal - 1) * n_bagging_ideal)
    
        ## If it's likely to fail, redistribute the bagging train sample
        if prbf >= 0.0001:

            ## Parameter limit
            param_lim = bicytr.sum()
        
            print("Warning! Sample is not balanced enough and would have failed with a propabitlity of %.2f%% while bagging.\nEqually redistributing the test data classes (will include doubles), just for the bagging process." %(prbf*100))
        
            ## Fix the random seed, TBD
            np.random.seed(0)

            ## Count number of OF/non-OF voxel
            toc = np.bincount(y_train)[0]
            tac = np.bincount(y_train)[1]
        
            ## Get indices of OF/non-OF voxel
            toi = np.where(y_train == 0)[0]
            tai = np.where(y_train == 1)[0]

            ## If there are enough OF voxel, draw 50% parameter indices without replacement
            if toc >= param_lim * .5:
                toii = np.random.choice(toc, int(param_lim*.5), replace=False)

            ## If not, draw with replacement (--> doubles)
            else:
                toii = np.random.choice(toc, int(param_lim*.5), replace=True)

            ## Same as for outflow case
            if tac >= param_lim * .5:
                taii = np.random.choice(tac, int(param_lim*.5), replace=False)

            else:
                taii = np.random.choice(tac, int(param_lim*.5), replace=True)

            
            ## Get and buffer the new training data
            X_traino = X_train[toi[toii]]
            X_traina = X_train[tai[taii]]
            
            y_traino = y_train[toi[toii]]
            y_traina = y_train[tai[taii]]
            
            X_train_bag = np.append(X_traino, X_traina, axis=0)
            y_train_bag = np.append(y_traino, y_traina, axis=0)   

        else:
            X_train_bag = X_train
            y_train_bag = y_train
        
        
        ## Set the number of bagging samples to the ideal one, it might change later on
        n_bagging = n_bagging_ideal

        print("Start trainin a SVM.")
        tists = ti.time()

        ## Train the SVM model using the Bagging Classifier
        ## In case the training fails, enlarge the training set till a reference 
        while n_bagging <= len(X_train_bag)/10:
            
            try:

                ## Set up the Bagging Classifier
                clf = BaggingClassifier(SVC(kernel=CalcParameters["kernel"], probability=True,
                                            C=cbest, gamma=gammabest,
                                            class_weight="balanced",
                                            random_state=rng,
                                            cache_size=psutil.virtual_memory().available*8e-7,
                                            ),
                                         max_samples=1.0 / n_bagging, 
                                         n_estimators=n_bagging,
                                         max_features = np.shape(X_train)[1],
                                         n_jobs=CalcParameters["npro"])


                ## Train the model
                clf.fit(X_train_bag, y_train_bag)

                ## If trained succesfull, break out of loop
                break

            ## Else increase the sample number
            except Exception as e:
                n_bagging *= 10

                print("Failed to fit the data.")
                print("The error:", e)
                print("Increasing the number of estimators by 10 to reduce data usage. The new number of estimators is %i" %(n_bagging))


        ## In case the bagging was not succesfull, raise an error
        if n_bagging > len(X_train_bag):

            print("\n\nThe BaggingClassifier was not able to fit the data. The training has to be aborted.")

            sys.exit(1)

        print("Done training SVM.")
        print("It took %.3f s." %(ti.time() - tists))

        # Save the model to disk
        if len(data_pd.index) > 1:
            SVM_name = "%sSVM_Sub_%i.pkl" %(CalcParameters["SVMPath"], index)
            pkl.dump(clf, open(SVM_name, 'wb'))
            #clfs = np.append(clfs, SVM_name)
            clfs = np.append(clfs, clf)
            ndps = np.append(ndps, NoFV)

            ## Buffer the Percentiles
            pers = np.append(pers, peud)

        else:
            SVM_name = "%s%s_SVM.pkl" %(CalcParameters["SVMPath"], CalcParameters["database_short"][:-4])
            pkl.dump(clf, open(SVM_name, 'wb'))
             

    ## Add all SVMs up via VotingClassifier if more than one SVM was trained
    if len(data_pd.index) > 1:
        ## Define sample weights
        weights = np.array(ndps)
        weights /= weights.sum()

        ## Set up the SVM model
        SVM_model = EnsembleVoteClassifier(clfs=clfs, voting="soft", weights=weights, use_clones=False, fit_base_estimators=False)

        ## "fit" the SVM_model, doesn't do anything as fit_base_estimators=False
        SVM_model.fit(X_train_bag, y_train_bag)

        ## Save the SVM to disk
        SVM_name = "%s%s_SVM.pkl" %(CalcParameters["SVMPath"], CalcParameters["database_short"][:-4])
        pkl.dump(clf, open(SVM_name, 'wb'))

    ## Plot the feature distribution on a grid
    if CalcParameters["Grid_SVM"] == True:
        print("Start Grid SVM")

        time_grid = ti.time()

        ## Get the percentiles
        if len(data_pd.index) > 1:
            peco = np.nanmean(np.array(pers))
            CalcParameters["pedo"] = peco[0]
            CalcParameters["peup"] = peco[1]

        else:
            CalcParameters["pedo"] = pedo
            CalcParameters["peup"] = peup

        ## Generate overview figure, if there is more than just one figure
        if CalcParameters["ThesisMode"] == False:
            fig_o_kde, ax_o_kde = plt.subplots(nf, nf, figsize=(4*nf+2,4*nf))
            fig_o_kde_1, ax_o_kde_1 = plt.subplots(figsize=(8,8))
            if nf >= 2:
                fig_o_kde_1v2, ax_o_kde_1v2 = plt.subplots(figsize=(8,8))
        else:
            fig_o_kde, ax_o_kde = plt.subplots(nf, nf, figsize=(7.47, 7.47))
            fig_o_kde_1, ax_o_kde_1 = plt.subplots(figsize=(7.47,7.47))
            if nf >= 2:
                fig_o_kde_1v2, ax_o_kde_1v2 = plt.subplots(figsize=(7.47,7.47))

        ## Generate nf random uniform distributed feature vectors in the defined ranges
        nn = int(min(100*10**nf, 5e5))
        rd = np.random.uniform(size=(nf, nn))
        rd = (rd.T*(CalcParameters["peup"]-CalcParameters["pedo"])+CalcParameters["pedo"]) # .T

        if len(data_pd.index) > 1:
            pred = Model_Predict(SVM_model, rd)
        else:
            pred = Model_Predict(clf, rd)

        ## Seperate the two classes
        pred_0 = rd[pred == 0]
        pred_1 = rd[pred == 1]

        ## Define number of plot steps in each direction
        xn = 100
        yn = 100

        ## Iterate over all feature combinations
        for i in range(1,nf+1):
            
            for ii in range(1,nf+1):

                ## Avoid double distribution claculation
                if i <= ii:

                    ## Extend in Fi and Fii direction
                    xmi = CalcParameters["pedo"][i-1]
                    xma = CalcParameters["peup"][i-1]
                    ymi = CalcParameters["pedo"][ii-1]
                    yma = CalcParameters["peup"][ii-1]

                    ## Define the plot and kde grid
                    xf = np.linspace(xmi,xma,xn)
                    yf = np.linspace(ymi,yma,yn)

                    xm, ym = np.meshgrid(xf, yf, indexing="ij")
                    mgi = xf.reshape(-1,1)
                    mgii = np.array([xm.reshape(-1), ym.reshape(-1)]).T

                    ## Get the i_th and ii_th value of the features
                    if i == ii:
                        pred_0_i = pred_0[:,[i-1]]
                        pred_1_i = pred_1[:,[i-1]]

                    else:
                        pred_0_ii = pred_0[:,[i-1,ii-1]]
                        pred_1_ii = pred_1[:,[i-1,ii-1]]

                    
                    ## Fit and plot the results
                    if i == ii:

                        ## Kernel density estimate width
                        bw = (xma - xmi) / max(100, xn/10)

                        ## Do a kernel estimate for both classes and calculate the probability distribution from them
                        if pred_0_i.sum() != 0:
                            kdex0 = KernelDensity(kernel="gaussian", bandwidth=abs(bw)).fit(X=pred_0_i)
                            dex0 = np.exp(kdex0.score_samples(mgi.reshape(-1,1)))
                        else:
                            dex0 = np.zeros_like(mgi)

                        if pred_1_i.sum() != 0:
                            kdex1 = KernelDensity(kernel="gaussian", bandwidth=abs(bw)).fit(X=pred_1_i)
                            dex1 = np.exp(kdex1.score_samples(mgi.reshape(-1,1)))
                        else:
                            dex1 = np.zeros_like(mgi)
                        
                        
                        dex = dex1 / (dex0+dex1)

                        ## Transform decimal to %
                        if nf != 1:
                            dex *= 100

                        ## Plot the results
                        if nf == 1:
                            ax_o_kde.plot(mgi, dex, color="red")
                            ax_o_kde.set_ylim(0,1)
                            ax_o_kde.set_xlabel(r"Feature %i [Standarized]" %(ii))
                            ax_o_kde.set_ylabel(r"Classification")
                            ax_o_kde.grid()
                        
                            if i == 0 and ii == 0:
                                ax_o_kde_1.plot(mgi, dex, color="red")
                                ax_o_kde_1.set_ylim(0,1)
                                ax_o_kde_1.set_xlabel(r"Feature %i [Standarized]" %(ii))
                                ax_o_kde_1.set_ylabel(r"Classification")
                                ax_o_kde_1.grid()
                        
                        else:
                            ax_o_kde[ii-1,i-1].plot(mgi, dex, color="red")
                            ax_o_kde[ii-1,i-1].set_ylim(0,100)
                            ax_o_kde[ii-1,i-1].set_xlabel(r"Feature %i [Standarized]" %(ii))
                            ax_o_kde[ii-1,i-1].set_ylabel(r"'1' classifications in %")
                            ax_o_kde[ii-1,i-1].grid()

                            if i == 0 and ii == 0:
                                ax_o_kde_1.plot(mgi, dex, color="red")
                                ax_o_kde_1.set_ylim(0,100)
                                ax_o_kde_1.set_xlabel(r"Feature %i [Standarized]" %(ii))
                                ax_o_kde_1.set_ylabel(r"'1' classifications in %")
                                ax_o_kde_1.grid()


                    elif nf >= 2:
                    
                        ## Kernel density estimate width
                        bw = max((xma - xmi) / max(100, xn/10), (yma - ymi) / min(100, yn/10))

                        ## Do a kernel estimate for both classes and calculate the probability distribution from them
                        if pred_0_ii.sum() != 0:
                            kdex0 = KernelDensity(kernel="gaussian", bandwidth=abs(bw)).fit(X=pred_0_ii)
                            dex0 = np.exp(kdex0.score_samples(mgii)).reshape(xn, yn)
                        else:
                            dex0 = np.zeros_like(xm)

                        if pred_1_ii.sum() != 0:
                            kdex1 = KernelDensity(kernel="gaussian", bandwidth=abs(bw)).fit(X=pred_1_ii)
                            dex1 = np.exp(kdex1.score_samples(mgii)).reshape(xn, yn)
                        else:
                            dex1 = np.zeros_like(xm)
                        
                        dex = dex1 / (dex0+dex1)
                        
                        if nf != 2:
                            pred *= 100
                            dex *= 100

                        ## Plot extends
                        fxmin = xf[0] - (xf[1]-xf[0])/2
                        fxmax = xf[-1] + (xf[1]-xf[0])/2
                        fymin = yf[0] - (yf[1]-yf[0])/2
                        fymax = yf[-1] + (yf[1]-yf[0])/2

                        ## Plot the results
                        if nf == 2:
                            im_o_kde = ax_o_kde[ii-1,i-1].imshow(dex.T, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1, extent=[fxmin, fxmax, fymin, fymax])

                            if i == 0 and ii == 1:
                                ax_o_kde_1v2.imshow(dex.T, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1, extent=[fxmin, fxmax, fymin, fymax])

                        else:
                            im_o_kde = ax_o_kde[ii-1,i-1].imshow(dex.T, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=100, extent=[fxmin, fxmax, fymin, fymax])

                            if i == 0 and ii == 1:
                                ax_o_kde_1v2.imshow(dex.T, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=100, extent=[fxmin, fxmax, fymin, fymax])

                        ### Add colorbar     
                        im_ratio = dex.shape[0]/dex.data.shape[1]
                        cbar_o_kde = plt.colorbar(im_o_kde, fraction=0.047*im_ratio)

                        if nf == 2:
                            cbar_o_kde.set_label("Classification")

                        else:
                            cbar_o_kde.set_label("'1' classifications in %")
    
                        if i == 0 and ii == 1:
                            cbar_o_kde = plt.colorbar(im_o_kde, fraction=0.047*im_ratio)

                            if nf == 2:
                                cbar_o_kde.set_label("Classification")

                            else:
                                cbar_o_kde.set_label("'1' classifications in %")

                        ## Add x/y labels
                        ax_o_kde[ii-1,i-1].set_xlabel(r"Feature %i [Standarized]" %(i))
                        ax_o_kde[ii-1,i-1].set_ylabel(r"Feature %i [Standarized]" %(ii))
                        ax_o_kde[ii-1,i-1].grid()

                        if i == 0 and ii == 1:
                            ax_o_kde_1v2.set_xlabel(r"Feature %i [Standarized]" %(i))
                            ax_o_kde_1v2.set_ylabel(r"Feature %i [Standarized]" %(ii))
                            ax_o_kde_1v2.grid()


                    ## Buffer the results
                    if i == ii:
                        CalcParameters["SVM_grid_F%i_o" %(i)] = dex

                    else:
                        CalcParameters["SVM_grid_F%ivsF%i" %(i, ii)] = dex

        fig_o_kde.tight_layout()
        fig_o_kde_1.tight_layout()
        if n >= 2:
            fig_o_kde_1v2.tight_layout()

        ## Remove empty plots
        for i in range(nf):
            for ii in range(nf):
                if i > ii:
                    fig_o_kde.delaxes(ax_o_kde[ii, i])

        ## Save and close the plots
        fig_o_kde.savefig("%sModel_Distribution.pdf" %(CalcParameters['PlotPath']), dpi='figure', format="pdf",
                    metadata=None, bbox_inches="tight", pad_inches=0.1, backend=None)

        fig_o_kde_1.savefig("%sModel_Distribution_1v1.pdf" %(CalcParameters['PlotPath']), dpi='figure', format="pdf",
                    metadata=None, bbox_inches="tight", pad_inches=0.1, backend=None)

        if nf >= 1:        
            fig_o_kde_1v2.savefig("%sModel_Distribution_1v2.pdf" %(CalcParameters['PlotPath']), dpi='figure', format="pdf",
                        metadata=None, bbox_inches="tight", pad_inches=0.1, backend=None)

        plt.close("all")

        print("Calculating the SVM Grid took %.2f s." %(ti.time() - time_grid))


    ## Return the updated CalcParameters
    return CalcParameters


def ApplySupportVectorMachine(CalcParameters, mode="full"):
    
    print("Reading SVM.")

    ## Read out SVM
    SVM_name = "%s%s_SVM.pkl" %(CalcParameters["ReadSVM"], CalcParameters["database_short"][:-4])

    svm_trained = open("%s" %(SVM_name), "rb")

    ## Read out standard scaler
    sc = pkl.load(open('%sscaler.pkl' %(CalcParameters['ReadSVM']),'rb'))

    ## Reading SVM
    svm = pkl.load(svm_trained)
    
    ## Closing file
    svm_trained.close()

    ## Adjust the number of jobs
    svm.set_params(**{"n_jobs": CalcParameters["npro"]})

    ## Reading database, depends on mode
    if mode == "full":
        data_pd = pd.read_csv("%s%s" %(CalcParameters["DatePath"], CalcParameters["database_short"]), index_col=0)

    elif mode == "test":
        data_pd = pd.read_csv("%sReduced_Test_DataBase.csv" %(CalcParameters["TempPath"]), index_col=0)

    ## Get name of dirs and cubes; also define mask flag
    Parent_Dirs = data_pd["Parent dir"]
    Cube_Names = data_pd["Cube name"]

    if "Mask name" in data_pd.columns and CalcParameters["MASK"] == True:
        Mask_Names = data_pd["Mask name"]
        Mask_Flag = True

    else:
        Mask_Flag = False

    ## Add new columns at the beginning or reset them to save the model accuracy
    if "Outflow score" in data_pd.columns and Mask_Flag == True:
        data_pd["Outflow score"] = 0.
        data_pd["Non-outflow score"] = 0.
        data_pd["Overall score"] = 0.
        data_pd["Balanced score"] = 0.

    elif Mask_Flag == True:
        data_pd.insert(0, "Outflow score", 0.)
        data_pd.insert(1, "Non-outflow score", 0.)
        data_pd.insert(2, "Overall score", 0.)
        data_pd.insert(3, "Balanced score", 0.)

    ## If SVM wasn't trained, go throught the FV sets to determine the feature percentiles
    if CalcParameters["TRAIN_SVM"] == False:
                
        pers = list()

        for i_set, (Parent_Dir, Cube_Name) in enumerate(zip(Parent_Dirs, Cube_Names)):
 
            FV_Name = "%s_FV.pkl" %(".".join(Cube_Name.split(".")[:-1]))
            ReadPath = CalcParameters["ReadFV"]

            ## Try reading the FV
            if os.path.isfile("%s%s" %(ReadPath, FV_Name)):
        
                FV_data = pd.read_pickle("%s%s" %(ReadPath, FV_Name))


            else:
                print("There is no precalculated FV set named\n%s\nsaved at\n%s\n. Creating it now." %(FV_Name, ReadPath))
                ## Calculate the FV if it does not exist
                ## Pass a small database containing the path and data cube to the cfv function
                if "Mask name" in data_pd.columns and CalcParameters["MASK"] == True:
                    header = ["Parent dir", "Cube name", "FV name", "Mask name"]
                    data = [[Parent_Dir, Cube_Name, FV_Name, Mask_Names]]

                ## And add mask information if necessary
                else:
                    header = ["Parent dir", "Cube name", "FV name"]
                    data = [[Parent_Dir, Cube_Name, FV_Name]]

                data_set = pd.DataFrame(data=data, columns=header, index=None)

                ## Calculate the FV
                FV_data = cfv(CalcParameters, data_set=data_set)
    
            FV_data = FV_data.to_numpy()
            y = FV_data[:,4]
            FV = FV_data[:,5:]
            FV = sc.transform(FV)

            ## Determining the percentiles ...
            if Mask_Flag == True:
                peup_0 = np.nanpercentile(FV[y==0],95,axis=0)
                pedo_0 = np.nanpercentile(FV[y==0],5,axis=0)
                peup_1 = np.nanpercentile(FV[y==1],95,axis=0)
                pedo_1 = np.nanpercentile(FV[y==1],5,axis=0)
                peup = np.nanmax((peup_0, peup_1), axis=0)
                pedo = np.nanmin((pedo_0, pedo_1), axis=0)

            else:
                peup = np.nanpercentile(FV,95,axis=0)
                pedo = np.nanpercentile(FV,5,axis=0)

            ## ... and combine them
            peud = np.array((peup,pedo))

            ## Buffer the Percentiles
            pers = np.append(pers, peud)

        ## Determine the upper and lower percentiles, buffer them
        peco = np.array(np.nanmean(pers.reshape(len(Parent_Dirs),-1), axis=0)).reshape(-1,np.shape(FV)[1])
        CalcParameters["pedo"] = peco[1]
        CalcParameters["peup"] = peco[0]


    ## Loop over all data sets
    for i_set, (df_index, Parent_Dir, Cube_Name) in enumerate(zip(data_pd.index, Parent_Dirs, Cube_Names)):

        tiis = ti.time()

        ## Get the name of the FV
        if mode == "full":
            FV_Name = "%s_FV.pkl" %(".".join(Cube_Name.split(".")[:-1]))
            ReadPath = CalcParameters["ReadFV"]

        elif mode == "test":
            FV_Name = Cube_Name
            ReadPath = Parent_Dir

        ## Return a status update
        print("\n\nApplying the SVM to set %i of %i" %(i_set+1, len(Parent_Dirs)))

        ## Get the name of the mask
        if Mask_Flag == True:
            Mask_Name = Mask_Names.iloc[i_set]

        ## Try reading the FV
        if os.path.isfile("%s%s" %(ReadPath, FV_Name)):
        
            if mode == "full":
                FV_data = pd.read_pickle("%s%s" %(ReadPath, FV_Name))

            elif mode == "test":
                FV_data = pd.read_csv("%s%s" %(ReadPath, FV_Name), index_col=0)

        else:
            print("There is no precalculated FV set named\n%s\nsaved at\n%s\n. Creating it now." %(FV_Name, ReadPath))
            ## Calculate the FV if it does not exist
            ## Pass a small database containing the path and data cube to the cfv function
            if "Mask name" in data_pd.columns and CalcParameters["MASK"] == True:
                header = ["Parent dir", "Cube name", "FV name", "Mask name"]
                data = [[Parent_Dir, Cube_Name, FV_Name, Mask_Name]]

            ## And add mask information if necessary
            else:
                header = ["Parent dir", "Cube name", "FV name"]
                data = [[Parent_Dir, Cube_Name, FV_Name]]

            data_set = pd.DataFrame(data=data, columns=header, index=None)

            ## Calculate the FV
            FV_data = cfv(CalcParameters, data_set=data_set)


        ## Reading the data
        FV_data = FV_data.to_numpy()

        ## Splitting dataset into three subsets
        if mode == "full":
            X_pos = FV_data[:, :3]
            y = FV_data[:,4]
            FV = FV_data[:,5:]

        elif mode == "test":
            y = FV_data[:,0]
            FV = FV_data[:,1:]

        ## Apply the standard scaler
        FV = sc.transform(FV)

        ## Buffer it to recalculate the original values later on
        sc_means = sc.mean_
        sc_stds = sc.scale_
            
        ## Get the number of features
        nf = len(FV[0])

        ## Generate display feature header
        if i_set == 0 and CalcParameters["CHECK_FV"] == True:            
            ## Initial bin size
            dbx = 0.001

            ## Generate header
            header = list()

            ## O --> Outflow voxel, N --> Non outflow voxel
            ## M --> Mask, P --> Prediction, T --> True prediction, F --> False prediction
            for i in range(nf):
                header.append("Feature %i POC" %(i+1))
                header.append("Feature %i PNC" %(i+1))

                if Mask_Flag == True:
                    header.append("Feature %i MOC" %(i+1))
                    header.append("Feature %i MNC" %(i+1))
                    header.append("Feature %i TOC" %(i+1))
                    header.append("Feature %i TNC" %(i+1))
                    header.append("Feature %i FOC" %(i+1))
                    header.append("Feature %i FNC" %(i+1))

            ## Get the minimal/maximal range value
            vmin = np.floor(CalcParameters["pedo"].min()/dbx)*dbx
            vmax = np.floor(CalcParameters["peup"].max()/dbx)*dbx

            ## Generate empty database that will be enlarged automatically
            index = np.arange(vmin, vmax+dbx, dbx)
            data = np.zeros([len(index),len(header)], dtype=int)
            df_count = pd.DataFrame(data=data, index=index, columns=header)

        ## Predicting non-invalid FV
        ti_pred = ti.time()
        FV_pred = Model_Predict(svm, FV)
        time_predict = ti.time() - ti_pred
        print("Predicting the FV took %.2f s" %(time_predict))
        
        ## Calculating scores if a mask is given
        if Mask_Flag == True:

            ## Calculate the scores via the confusion matrix
            tn, fp, fn, tp = confusion_matrix(y, FV_pred).ravel()
            score_n = tn / (tn+fp) * 100
            score_o = tp / (tp+fn) * 100
            score_c = (tn+tp) / (tp+tn+fp+fn) * 100
            score_b = (score_o + score_n) / 2

            print("Scoring results:")
            print("Score outflow voxel %.3f%% (misclassification %.3f%%)" %(score_o, 100-score_o))
            print("Score non-outflow voxel %.3f%% (misclassification %.3f%%)" %(score_n, 100-score_n))
            print("Score full cube %.3f%% (misclassification %.3f%%)" %(score_c, 100-score_c))
            print("Balanced score full cube %.3f%% (misclassification %.3f%%)" %(score_b, 100-score_b))

        ## Else set score to nan
        else:
            score_o = None
            score_n = None
            score_c = None
            score_b = None

        ## Saving scores to data base
        data_pd.loc[df_index, "Outflow score"] = score_o
        data_pd.loc[df_index, "Non-outflow score"] = score_n
        data_pd.loc[df_index, "Overall score"] = score_c
        data_pd.loc[df_index, "Balanced score"] = score_b

        ## Save the updated database
        if mode == "full" and Mask_Flag ==  True:
            data_pd.to_csv("%s%s" %(CalcParameters["DatePath"], CalcParameters["database_short"]))
    
        elif mode == "test" and Mask_Flag ==  True:
            data_pd.to_csv("%sTest-Mode_%s" %(CalcParameters["DatePath"], CalcParameters["database_short"]))

        if Mask_Flag == True:
            print(data_pd[["Outflow score", "Non-outflow score", "Overall score", "Balanced score"]])
        
        ## Construct an array based on the prediction
        ## Find all outflow voxel
        if mode == "full":
            of_pos_fv = np.where(FV_pred==1)

            ## Create an array out of them
            of_array_p = np.zeros(np.array([X_pos.T[0].max()+1, X_pos.T[1].max()+1, 
                                            X_pos.T[2].max()+1], dtype=int))
            of_array_p[(np.array(X_pos[of_pos_fv].T[0], dtype=int),np.array(X_pos[of_pos_fv].T[1], dtype=int),
                        np.array(X_pos[of_pos_fv].T[2], dtype=int))] = 1
            
            ## Construct an array based on the mask, same as previous
            if CalcParameters["MASK"] == True:
                of_pos_y = np.where(y==1)
                of_array_m = np.zeros(np.array([X_pos.T[0].max()+1, X_pos.T[1].max()+1, 
                                                X_pos.T[2].max()+1], dtype=int))
                of_array_m[(np.array(X_pos[of_pos_y].T[0], dtype=int),np.array(X_pos[of_pos_y].T[1], dtype=int),
                            np.array(X_pos[of_pos_y].T[2], dtype=int))] = 1
                
    
            ## Save the arrays as fits cubes
            if CalcParameters['create_fits'] == True:

                print("Creating fits")
            
                ## Reading out the header from the FV-cube
                cubepath = pd.read_csv("%s%s" %(CalcParameters["DatePath"], CalcParameters["database_file"]), index_col=0)["Parent dir"].iloc[i_set] # + "MapFit/"
                cubename = pd.read_csv("%s%s" %(CalcParameters["DatePath"], CalcParameters["database_file"]), index_col=0)["Cube name"].iloc[i_set]
                header = fits.getheader(cubepath+cubename, ext=0)

                ## Copy the original data cube to the cube directory
                shutil.copy(cubepath+cubename, CalcParameters["CubesPath"]+cubename)

                ## Change header type and unit
                header['BTYPE'] = "Mask - Prediction"
                header['BUNIT'] = "None"

                ## Save predictions and the smoothed cube
                outfile_predict = Cube_Name[:-4] + "_predict.fits"

                hdu = fits.PrimaryHDU(np.array(of_array_p, dtype=int), header=header)
                hdu.writeto(CalcParameters["CubesPath"]+outfile_predict, overwrite=True)
                
                ## Save the mask arrays
                if CalcParameters["MASK"] == True:

                    ## Change header type and unit
                    header['BTYPE'] = "Mask - Original"

                    outfile_mask = Cube_Name[:-4]+"_mask.fits"
                
                    hdu = fits.PrimaryHDU(np.array(of_array_m, dtype=int), header=header)
                    hdu.writeto(CalcParameters["CubesPath"]+outfile_mask, overwrite=True)
            
            
            if CalcParameters["FV_CUBES"] == True:

                ## Reading out the header from the FV-cube
                cubepath = pd.read_csv("%s%s" %(CalcParameters["DatePath"], CalcParameters["database_file"]), index_col=0)["Parent dir"].iloc[i_set]
                cubename = pd.read_csv("%s%s" %(CalcParameters["DatePath"], CalcParameters["database_file"]), index_col=0)["Cube name"].iloc[i_set]
                header = fits.getheader(cubepath+cubename, ext=0)

                ## Change header unit
                header['BUNIT'] = "None"

                ## Save the nth feature of the FV as fits
                for i in range(len(FV.T)):

                    ## Change header type
                    header['BTYPE'] = "Feature %i [standarized]" %(i+1)
   
                    ## Fill feature vector array
                    FV_array = np.zeros_like(of_array_p)
            
                    for ii, pos in enumerate(np.array(X_pos, dtype=int)):
                        vii = FV.T[i][ii]
                        FV_array[pos[0], pos[1], pos[2]] = vii

                    FV_array_co = np.ma.masked_equal(FV_array, 0) 
                    FV_array_co = np.ma.filled(FV_array_co.astype(np.float32), np.nan)

                    ## Define name of cubes
                    FV_array_file_co = Cube_Name+"_complete_feature-%i.fits" %(i+1)

                    ## Save cubes                
                    hdu = fits.PrimaryHDU(FV_array_co, header=header)
                    hdu.writeto(CalcParameters["CubesPath"]+FV_array_file_co, overwrite=True)
                    
        ## Fill feature count database and plot the distributions both in 1d and 2d
        if CalcParameters["CHECK_FV"] == True:

            print("Calculating and Ploting Feature Distribution.")

            ## Split the dataframe into outflow and non outflow voxel
            poci = np.where(FV_pred==1)
            pnci = np.where(FV_pred==0)

            ## ... and other classes if a mask is present
            if Mask_Flag == True:
                moci = np.where(y==1)
                mnci = np.where(y==0)
                toci = np.where(np.logical_and(y==1, FV_pred==1))
                tnci = np.where(np.logical_and(y==0, FV_pred==0))
                foci = np.where(np.logical_and(y==0, FV_pred==1))
                fnci = np.where(np.logical_and(y==1, FV_pred==0))

            ## Generate bins with fixed length
            bins = np.append(df_count.index, df_count.index[-1]+dbx)

            ## Get the length of the plot
            px = int(np.ceil(np.sqrt(nf)))
            py = int(np.ceil(nf/px))

            ## Initate the figure to plot all predicted features
            if CalcParameters["ThesisMode"] == False:
                fig, axs = plt.subplots(px, py, figsize=(4*px,4*py))
                fig_kde, axs_kde = plt.subplots(px, py, figsize=(4*px,4*py))
                fig_kde_log, axs_kde_log = plt.subplots(px, py, figsize=(4*px,4*py))
            else:
                fig, axs = plt.subplots(px, py, figsize=(7.47, 7.47))
                fig_kde, axs_kde = plt.subplots(px, py, figsize=(7.47, 7.47))
                fig_kde_log, axs_kde_log = plt.subplots(px, py, figsize=(7.47, 7.47))
                    
            fig.suptitle("Feature Distribution of set %i" %(i_set))
            fig_kde.suptitle("Feature Distribution of set %i using KDE" %(i_set))
            fig_kde_log.suptitle("Feature Distribution of set %i using KDE" %(i_set))

            ## Initate the figure to plot all predicted features
            if Mask_Flag == True:
                if CalcParameters["ThesisMode"] == False:
                    fig_tf, axs_tf = plt.subplots(px, py, figsize=(4*px,4*py))
                    fig_tf_kde, axs_tf_kde = plt.subplots(px, py, figsize=(4*px,4*py))
                    fig_tf_kde_log, axs_tf_kde_log = plt.subplots(px, py, figsize=(4*px,4*py))

                else:
                    fig_tf, axs_tf = plt.subplots(px, py, figsize=(7.47, 7.47))
                    fig_tf_kde, axs_tf_kde = plt.subplots(px, py, figsize=(7.47, 7.47))
                    fig_tf_kde_log, axs_tf_kde_log = plt.subplots(px, py, figsize=(7.47, 7.47))

                fig_tf.suptitle("Feature Distribution of set %i" %(i_set))
                fig_tf_kde.suptitle("Feature Distribution of set %i unsing KDE" %(i_set))
                fig_tf_kde_log.suptitle("Feature Distribution of set %i unsing KDE" %(i_set))


            ## Bin all features
            for i in range(nf):

                ## Read out the features
                xpoc = FV[poci, i]
                xpnc = FV[pnci, i]

                if Mask_Flag == True:
                    xmoc = FV[moci, i]
                    xmnc = FV[mnci, i]
                    xtoc = FV[toci, i]
                    xtnc = FV[tnci, i]
                    xfoc = FV[foci, i]
                    xfnc = FV[fnci, i]

                df_count_temp = pd.DataFrame(index = df_count.index)

                ## Bin the data
                df_count_temp["POC"] = np.histogram(xpoc, bins=bins)[0]
                df_count_temp["PNC"] = np.histogram(xpnc, bins=bins)[0]

                if Mask_Flag == True:
                    df_count_temp["MOC"] = np.histogram(xmoc, bins=bins)[0]
                    df_count_temp["MNC"] = np.histogram(xmnc, bins=bins)[0]
                    df_count_temp["TOC"] = np.histogram(xtoc, bins=bins)[0]
                    df_count_temp["TNC"] = np.histogram(xtnc, bins=bins)[0]
                    df_count_temp["FOC"] = np.histogram(xfoc, bins=bins)[0]
                    df_count_temp["FNC"] = np.histogram(xfnc, bins=bins)[0]

                ## Add new bins to database
                df_count["Feature %i POC" %(i+1)] += df_count_temp["POC"]
                df_count["Feature %i PNC" %(i+1)] += df_count_temp["PNC"]

                if Mask_Flag == True:
                    df_count["Feature %i MOC" %(i+1)] += df_count_temp["MOC"]
                    df_count["Feature %i MNC" %(i+1)] += df_count_temp["MNC"]
                    df_count["Feature %i TOC" %(i+1)] += df_count_temp["TOC"]
                    df_count["Feature %i TNC" %(i+1)] += df_count_temp["TNC"]
                    df_count["Feature %i FOC" %(i+1)] += df_count_temp["FOC"]
                    df_count["Feature %i FNC" %(i+1)] += df_count_temp["FNC"]


                ## Get the plot range
                vmin = CalcParameters["pedo"][i]
                vmax = CalcParameters["peup"][i] + dbx

                vmini = np.where(df_count_temp.index <= vmin)[0][-1]
                vmaxi = np.where(df_count_temp.index <= vmax)[0][-1]

                ## Extract the relevant data ranges
                x1_temp = df_count_temp["POC"].iloc[vmini:vmaxi]
                x2_temp = df_count_temp["PNC"].iloc[vmini:vmaxi]

                if Mask_Flag == True:
                    x3_temp = df_count_temp["MOC"].iloc[vmini:vmaxi]
                    x4_temp = df_count_temp["MNC"].iloc[vmini:vmaxi]
                    x5_temp = df_count_temp["TOC"].iloc[vmini:vmaxi]
                    x6_temp = df_count_temp["TNC"].iloc[vmini:vmaxi]
                    x7_temp = df_count_temp["FOC"].iloc[vmini:vmaxi]
                    x8_temp = df_count_temp["FNC"].iloc[vmini:vmaxi]

                ## Set up x-variable and rescale it
                xvals = np.linspace(vmin-10*dbx, vmax+10*dbx, 1000, True)
                bw = (xvals.max() - xvals.min()) / 100

                ## Fit a KDE to the data
                ## If the temp array isn't empty, fit gaussian kde to the data; else return zero array
                if not np.all(x1_temp == 0):
                    x1kde = x1_temp.loc[~(x1_temp==0)]
                    try:
                        kdex1 = KernelDensity(kernel="gaussian", bandwidth=abs(bw)).fit(X=np.array(x1kde.index).reshape(-1,1), sample_weight=x1kde)
                    except:
                        kdex1 = KernelDensity(kernel="gaussian", bandwidth=abs(2*dbx)).fit(X=np.array(x1kde.index).reshape(-1,1), sample_weight=x1kde)
                    dex1 = np.exp(kdex1.score_samples(xvals.reshape(-1,1)))
                else:
                    dex1 = np.zeros_like(xvals)

                if not np.all(x2_temp == 0):
                    x2kde = x2_temp.loc[~(x2_temp==0)]
                    try:
                        kdex2 = KernelDensity(kernel="gaussian", bandwidth=abs(bw)).fit(X=np.array(x2kde.index).reshape(-1,1), sample_weight=x2kde)
                    except:
                        kdex2 = KernelDensity(kernel="gaussian", bandwidth=abs(2*dbx)).fit(X=np.array(x2kde.index).reshape(-1,1), sample_weight=x2kde)
                    dex2 = np.exp(kdex2.score_samples(xvals.reshape(-1,1)))
                else:
                    dex2 = np.zeros_like(xvals)

                if Mask_Flag == True:
                    if not np.all(x3_temp == 0):
                        x3kde = x3_temp.loc[~(x3_temp==0)]
                        try:
                            kdex3 = KernelDensity(kernel="gaussian", bandwidth=abs(bw)).fit(X=np.array(x3kde.index).reshape(-1,1), sample_weight=x3kde)
                        except:
                            kdex3 = KernelDensity(kernel="gaussian", bandwidth=abs(2*dbx)).fit(X=np.array(x3kde.index).reshape(-1,1), sample_weight=x3kde)
                        dex3 = np.exp(kdex3.score_samples(xvals.reshape(-1,1)))
                    else:
                        dex3 = np.zeros_like(xvals)

                    if not np.all(x4_temp == 0):
                        x4kde = x4_temp.loc[~(x4_temp==0)]
                        try:
                            kdex4 = KernelDensity(kernel="gaussian", bandwidth=abs(bw)).fit(X=np.array(x4kde.index).reshape(-1,1), sample_weight=x4kde)
                        except:
                            kdex4 = KernelDensity(kernel="gaussian", bandwidth=abs(2*dbx)).fit(X=np.array(x4kde.index).reshape(-1,1), sample_weight=x4kde)
                        dex4 = np.exp(kdex4.score_samples(xvals.reshape(-1,1)))
                    else:
                        dex4 = np.zeros_like(xvals)

                    if not np.all(x5_temp == 0):
                        x5kde = x5_temp.loc[~(x5_temp==0)]
                        try:
                            kdex5 = KernelDensity(kernel="gaussian", bandwidth=abs(bw)).fit(X=np.array(x5kde.index).reshape(-1,1), sample_weight=x5kde)
                        except:
                            kdex5 = KernelDensity(kernel="gaussian", bandwidth=abs(2*dbx)).fit(X=np.array(x5kde.index).reshape(-1,1), sample_weight=x5kde)
                        dex5 = np.exp(kdex5.score_samples(xvals.reshape(-1,1)))
                    else:
                        dex5 = np.zeros_like(xvals)

                    if not np.all(x6_temp == 0):
                        x6kde = x6_temp.loc[~(x6_temp==0)]
                        try:
                            kdex6 = KernelDensity(kernel="gaussian", bandwidth=abs(bw)).fit(X=np.array(x6kde.index).reshape(-1,1), sample_weight=x6kde)
                        except:
                            kdex6 = KernelDensity(kernel="gaussian", bandwidth=abs(2*dbx)).fit(X=np.array(x6kde.index).reshape(-1,1), sample_weight=x6kde)
                        dex6 = np.exp(kdex6.score_samples(xvals.reshape(-1,1)))
                    else:
                        dex6 = np.zeros_like(xvals)

                    if not np.all(x7_temp == 0):
                        x7kde = x7_temp.loc[~(x7_temp==0)]
                        try:
                            kdex7 = KernelDensity(kernel="gaussian", bandwidth=abs(bw)).fit(X=np.array(x7kde.index).reshape(-1,1), sample_weight=x7kde)
                        except:
                            kdex7 = KernelDensity(kernel="gaussian", bandwidth=abs(2*dbx)).fit(X=np.array(x7kde.index).reshape(-1,1), sample_weight=x7kde)
                        dex7 = np.exp(kdex7.score_samples(xvals.reshape(-1,1)))
                    else:
                        dex7 = np.zeros_like(xvals)

                    if not np.all(x8_temp == 0):
                        x8kde = x8_temp.loc[~(x8_temp==0)]
                        try:
                            kdex8 = KernelDensity(kernel="gaussian", bandwidth=abs(bw)).fit(X=np.array(x8kde.index).reshape(-1,1), sample_weight=x8kde)
                        except:
                            kdex8 = KernelDensity(kernel="gaussian", bandwidth=abs(2*dbx)).fit(X=np.array(x8kde.index).reshape(-1,1), sample_weight=x8kde)
                        dex8 = np.exp(kdex8.score_samples(xvals.reshape(-1,1)))
                    else:
                        dex8 = np.zeros_like(xvals)

                ## Rescale the x-positions to the original picture
                xvals = xvals*sc_stds[i]+sc_means[i]

                ## Get the relevant axis
                if nf == 1:
                    axs_kde_t = axs_kde
                    axs_kde_log_t = axs_kde_log
                    if Mask_Flag == True:
                        axs_tf_kde_t = axs_tf_kde
                        axs_tf_kde_log_t = axs_tf_kde_log

                elif py == 1:
                    axs_kde_t = axs_kde[i]
                    axs_kde_log_t = axs_kde_log[i]
                    if Mask_Flag == True:
                        axs_tf_kde_t = axs_tf_kde[i]
                        axs_tf_kde_log_t = axs_tf_kde_log[i]
                else:
                    axs_kde_t = axs_kde[i%px, int(np.floor(i/px))]
                    axs_kde_log_t = axs_kde_log[i%px, int(np.floor(i/px))]
                    if Mask_Flag == True:
                        axs_tf_kde_t = axs_tf_kde[i%px, int(np.floor(i/px))]
                        axs_tf_kde_log_t = axs_tf_kde_log[i%px, int(np.floor(i/px))]

                ## Plot the histograms and color the area below them
                axs_kde_t.plot(xvals, dex1*100/sc_stds[i], color="blue", linestyle="-", label="Pred OF")
                axs_kde_t.plot(xvals, dex2*100/sc_stds[i], color="red", linestyle="-", label="Pred NF")

                axs_kde_log_t.plot(xvals, dex1*100/sc_stds[i], color="blue", linestyle="-", label="Pred OF")
                axs_kde_log_t.plot(xvals, dex2*100/sc_stds[i], color="red", linestyle="-", label="Pred NF")

                if Mask_Flag == True:
                    axs_kde_t.plot(xvals, dex3*100/sc_stds[i], color="green", linestyle="-", label="Mask OF")
                    axs_kde_t.plot(xvals, dex4*100/sc_stds[i], color="yellow", linestyle="-", label="Mask NF")

                    axs_kde_log_t.plot(xvals, dex3*100/sc_stds[i], color="green", linestyle="-", label="Mask OF")
                    axs_kde_log_t.plot(xvals, dex4*100/sc_stds[i], color="yellow", linestyle="-", label="Mask NF")
                    
                    axs_tf_kde_t.plot(xvals, dex5*100/sc_stds[i], color="blue", linestyle="-", label="True OF")
                    axs_tf_kde_t.plot(xvals, dex6*100/sc_stds[i], color="red", linestyle="-", label="True NF")
                    axs_tf_kde_t.plot(xvals, dex7*100/sc_stds[i], color="green", linestyle="-", label="False OF")
                    axs_tf_kde_t.plot(xvals, dex8*100/sc_stds[i], color="yellow", linestyle="-", label="False NF")
                    
                    axs_tf_kde_log_t.plot(xvals, dex5*100/sc_stds[i], color="blue", linestyle="-", label="True OF")
                    axs_tf_kde_log_t.plot(xvals, dex6*100/sc_stds[i], color="red", linestyle="-", label="True NF")
                    axs_tf_kde_log_t.plot(xvals, dex7*100/sc_stds[i], color="green", linestyle="-", label="False OF")
                    axs_tf_kde_log_t.plot(xvals, dex8*100/sc_stds[i], color="yellow", linestyle="-", label="False NF")
                    

                ## Set the axis ranges, names, title and legend
                axs_kde_std = axs_kde_t.twiny()
                ax1, ax2 = axs_kde_t.get_xlim()
                axs_kde_std.set_xlim((ax1-sc_means[i])/sc_stds[i], (ax2-sc_means[i])/sc_stds[i])
                axs_kde_std.set_xlabel("Feature value (standarized)")

                axs_kde_t.set_xlabel("Feature value (original)")
                axs_kde_t.set_ylabel("Distribution in %")
                axs_kde_t.set_title("Feature %i" %(i+1))
                axs_kde_t.legend()
                axs_kde_t.grid()

                ## Twin the y axis and label it
                axs_kde_log_std = axs_kde_log_t.twiny()
                ax1, ax2 = axs_kde_log_t.get_xlim()
                axs_kde_log_std.set_xlim((ax1-sc_means[i])/sc_stds[i], (ax2-sc_means[i])/sc_stds[i])
                axs_kde_log_std.set_xlabel("Feature value (standarized)")

                ## Set the x/y labels and other plot parameters
                axs_kde_log_t.set_xlabel("Feature value (original)")
                axs_kde_log_t.set_ylabel("Distribution in %")
                axs_kde_log_t.set_title("Feature %i" %(i+1))
                axs_kde_log_t.legend()
                axs_kde_log_t.grid()
                axs_kde_log_t.set_yscale("log")
                if Mask_Flag != True:
                    axs_kde_log_t.set_ylim(bottom=1e-5, top=10**np.ceil(np.log10(max(dex1.max(), dex2.max()))))
                elif Mask_Flag == True:
                    axs_kde_log_t.set_ylim(bottom=1e-5, top=10**np.ceil(np.log10(max(dex1.max(), dex2.max(), dex3.max(), dex4.max())*100)))

                ## Repeat for mask data
                if Mask_Flag == True:
                    axs_tf_kde_std = axs_tf_kde_t.twiny()
                    ax1, ax2 = axs_tf_kde_t.get_xlim()
                    axs_tf_kde_std.set_xlim((ax1-sc_means[i])/sc_stds[i], (ax2-sc_means[i])/sc_stds[i])
                    axs_tf_kde_std.set_xlabel("Feature value (standarized)")

                    axs_tf_kde_t.set_xlabel("Feature value (original)")
                    axs_tf_kde_t.set_ylabel("Distribution in %")
                    axs_tf_kde_t.set_title("Feature %i" %(i+1))
                    axs_tf_kde_t.legend()
                    axs_tf_kde_t.grid()


                    axs_tf_kde_log_std = axs_tf_kde_log_t.twiny()
                    ax1, ax2 = axs_tf_kde_log_t.get_xlim()
                    axs_tf_kde_log_std.set_xlim((ax1-sc_means[i])/sc_stds[i], (ax2-sc_means[i])/sc_stds[i])
                    axs_tf_kde_log_std.set_xlabel("Feature value (standarized)")

                    axs_tf_kde_log_t.set_xlabel("Feature value (original)")
                    axs_tf_kde_log_t.set_ylabel("Distribution in %")
                    axs_tf_kde_log_t.set_title("Feature %i" %(i+1))
                    axs_tf_kde_log_t.legend()
                    axs_tf_kde_log_t.grid()
                    axs_tf_kde_log_t.set_yscale("log")
                    axs_tf_kde_log_t.set_ylim(bottom=1e-5, top=10**np.ceil(np.log10(max(dex5.max(), dex6.max(), dex7.max(), dex8.max())*100)))


            ## Remove empty subplots
            for i in range(nf, px*py):
                fig_kde.delaxes(axs_kde[-1, int(np.floor(i/px))])
                fig_kde_log.delaxes(axs_kde_log[-1, int(np.floor(i/px))])

                if Mask_Flag == True:
                    fig_tf_kde.delaxes(axs_tf_kde[-1, int(np.floor(i/px))])
                    fig_tf_kde_log.delaxes(axs_tf_kde_log[-1, int(np.floor(i/px))])

            fig_kde.tight_layout()
            fig_kde_log.tight_layout()
            
            if Mask_Flag == True:
                fig_tf_kde.tight_layout()
                fig_tf_kde_log.tight_layout()

            ## Save figs
            if mode == "full":
                fig_kde.savefig("%sFeature_Distribution_Predict_kde_set-%i.pdf" %(CalcParameters['PlotPath'], i_set), dpi='figure', format="pdf",
                            metadata=None, bbox_inches="tight", pad_inches=0.1, backend=None)

                fig_kde_log.savefig("%sFeature_Distribution_Predict_kde_log_set-%i.pdf" %(CalcParameters['PlotPath'], i_set), dpi='figure', format="pdf",
                            metadata=None, bbox_inches="tight", pad_inches=0.1, backend=None)

                if Mask_Flag == True:
                    fig_tf_kde.savefig("%sFeature_Distribution_TrueFalse_kde_set-%i.pdf" %(CalcParameters['PlotPath'], i_set), dpi='figure', format="pdf",
                                metadata=None, bbox_inches="tight", pad_inches=0.1, backend=None)
                    fig_tf_kde_log.savefig("%sFeature_Distribution_TrueFalse_kde_log_set-%i.pdf" %(CalcParameters['PlotPath'], i_set), dpi='figure', format="pdf",
                                metadata=None, bbox_inches="tight", pad_inches=0.1, backend=None)

            elif mode == "test":
                fig_kde.savefig("%sTest-Mode_Feature_Distribution_Predict_kde_set-%i.pdf" %(CalcParameters['PlotPath'], i_set), dpi='figure', format="pdf",
                            metadata=None, bbox_inches="tight", pad_inches=0.1, backend=None)
                
                fig_kde_log.savefig("%sTest-Mode_Feature_Distribution_Predict_kde_log_set-%i.pdf" %(CalcParameters['PlotPath'], i_set), dpi='figure', format="pdf",
                            metadata=None, bbox_inches="tight", pad_inches=0.1, backend=None)

                if Mask_Flag == True:
                    fig_tf_kde.savefig("%sTest-Mode_Feature_Distribution_TrueFalse_kde_set-%i.pdf" %(CalcParameters['PlotPath'], i_set), dpi='figure', format="pdf",
                                metadata=None, bbox_inches="tight", pad_inches=0.1, backend=None)
                    fig_tf_kde_log.savefig("%sTest-Mode_Feature_Distribution_TrueFalse_kde_log_set-%i.pdf" %(CalcParameters['PlotPath'], i_set), dpi='figure', format="pdf",
                                metadata=None, bbox_inches="tight", pad_inches=0.1, backend=None)

            plt.close("all")

        ## Plot the 2D distribution
        if CalcParameters["CHECK_FV2D"] == True:

            time_grid = ti.time()

            ## Pixel in x/y direction (for plotting)
            xn = 101
            yn = 101
                
            ## Prepare empty arrays to buffer in resutls and number of used data points
            if i_set == 0:
                for i in range(1,nf+1):
                    for ii in range(1,nf+1):
                        if i == ii:
                            CalcParameters["FD2D_pred_F%i_o" %(i)] = np.zeros(shape=(xn))
                            if Mask_Flag == True:
                                CalcParameters["FD2D_mask_F%i_o" %(i)] = np.zeros(shape=(xn))
                                CalcParameters["FD2D_false_F%i_o" %(i)] = np.zeros(shape=(xn))
                                
                            CalcParameters["FD2D_pred_F%i_n" %(i)] = np.zeros(shape=(xn))
                            if Mask_Flag == True:
                                CalcParameters["FD2D_mask_F%i_n" %(i)] = np.zeros(shape=(xn))
                                CalcParameters["FD2D_false_F%i_n" %(i)] = np.zeros(shape=(xn))

                        else:
                            CalcParameters["FD2D_pred_F%ivsF%i" %(i, ii)] = np.zeros(shape=(xn, yn))
                            if Mask_Flag == True:
                                CalcParameters["FD2D_mask_F%ivsF%i" %(i, ii)] = np.zeros(shape=(xn, yn))
                                CalcParameters["FD2D_false_F%ivsF%i" %(i, ii)] = np.zeros(shape=(xn, yn))

                CalcParameters["FD2S_dp"] = 0

            ## Reduce the number of data points
            distr_lim = int(1e5)
    
            ## If the trainings set is larger, shorten it to distr_lim to avoid long calculation times
            if len(y) > distr_lim:

                distr_in = np.random.choice(range(len(FV)), distr_lim)

                FV_distr = FV[distr_in]
                pred_distr = FV_pred[distr_in]
                y_distr = y[distr_in]
        
            ## Else take the whole data set
            else:

                FV_distr = FV
                pred_distr = FV_pred
                y_distr = y
                    
            ## Generate overview figures
            if CalcParameters["ThesisMode"] == False:
                fig_o_p_kde, ax_o_p_kde = plt.subplots(nf, nf, figsize=(4*nf+2,4*nf))
                fig_o_p_kde_1, ax_o_p_kde_1 = plt.subplots(figsize=(8,8))

                if nf >= 2:
                    fig_o_p_kde_1v2, ax_o_p_kde_1v2 = plt.subplots(figsize=(8,8))
                    fig_o_p_kde_2v1, ax_o_p_kde_2v1 = plt.subplots(figsize=(8,8))
            else:
                fig_o_p_kde, ax_o_p_kde = plt.subplots(nf, nf, figsize=(7.47, 7.47))
                fig_o_p_kde_1, ax_o_p_kde_1 = plt.subplots(figsize=(7.47,7.47))

                if nf >= 2:
                    fig_o_p_kde_1v2, ax_o_p_kde_1v2 = plt.subplots(figsize=(7.47,7.47))
                    fig_o_p_kde_2v1, ax_o_p_kde_2v1 = plt.subplots(figsize=(7.47,7.47))

            ## Generate additional images if mask is true
            if Mask_Flag == True:
                if CalcParameters["ThesisMode"] == False:
                    fig_o_m_kde, ax_o_m_kde = plt.subplots(nf, nf, figsize=(4*nf+2,4*nf))
                    fig_o_f_kde, ax_o_f_kde = plt.subplots(nf, nf, figsize=(4*nf+2,4*nf))
                    fig_o_m_kde_1, ax_o_m_kde_1 = plt.subplots(figsize=(8,8))
                    fig_o_f_kde_1, ax_o_f_kde_1 = plt.subplots(figsize=(8,8))

                    if nf >= 2:
                        fig_o_m_kde_1v2, ax_o_m_kde_1v2 = plt.subplots(figsize=(8,8))
                        fig_o_m_kde_2v1, ax_o_m_kde_2v1 = plt.subplots(figsize=(8,8))
                        fig_o_f_kde_1v2, ax_o_f_kde_1v2 = plt.subplots(figsize=(8,8))
                        fig_o_f_kde_2v1, ax_o_f_kde_2v1 = plt.subplots(figsize=(8,8))
                else:
                    fig_o_m_kde, ax_o_m_kde = plt.subplots(nf, nf, figsize=(7.47, 7.47))
                    fig_o_f_kde, ax_o_f_kde = plt.subplots(nf, nf, figsize=(7.47, 7.47))
                    fig_o_m_kde_1, ax_o_m_kde_1 = plt.subplots(figsize=(7.47,7.47))
                    fig_o_f_kde_1, ax_o_f_kde_1 = plt.subplots(figsize=(7.47,7.47))

                    if nf >= 2:
                        fig_o_m_kde_1v2, ax_o_m_kde_1v2 = plt.subplots(figsize=(7.47,7.47))
                        fig_o_m_kde_2v1, ax_o_m_kde_2v1 = plt.subplots(figsize=(7.47,7.47))
                        fig_o_f_kde_1v2, ax_o_f_kde_1v2 = plt.subplots(figsize=(7.47,7.47))
                        fig_o_f_kde_2v1, ax_o_f_kde_2v1 = plt.subplots(figsize=(7.47,7.47))

            ## Iterate over all feature combinations and extract the relevant data
            for i in range(1,nf+1):

                xpnci = FV_distr[pred_distr==0].T[i-1]
                xpoci = FV_distr[pred_distr==1].T[i-1]
    
                if Mask_Flag == True:
                    xmnci = FV_distr[y_distr==0].T[i-1]
                    xmoci = FV_distr[y_distr==1].T[i-1]
                    xfnci = FV_distr[np.logical_and(pred_distr==1, y_distr==0)].T[i-1]
                    xfpci = FV_distr[np.logical_and(pred_distr==0, y_distr==1)].T[i-1]

                for ii in range(1,nf+1):
                    xpncii = FV_distr[pred_distr==0].T[ii-1]
                    xpocii = FV_distr[pred_distr==1].T[ii-1]

                    if Mask_Flag == True:
                        xmncii = FV_distr[y_distr==0].T[ii-1]
                        xmocii = FV_distr[y_distr==1].T[ii-1]
                        xfncii = FV_distr[np.logical_and(pred_distr==1, y_distr==0)].T[ii-1]
                        xfpcii = FV_distr[np.logical_and(pred_distr==0, y_distr==1)].T[ii-1]

                    if i == ii:
                        print("Calculate/plot SVM grid %i" %(i), end="\r")
                    else:
                        print("Calculate/plot SVM grid %i vs %i" %(i, ii), end="\r")

                    ## Extend in Fi and Fii direction
                    xmi = CalcParameters["pedo"][i-1]
                    xma = CalcParameters["peup"][i-1]
                    ymi = CalcParameters["pedo"][ii-1]
                    yma = CalcParameters["peup"][ii-1]

                    ## xg/yg grid
                    xg = np.linspace(xmi, xma, xn)
                    yg = np.linspace(ymi, yma, yn)

                    xm, ym = np.meshgrid(xg, yg, indexing="ij")

                    mg = np.array([xm.reshape(-1), ym.reshape(-1)]).T
                       
                    ## Set diagonal entries (1d profiles)
                    if i == ii:
                        ## Do a kernel estimate, if it fails set fit to 0
                        bw = (xma - xmi) / min(50, xn/1)

                        if np.shape(xpoci) != (0,) and np.shape(xpoci) != (1,0):

                            try:
                                kdexppo = KernelDensity(kernel="gaussian", bandwidth=abs(bw)).fit(X=xpoci.reshape(-1,1))
                                dexppo = np.exp(kdexppo.score_samples(xg.reshape(-1,1))) * 100

                            except:
                                print("KDE failed.")
                                print(np.shape(xpoci))
                                dexppo = np.zeros_like(xg)

                        else:
                            dexppo = np.zeros_like(xg)

                        if np.shape(xpnci) != (0,) and np.shape(xpnci) != (1,0):

                            try:
                                kdexppn = KernelDensity(kernel="gaussian", bandwidth=abs(bw)).fit(X=xpnci.reshape(-1,1))
                                dexppn = np.exp(kdexppn.score_samples(xg.reshape(-1,1))) * 100                                    

                            except:
                                print("KDE failed.")
                                print(np.shape(xpnci))
                                dexppn = np.zeros_like(xg)
                        else:
                            dexppn = np.zeros_like(xg)


                        if Mask_Flag == True:

                            if np.shape(xmoci) != (0,) and np.shape(xmoci) != (1,0):

                                try:
                                    kdexpmo = KernelDensity(kernel="gaussian", bandwidth=abs(bw)).fit(X=xmoci.reshape(-1,1))
                                    dexpmo = np.exp(kdexpmo.score_samples(xg.reshape(-1,1))) * 100

                                except:
                                    print("KDE failed.")
                                    print(np.shape(xmoci))
                                    dexpmo = np.zeros_like(xg)
                            else:
                                dexpmo = np.zeros_like(xg)

                                        
                            if np.shape(xmnci) != (0,) and np.shape(xmnci) != (1,0):

                                try:
                                    kdexpmn = KernelDensity(kernel="gaussian", bandwidth=abs(bw)).fit(X=xmnci.reshape(-1,1))
                                    dexpmn = np.exp(kdexpmn.score_samples(xg.reshape(-1,1))) * 100

                                except:
                                    print("KDE failed.")
                                    print(np.shape(xmnci))
                                    dexpmn = np.zeros_like(xg)
                            else:
                                dexpmn = np.zeros_like(xg)

                                        
                            if np.shape(xfpci) != (0,) and np.shape(xfpci) != (1,0):

                                try:
                                    kdexpfo = KernelDensity(kernel="gaussian", bandwidth=abs(bw)).fit(X=xfpci.reshape(-1,1))
                                    dexpfo = np.exp(kdexpfo.score_samples(xg.reshape(-1,1))) * 100

                                except:
                                    print("KDE failed.")
                                    print(np.shape(xfpci))
                                    dexpfo = np.zeros_like(xg)

                            else:
                                dexpfo = np.zeros_like(xg)

                                        
                            if np.shape(xfnci) != (0,) and np.shape(xfnci) != (1,0):

                                try:
                                    kdexpfn = KernelDensity(kernel="gaussian", bandwidth=abs(bw)).fit(X=xfnci.reshape(-1,1))
                                    dexpfn = np.exp(kdexpfn.score_samples(xg.reshape(-1,1))) * 100


                                except:
                                    print("KDE failed.")
                                    print(np.shape(xfnci))
                                    dexpfn = np.zeros_like(xg)

                            else:
                                dexpfn = np.zeros_like(xg)


                        ## Plot data
                        ax_o_p_kde[ii-1,i-1].plot(xg, dexppo, label="Outflow", color="red")
                        ax_o_p_kde[ii-1,i-1].plot(xg, dexppn, label="Non-Outflow", color="blue")
                        ax_o_p_kde[ii-1,i-1].legend()

                        if i == 0:
                            ax_o_p_kde_1.plot(xg, dexppo, label="Outflow", color="red")
                            ax_o_p_kde_1.plot(xg, dexppn, label="Non-Outflow", color="blue")
                            ax_o_p_kde_1.legend()

                        if Mask_Flag == True:
                            if nf != 1:
                                ax_o_m_kde[ii-1,i-1].plot(xg, dexpmo, label="Outflow", color="red")
                                ax_o_m_kde[ii-1,i-1].plot(xg, dexpmn, label="Non-Outflow", color="blue")
                                ax_o_m_kde[ii-1,i-1].legend()
                                ax_o_f_kde[ii-1,i-1].plot(xg, dexpfo, label="FP", color="red")
                                ax_o_f_kde[ii-1,i-1].plot(xg, dexpfn, label="FN", color="blue")
                                ax_o_f_kde[ii-1,i-1].legend()

                            if i == 0:
                                    ax_o_m_kde_1.plot(xg, dexpmo, label="Outflow", color="red")
                                    ax_o_m_kde_1.plot(xg, dexpmn, label="Non-Outflow", color="blue")
                                    ax_o_m_kde_1.legend()
                                    ax_o_f_kde_1.plot(xg, dexpfo, label="FP", color="red")
                                    ax_o_f_kde_1.plot(xg, dexpfn, label="FN", color="blue")
                                    ax_o_f_kde_1.legend()

                    ## Calculate off diagonal entries (2d plots)
                    elif nf >= 2:

                        ## Set fits and colours depending on the position of i and ii
                        if i >= ii:
                            set_fit_p = np.append(xpoci, xpocii).reshape(2,-1).T
                            if Mask_Flag == True:
                                set_fit_m = np.append(xmoci, xmocii).reshape(2,-1).T
                                set_fit_f = np.append(xfpci, xfpcii).reshape(2,-1).T

                            color="red"
                            color2d="Reds"

                        else:
                            set_fit_p = np.append(xpnci, xpncii).reshape(2,-1).T
                            if Mask_Flag == True:
                                set_fit_m = np.append(xmnci, xmncii).reshape(2,-1).T
                                set_fit_f = np.append(xfnci, xfncii).reshape(2,-1).T

                            color="blue"
                            color2d="Blues"

                        ## Kernel density estimate
                        bw = max((xma - xmi) / min(50, xn/1), (yma - ymi) / min(50, yn/1))

                        ## Do a kernel estimate if the array is not empty
                        if np.shape(set_fit_p) != (0,) and np.shape(set_fit_p) != (1,0):

                            try:
                                kdexpo = KernelDensity(kernel="gaussian", bandwidth=abs(bw)).fit(X=set_fit_p)
                                dexpo = (np.exp(kdexpo.score_samples(mg)).reshape(xn, yn) * 100).T

                            except:
                                print("KDE failed.")
                                print(np.shape(set_fit_p))
                                dexpo = np.zeros((xn, yn))
                        else:
                            dexpo = np.zeros((xn, yn))


                        if Mask_Flag == True:
                            if np.shape(set_fit_m) != (0,) and np.shape(set_fit_m) != (1,0):

                                try:
                                    kdexpm = KernelDensity(kernel="gaussian", bandwidth=abs(bw)).fit(X=set_fit_m)
                                    dexpm = (np.exp(kdexpm.score_samples(mg)).reshape(xn, yn) * 100).T

                                except:
                                    print("KDE failed.")
                                    print(np.shape(set_fit_m))
                                    dexpm = np.zeros_like((xn, yn))
                            else:
                                dexpm = np.zeros_like((xn, yn))

                                    
                            if np.shape(set_fit_f) != (0,) and np.shape(set_fit_f) != (1,0):

                                try:
                                    kdexpf = KernelDensity(kernel="gaussian", bandwidth=abs(bw)).fit(X=set_fit_f)
                                    dexpf = (np.exp(kdexpf.score_samples(mg)).reshape(xn, yn) * 100).T

                                except:
                                    print("KDE failed.")
                                    print(np.shape(set_fit_f))
                                    dexpf = np.zeros_like((xn, yn))

                            else:
                                dexpf = np.zeros_like((xn, yn))

                        ## Get plot ranges
                        fxmin = xg[0] - (xg[1]-xg[0])/2
                        fxmax = xg[-1] + (xg[1]-xg[0])/2
                        fymin = yg[0] - (yg[1]-yg[0])/2
                        fymax = yg[-1] + (yg[1]-yg[0])/2

                        ## Plot data
                        im_o_p_kde = ax_o_p_kde[ii-1,i-1].imshow(dexpo, aspect="auto", origin="lower", cmap=color2d, extent=[fxmin, fxmax, fymin, fymax])

                        ### Add colorbar     
                        im_ratio = dexpo.shape[0]/dexpo.data.shape[1]
                        cbar_o_p_kde = plt.colorbar(im_o_p_kde, fraction=0.047*im_ratio)

                        ## Plot demo images
                        if i == 0 and ii == 1:
                            ax_o_p_kde_1v2.imshow(dexpo, aspect="auto", origin="lower", cmap=color2d, extent=[fxmin, fxmax, fymin, fymax])
                            cbar_o_p_kde_1v2 = plt.colorbar(ax_o_p_kde_1v2, fraction=0.047*im_ratio)

                        if i == 1 and ii == 0:
                            ax_o_p_kde_2v1.imshow(dexpo, aspect="auto", origin="lower", cmap=color2d, extent=[fxmin, fxmax, fymin, fymax])
                            cbar_o_p_kde_2v1 = plt.colorbar(ax_o_p_kde_2v1, fraction=0.047*im_ratio)
            
                        if Mask_Flag == True:
                            im_o_m_kde = ax_o_m_kde[ii-1,i-1].imshow(dexpm, aspect="auto", origin="lower", cmap=color2d, extent=[fxmin, fxmax, fymin, fymax])

                            ### Add colorbar     
                            cbar_o_m_kde = plt.colorbar(im_o_m_kde, fraction=0.047*im_ratio)

                            im_o_f_kde = ax_o_f_kde[ii-1,i-1].imshow(dexpf, aspect="auto", origin="lower", cmap=color2d, extent=[fxmin, fxmax, fymin, fymax])
                                
                            ### Add colorbar     
                            cbar_o_f_kde = plt.colorbar(im_o_f_kde, fraction=0.047*im_ratio)
                                
                            if i == 0 and ii == 1:
                                ax_o_m_kde_1v2.imshow(dexpm, aspect="auto", origin="lower", cmap=color2d, extent=[fxmin, fxmax, fymin, fymax])

                                ### Add colorbar     
                                cbar_o_m_kde_1v2 = plt.colorbar(ax_o_m_kde_1v2, fraction=0.047*im_ratio)

                                ax_o_f_kde_1v2.imshow(dexpf, aspect="auto", origin="lower", cmap=color2d, extent=[fxmin, fxmax, fymin, fymax])
                                
                                ### Add colorbar     
                                cbar_o_f_kde_1v2 = plt.colorbar(ax_o_f_kde_1v2, fraction=0.047*im_ratio)
                                    
                            if i == 1 and ii == 0:
                                ax_o_m_kde_2v1.imshow(dexpm, aspect="auto", origin="lower", cmap=color2d, extent=[fxmin, fxmax, fymin, fymax])

                                ### Add colorbar     
                                cbar_o_m_kde_2v1 = plt.colorbar(ax_o_m_kde_2v1, fraction=0.047*im_ratio)

                                ax_o_f_kde_2v1.imshow(dexpf, aspect="auto", origin="lower", cmap=color2d, extent=[fxmin, fxmax, fymin, fymax])
                                
                                ### Add colorbar     
                                cbar_o_f_kde_2v1 = plt.colorbar(ax_o_f_kde_2v1, fraction=0.047*im_ratio)


                        ## In case of two features only compute 50% contour
                        if nf == 2:

                            ## Contour levels in %
                            levels = (50)

                            ## Calculate area containing xx % of the integrated data --> contour value
                            if dexpo.sum() >= 0:
                                
                                dexpos = sorted(dexpo.reshape(-1)/dexpo.sum())

                                plevels = ()

                                for level in levels:
                                    ind = np.where(np.cumsum(dexpos) <= level/100)[0][-1]
                                    plevels = np.append(plevels, dexpos[ind]*dexpo.sum())

                            else:
                                plevels = (-1,0,1)

                            ## Plot contours for i and ii and for the mirror entry of ii and i
                            ax_o_p_kde[ii-1,i-1].contour(dexpo, levels=plevels, origin="lower", colors=color, extent=[fxmin, fxmax, fymin, fymax])
                            cbar_o_p_kde.set_label("Classification")
                            
                            ax_o_p_kde[i-1,ii-1].contour(dexpo.T, levels=plevels, origin="lower", colors=color, extent=[fymin, fymax, fxmin, fxmax])
                                
                            if i == 0 and ii == 1:
                                ax_o_p_kde_1v2.contour(dexpo, levels=plevels, origin="lower", colors=color, extent=[fxmin, fxmax, fymin, fymax])
                                cbar_o_p_kde_1v2.set_label("Classification")
                            
                                ax_o_p_kde_2v1.contour(dexpo.T, levels=plevels, origin="lower", colors=color, extent=[fymin, fymax, fxmin, fxmax])
                                    
                            if i == 1 and ii == 0:
                                ax_o_p_kde_2v1.contour(dexpo, levels=plevels, origin="lower", colors=color, extent=[fxmin, fxmax, fymin, fymax])
                                cbar_o_p_kde_2v1.set_label("Classification")
                            
                                ax_o_p_kde_1v2.contour(dexpo.T, levels=plevels, origin="lower", colors=color, extent=[fymin, fymax, fxmin, fxmax])

                            if Mask_Flag == True:
    
                                if dexpm.sum() >= 0:
                                    dexpms = sorted(dexpm.reshape(-1)/dexpm.sum())

                                    mlevels = ()

                                    for level in levels:
                                        ind = np.where(np.cumsum(dexpms) <= level/100)[0][-1]
                                        mlevels = np.append(mlevels, dexpms[ind]*dexpm.sum())
                                else:
                                    mlevels = (-1,0,1)

                                if dexpf.sum() >= 0:
                                    dexpfs = sorted(dexpf.reshape(-1)/dexpf.sum())

                                    flevels = ()

                                    for level in levels:
                                        ind = np.where(np.cumsum(dexpfs) <= level/100)[0][-1]
                                        flevels = np.append(flevels, dexpfs[ind]*dexpf.sum())
                                else:
                                    flevels = (-1,0,1)

                                ## Plot more contours
                                ax_o_m_kde[ii-1,i-1].contour(dexpm, levels=mlevels, origin="lower", colors=color, extent=[fxmin, fxmax, fymin, fymax])
                                cbar_o_m_kde.set_label("Classification")
                            
                                ax_o_m_kde[i-1,ii-1].contour(dexpm.T, levels=mlevels, origin="lower", colors=color, extent=[fymin, fymax, fxmin, fxmax])

                                ax_o_f_kde[ii-1,i-1].contour(dexpf, levels=flevels, origin="lower", colors=color, extent=[fxmin, fxmax, fymin, fymax])
                                cbar_o_f_kde.set_label("Classification")
                            
                                ax_o_f_kde[i-1,ii-1].contour(dexpf.T, levels=flevels, origin="lower", colors=color, extent=[fymin, fymax, fxmin, fxmax])
                                    
                                if i == 0 and ii == 1:
                                    ax_o_m_kde_1v2.contour(dexpm, levels=mlevels, origin="lower", colors=color, extent=[fxmin, fxmax, fymin, fymax])
                                    cbar_o_m_kde_1v2.set_label("Classification")
                            
                                    ax_o_m_kde_2v1.contour(dexpm.T, levels=mlevels, origin="lower", colors=color, extent=[fymin, fymax, fxmin, fxmax])

                                    ax_o_f_kde_1v2.contour(dexpf, levels=flevels, origin="lower", colors=color, extent=[fxmin, fxmax, fymin, fymax])
                                    cbar_o_f_kde_1v2.set_label("Classification")
                            
                                    ax_o_f_kde_2v1.contour(dexpf.T, levels=flevels, origin="lower", colors=color, extent=[fymin, fymax, fxmin, fxmax])
                                                                   
                                if i == 1 and ii == 0:
                                    ax_o_m_kde_2v1.contour(dexpm, levels=mlevels, origin="lower", colors=color, extent=[fxmin, fxmax, fymin, fymax])
                                    cbar_o_m_kde_2v1.set_label("Classification")
                            
                                    ax_o_m_kde_1v2.contour(dexpm.T, levels=mlevels, origin="lower", colors=color, extent=[fymin, fymax, fxmin, fxmax])

                                    ax_o_f_kde_2v1.contour(dexpf, levels=flevels, origin="lower", colors=color, extent=[fxmin, fxmax, fymin, fymax])
                                    cbar_o_f_kde_2v1.set_label("Classification")
                            
                                    ax_o_f_kde_1v2.contour(dexpf.T, levels=flevels, origin="lower", colors=color, extent=[fymin, fymax, fxmin, fxmax])

                        ## In case of features calculate more contours
                        if nf >= 3:
                                
                            ## Contour levels in %
                            levels = ( 5, 25, 50, 75, 95)
                                
                            ## Determine contour values
                            if dexpo.sum() >= 0:
                                dexpos = sorted(dexpo.reshape(-1)/dexpo.sum())

                                plevels = ()

                                for level in levels:
                                    ind = np.where(np.cumsum(dexpos) <= level/100)[0][-1]
                                    plevels = np.append(plevels, dexpos[ind]*dexpo.sum())
                                                                
                                plevels = list(dict.fromkeys(plevels))

                            else:
                                plevels = (-1,0,1)

                            ## Plot the contours
                            ax_o_p_kde[ii-1,i-1].contour(dexpo, levels=plevels, origin="lower", colors=color, extent=[fxmin, fxmax, fymin, fymax])
                            cbar_o_p_kde.set_label("Probability distribution in %") 
                            
                            ax_o_p_kde[i-1,ii-1].contour(dexpo.T, levels=plevels, origin="lower", colors=color, extent=[fymin, fymax, fxmin, fxmax])

                            if i == 0 and ii == 1:
                                ax_o_p_kde_1v2.contour(dexpo, levels=plevels, origin="lower", colors=color, extent=[fxmin, fxmax, fymin, fymax])
                                cbar_o_p_kde_1v2.set_label("Probability distribution in %")
                            
                                ax_o_p_kde_2v1.contour(dexpo.T, levels=plevels, origin="lower", colors=color, extent=[fymin, fymax, fxmin, fxmax])
                                    
                            if i == 1 and ii == 0:
                                ax_o_p_kde_2v1.contour(dexpo, levels=plevels, origin="lower", colors=color, extent=[fxmin, fxmax, fymin, fymax])
                                cbar_o_p_kde_2v1.set_label("Probability distribution in %")
                            
                                ax_o_p_kde_1v2.contour(dexpo.T, levels=plevels, origin="lower", colors=color, extent=[fymin, fymax, fxmin, fxmax])

                            if Mask_Flag == True:
    
                                if dexpm.sum() >= 0:
                                    dexpms = sorted(dexpm.reshape(-1)/dexpm.sum())

                                    mlevels = ()

                                    for level in levels:
                                        ind = np.where(np.cumsum(dexpms) <= level/100)[0][-1]
                                        mlevels = np.append(mlevels, dexpms[ind]*dexpm.sum())
                                    
                                    mlevels = list(dict.fromkeys(mlevels))

                                else:
                                    mlevels = (-1,0,1)
                                        
                                if dexpm.sum() >= 0:
                                    dexpfs = sorted(dexpf.reshape(-1)/dexpf.sum())

                                    flevels = ()

                                    for level in levels:
                                        ind = np.where(np.cumsum(dexpfs) <= level/100)[0][-1]
                                        flevels = np.append(flevels, dexpfs[ind]*dexpf.sum())
                                    
                                    flevels = list(dict.fromkeys(flevels))

                                else:
                                    flevels = (-1,0,1)

                                ## Plot more contours
                                ax_o_m_kde[ii-1,i-1].contour(dexpm, levels=mlevels, origin="lower", colors=color, extent=[fxmin, fxmax, fymin, fymax])
                                cbar_o_p_kde.set_label("Probability distribution in %")
                            
                                ax_o_m_kde[i-1,ii-1].contour(dexpm.T, levels=mlevels, origin="lower", colors=color, extent=[fymin, fymax, fxmin, fxmax])

                                ax_o_f_kde[ii-1,i-1].contour(dexpf, levels=flevels, origin="lower", colors=color, extent=[fxmin, fxmax, fymin, fymax])
                                cbar_o_f_kde.set_label("Probability distribution in %")
                            
                                ax_o_f_kde[i-1,ii-1].contour(dexpf.T, levels=flevels, origin="lower", colors=color, extent=[fymin, fymax, fxmin, fxmax])
                                    
                                if i == 0 and ii == 1:
                                    ax_o_m_kde_1v2.contour(dexpm, levels=mlevels, origin="lower", colors=color, extent=[fxmin, fxmax, fymin, fymax])
                                    cbar_o_p_kde_1v2.set_label("Probability distribution in %")
                            
                                    ax_o_m_kde_2v1.contour(dexpm.T, levels=mlevels, origin="lower", colors=color, extent=[fymin, fymax, fxmin, fxmax])

                                    ax_o_f_kde_1v2.contour(dexpf, levels=flevels, origin="lower", colors=color, extent=[fxmin, fxmax, fymin, fymax])
                                    cbar_o_f_kde_1v2.set_label("Probability distribution in %")
                            
                                    ax_o_f_kde_2v1.contour(dexpf.T, levels=flevels, origin="lower", colors=color, extent=[fymin, fymax, fxmin, fxmax])
                                        
                                if i == 1 and ii == 0:
                                    ax_o_m_kde_2v1.contour(dexpm, levels=mlevels, origin="lower", colors=color, extent=[fxmin, fxmax, fymin, fymax])
                                    cbar_o_p_kde_2v1.set_label("Probability distribution in %")
                            
                                    ax_o_m_kde_1v2.contour(dexpm.T, levels=mlevels, origin="lower", colors=color, extent=[fymin, fymax, fxmin, fxmax])

                                    ax_o_f_kde_2v1.contour(dexpf, levels=flevels, origin="lower", colors=color, extent=[fxmin, fxmax, fymin, fymax])
                                    cbar_o_f_kde_2v1.set_label("Probability distribution in %")
                            
                                    ax_o_f_kde_1v2.contour(dexpf.T, levels=flevels, origin="lower", colors=color, extent=[fymin, fymax, fxmin, fxmax])


                    ## Adjust the x and y_distr axis                    
                    if nf == 1:
                        ax_o_p_kde.set_xlabel(r"Feature %i [Standarized]" %(ii))
                        ax_o_p_kde.set_ylabel(r"Classification")
                            
                        if i == 0:
                            ax_o_p_kde_1.set_xlabel(r"Feature %i [Standarized]" %(ii))
                            ax_o_p_kde_1.set_ylabel(r"Classification")


                        if Mask_Flag == True:
                            ax_o_m_kde.set_xlabel(r"Feature %i [Standarized]" %(ii))
                            ax_o_m_kde.set_ylabel(r"Classification")
                            ax_o_f_kde.set_xlabel(r"Feature %i [Standarized]" %(ii))
                            ax_o_f_kde.set_ylabel(r"Classification")
                                
                            if i == 0:
                                ax_o_m_kde_1.set_xlabel(r"Feature %i [Standarized]" %(ii))
                                ax_o_m_kde_1.set_ylabel(r"Classification")
                                ax_o_f_kde_1.set_xlabel(r"Feature %i [Standarized]" %(ii))
                                ax_o_f_kde_1.set_ylabel(r"Classification")

                    elif i == ii:
                        ax_o_p_kde[ii-1,i-1].set_xlabel(r"Feature %i [Standarized]" %(ii))
                        ax_o_p_kde[ii-1,i-1].set_ylabel(r"Probability distribution in %")

                        if i == 0:
                            ax_o_p_kde_1.set_xlabel(r"Feature %i [Standarized]" %(ii))
                            ax_o_p_kde_1.set_ylabel(r"Probability distribution in %")

                        if Mask_Flag == True:
                            ax_o_m_kde[ii-1,i-1].set_xlabel(r"Feature %i [Standarized]" %(ii))
                            ax_o_m_kde[ii-1,i-1].set_ylabel(r"Probability distribution in %")
                            ax_o_f_kde[ii-1,i-1].set_xlabel(r"Feature %i [Standarized]" %(ii))
                            ax_o_f_kde[ii-1,i-1].set_ylabel(r"Probability distribution in %")
                                
                            if i == 0:
                                ax_o_m_kde_1.set_xlabel(r"Feature %i [Standarized]" %(ii))
                                ax_o_m_kde_1.set_ylabel(r"Probability distribution in %")
                                ax_o_f_kde_1.set_xlabel(r"Feature %i [Standarized]" %(ii))
                                ax_o_f_kde_1.set_ylabel(r"Probability distribution in %")
                        
                    else:
                        ax_o_p_kde[ii-1,i-1].set_xlabel(r"Feature %i [Standarized]" %(i))
                        ax_o_p_kde[ii-1,i-1].set_ylabel(r"Feature %i [Standarized]" %(ii))

                        if i == 0 and ii == 1:
                            ax_o_p_kde_1v2.set_xlabel(r"Feature %i [Standarized]" %(i))
                            ax_o_p_kde_1v2.set_ylabel(r"Feature %i [Standarized]" %(ii))
                                
                        if i == 1 and ii == 0:
                            ax_o_p_kde_2v1.set_xlabel(r"Feature %i [Standarized]" %(i))
                            ax_o_p_kde_2v1.set_ylabel(r"Feature %i [Standarized]" %(ii))

                        if Mask_Flag == True:
                            ax_o_m_kde[ii-1,i-1].set_xlabel(r"Feature %i [Standarized]" %(i))
                            ax_o_m_kde[ii-1,i-1].set_ylabel(r"Feature %i [Standarized]" %(ii))

                            ax_o_f_kde[ii-1,i-1].set_xlabel(r"Feature %i [Standarized]" %(i))
                            ax_o_f_kde[ii-1,i-1].set_ylabel(r"Feature %i [Standarized]" %(ii))

                            if i == 0 and ii == 1:
                                ax_o_m_kde_1v2.set_xlabel(r"Feature %i [Standarized]" %(i))
                                ax_o_m_kde_1v2.set_ylabel(r"Feature %i [Standarized]" %(ii))

                                ax_o_f_kde_1v2.set_xlabel(r"Feature %i [Standarized]" %(i))
                                ax_o_f_kde_1v2.set_ylabel(r"Feature %i [Standarized]" %(ii))
                                    
                            if i == 1 and ii == 0:
                                ax_o_m_kde_1v2.set_xlabel(r"Feature %i [Standarized]" %(i))
                                ax_o_m_kde_2v1.set_ylabel(r"Feature %i [Standarized]" %(ii))

                                ax_o_f_kde_2v1.set_xlabel(r"Feature %i [Standarized]" %(i))
                                ax_o_f_kde_2v1.set_ylabel(r"Feature %i [Standarized]" %(ii))


                    ## Set grids
                    if nf == 1:
                        ax_o_p_kde.grid()

                        if Mask_Flag == True:
                            ax_o_m_kde.grid()
                            ax_o_f_kde.grid()

                    else:
                        ax_o_p_kde[ii-1,i-1].grid()

                        if Mask_Flag == True:
                            ax_o_m_kde[ii-1,i-1].grid()
                            ax_o_f_kde[ii-1,i-1].grid()

                    if i == 0 and ii == 0:
                        ax_o_p_kde_1.grid()

                        if Mask_Flag == True:
                            ax_o_m_kde_1.grid()
                            ax_o_f_kde_1.grid()
                            
                    elif i == 0 and ii == 1:
                        ax_o_p_kde_1v2.grid()

                        if Mask_Flag == True:
                            ax_o_m_kde_1v2.grid()
                            ax_o_f_kde_1v2.grid()
                            
                    if i == 1 and ii == 0:
                        ax_o_p_kde_2v1.grid()

                        if Mask_Flag == True:
                            ax_o_m_kde_2v1.grid()
                            ax_o_f_kde_2v1.grid()
                            
                    ## Buffer data for the upcoming overview image
                    if i == ii:
                        CalcParameters["FD2D_pred_F%i_o" %(i)] += dexppo * len(y_distr)
                        if Mask_Flag == True:
                            CalcParameters["FD2D_mask_F%i_o" %(i)] += dexpmo * len(y_distr)
                            CalcParameters["FD2D_false_F%i_o" %(i)] += dexpfo * len(y_distr)
                                
                        CalcParameters["FD2D_pred_F%i_n" %(i)] += dexppn * len(y_distr)

                        if Mask_Flag == True:
                            CalcParameters["FD2D_mask_F%i_n" %(i)] += dexpmn * len(y_distr)
                            CalcParameters["FD2D_false_F%i_n" %(i)] += dexpfn * len(y_distr)

                    else:
                        CalcParameters["FD2D_pred_F%ivsF%i" %(i, ii)] += dexpo * len(y_distr)
                        if Mask_Flag == True:
                            CalcParameters["FD2D_mask_F%ivsF%i" %(i, ii)] += dexpm * len(y_distr)
                            CalcParameters["FD2D_false_F%ivsF%i" %(i, ii)] += dexpf * len(y_distr)


            CalcParameters["FD2S_dp"] += len(y_distr)

            ## Save image
            fig_o_p_kde.tight_layout()
            fig_o_p_kde.savefig("%sFeatureDistributionGrid_pred_kde_Overview_set-%i.pdf" %(CalcParameters['PlotPath'], i_set), dpi='figure', format="pdf",
                        metadata=None, bbox_inches="tight", pad_inches=0.1, backend=None)
            fig_o_p_kde_1.savefig("%sFeatureDistributionGrid_pred_kde_Overview_1v1_set-%i.pdf" %(CalcParameters['PlotPath'], i_set), dpi='figure', format="pdf",
                        metadata=None, bbox_inches="tight", pad_inches=0.1, backend=None)
            if nf >= 2:
                fig_o_p_kde_1v2.savefig("%sFeatureDistributionGrid_pred_kde_Overview_1v2_set-%i.pdf" %(CalcParameters['PlotPath'], i_set), dpi='figure', format="pdf",
                            metadata=None, bbox_inches="tight", pad_inches=0.1, backend=None)
                fig_o_p_kde_2v1.savefig("%sFeatureDistributionGrid_pred_kde_Overview_2v1_set-%i.pdf" %(CalcParameters['PlotPath'], i_set), dpi='figure', format="pdf",
                            metadata=None, bbox_inches="tight", pad_inches=0.1, backend=None)

            if Mask_Flag == True:
                fig_o_m_kde.tight_layout()
                fig_o_m_kde.savefig("%sFeatureDistributionGrid_mask_kde_Overview_set-%i.pdf" %(CalcParameters['PlotPath'], i_set), dpi='figure', format="pdf",
                            metadata=None, bbox_inches="tight", pad_inches=0.1, backend=None)
                fig_o_m_kde_1.savefig("%sFeatureDistributionGrid_mask_kde_Overview_1v1_set-%i.pdf" %(CalcParameters['PlotPath'], i_set), dpi='figure', format="pdf",
                            metadata=None, bbox_inches="tight", pad_inches=0.1, backend=None)
                    
                if nf >= 2:
                    fig_o_m_kde_1v2.savefig("%sFeatureDistributionGrid_mask_kde_Overview_1v2_set-%i.pdf" %(CalcParameters['PlotPath'], i_set), dpi='figure', format="pdf",
                                    metadata=None, bbox_inches="tight", pad_inches=0.1, backend=None)
                    fig_o_m_kde_2v1.savefig("%sFeatureDistributionGrid_mask_kde_Overview_2v1_set-%i.pdf" %(CalcParameters['PlotPath'], i_set), dpi='figure', format="pdf",
                                    metadata=None, bbox_inches="tight", pad_inches=0.1, backend=None)

                fig_o_f_kde.tight_layout()
                fig_o_f_kde.savefig("%sFeatureDistributionGrid_false_kde_Overview_set-%i.pdf" %(CalcParameters['PlotPath'], i_set), dpi='figure', format="pdf",
                            metadata=None, bbox_inches="tight", pad_inches=0.1, backend=None)

                fig_o_f_kde_1.savefig("%sFeatureDistributionGrid_false_kde_Overview_1v1_set-%i.pdf" %(CalcParameters['PlotPath'], i_set), dpi='figure', format="pdf",
                            metadata=None, bbox_inches="tight", pad_inches=0.1, backend=None)
                    
                if nf >= 2:
                    fig_o_f_kde_1v2.savefig("%sFeatureDistributionGrid_false_kde_Overview_1v2_set-%i.pdf" %(CalcParameters['PlotPath'], i_set), dpi='figure', format="pdf",
                            metadata=None, bbox_inches="tight", pad_inches=0.1, backend=None)
                    fig_o_f_kde_2v1.savefig("%sFeatureDistributionGrid_false_kde_Overview_2v1_set-%i.pdf" %(CalcParameters['PlotPath'], i_set), dpi='figure', format="pdf",
                            metadata=None, bbox_inches="tight", pad_inches=0.1, backend=None)

            plt.close("all")


            print("Calculating the SVM Grid took %.2f s." %(ti.time() - time_grid))


        ## Calculate and plot the confusion matrix
        if CalcParameters["Plot_CM"] == True and CalcParameters["MASK"] == True:

            print("Calculating and Ploting Confusion Matrix.")


            ## Calculate the confusion matrix in absolute numbers
            cm_temp = confusion_matrix(y, FV_pred)

            ## Define Class Names
            class_label = ["Non-OF Voxel", "OF Voxel"]

            ## Plot the confusion matrix
            if mode == "full":
                Plot_Confusion_Matrix(cm_temp, ThesisMode=CalcParameters["ThesisMode"], class_label=class_label, ppath=CalcParameters["PlotPath"], pname="Confusion_Matrix_set-%i.pdf" %(i_set))
                
            elif mode == "test":
                Plot_Confusion_Matrix(cm_temp, ThesisMode=CalcParameters["ThesisMode"], class_label=class_label, ppath=CalcParameters["PlotPath"], pname="Test-Mode_Confusion_Matrix_set-%i.pdf" %(i_set))

            ## Initiate empty 
            if i_set == 0:
                cm_comp = np.zeros_like(cm_temp, dtype=int)
                cm_comp_p = np.zeros_like(cm_temp, dtype=float)

            ## Update complete confusion matrix
            cm_comp += cm_temp
            cm_comp_p += cm_temp/cm_temp.sum()

        ## Plot Feature Importance if a mask is given
        if CalcParameters["CHECK_FI"] and Mask_Flag == True:

            print("Calculating and Ploting Feature Permutation Improtance.")

            tis = ti.time()

            ## Define Feature Names
            feature_names = np.array([f"Feature {i}" for i in range(1,nf+1)])
            ## Max number of permutation points
            param_lim = 1e3

            ## Read save file
            if mode == "full" and i_set == 0:
                data_pd_fi = pd.read_csv("%s%s" %(CalcParameters["DatePath"], CalcParameters["database_short"]), index_col=0)

            elif mode == "test" and i_set == 0:
                data_pd_fi = pd.read_csv("%sReduced_Test_DataBase.csv" %(CalcParameters["TempPath"]), index_col=0)

            ## Get the scoring function
            if CalcParameters["fi_scoring"] in ["accuracy", "balanced_accuracy"]:
                scoring_function = CalcParameters["fi_scoring"]
            elif CalcParameters["fi_scoring"] in ["TPR", "tpr"]:
                scoring_function = TPR
            elif CalcParameters["fi_scoring"] in ["TNR", "tnr"]:
                scoring_function = TNR 
            elif CalcParameters["fi_scoring"] in ["TPPNR", "tppnr"]:
                scoring_function = TPPNR
            elif CalcParameters["fi_scoring"] in ["TPNNR", "tpnnr"]:
                scoring_function = TPNNR 
                
            ## Generate empty buffer array
            result_importances = np.zeros((nf,0))

            ## Execute the FPI n times to analyze more data and get a more general picture
            for i in range(10):

                ## If the trainings set is larger, shorten it to param_lim to avoid long calculation times
                if len(FV) > param_lim:

                    FV_param, _, y_param, _ = train_test_split(
                        FV, y, train_size=int(param_lim), random_state=1, stratify=y)


                ## Else take the whole data set
                else:
        
                    FV_param = FV
                    y_param = y
                    

                ## Calculate the permutation importance and sort it
                result = permutation_importance(svm, FV_param, y_param, n_repeats=10, scoring=scoring_function, random_state=0, n_jobs=1)

                ## Buffer restults
                result_importances = np.append(result_importances, result.importances, axis=1)

            sorted_idx = np.mean(result_importances, axis=1).argsort()
            sorted_idx_rev = sorted_idx.argsort()

            ## Plot the results as a violin plot with score=0 highlighted and set labels
            if CalcParameters["ThesisMode"] == False:
                fig, ax = plt.subplots(figsize=(8,8))
            else:
                fig, ax = plt.subplots(figsize=(7.47, 7.47))

            ax.axvline(x=0, color="gray", linestyle="--", lw=2)
            vp = ax.violinplot(result_importances[sorted_idx].T, vert=False, showmeans=True)
            ax.set_title("Permutation Importance of each Feature")
            ax.set_xlabel("Decrease in %s score" %(CalcParameters["fi_scoring"]))
            ax.set_yticks(range(1,nf+1))
            ax.set_yticklabels(feature_names[sorted_idx])

            fig.tight_layout()

            ## Extract violin ranges
            cmins = np.array([item.vertices[:,0] for item in vp["cmins"].get_paths()])[:,0][sorted_idx_rev]
            cmeans = np.array([item.vertices[:,0] for item in vp["cmeans"].get_paths()])[:,0][sorted_idx_rev]
            cmaxes = np.array([item.vertices[:,0] for item in vp["cmaxes"].get_paths()])[:,0][sorted_idx_rev]
            
            for feature_name, cmin, cmean, cmax in zip(feature_names, cmins, cmeans, cmaxes):
                        
                ## Saving violin mean and extenses to data base (I know, it ain't whiskers anymore, but it works...)
                data_pd_fi.loc[df_index, "%s" %(feature_name)] = cmean
                data_pd_fi.loc[df_index, "%s lower whiskers length" %(feature_name)] = abs(cmean-cmin)
                data_pd_fi.loc[df_index, "%s upper whiskers length" %(feature_name)] = abs(cmax-cmean)
                        
            ## Save images
            if mode == "full":
                fig.savefig("%sFeature_Permutation_Importance_scoring-%s_set-%i.pdf" %(CalcParameters['PlotPath'], CalcParameters["fi_scoring"], i_set), dpi='figure', format="pdf",
                            metadata=None, bbox_inches="tight", pad_inches=0.1, backend=None)

            elif mode == "test":
                fig.savefig("%sTest-Mode_Feature_Permutation_Importance_scoring-%s_set-%i.pdf" %(CalcParameters['PlotPath'], CalcParameters["fi_scoring"], i_set), dpi='figure', format="pdf",
                            metadata=None, bbox_inches="tight", pad_inches=0.1, backend=None)
                        
            plt.close("all")

            print("\nPermutation took %.3fs." %(ti.time()-tis))

            ## Saving the feature improtance database
            data_pd_fi.to_csv("%sFeatureImportance.csv" %(CalcParameters["DatePath"]))

        print("The iteration took %.3f s." %(ti.time()-tiis))


    ## Plot feature count database, not as well commented but the code is almost identical to the previous one for one cube
    if CalcParameters["CHECK_FV"] == True:

        ## Get the length of the plot
        px = int(np.ceil(np.sqrt(nf)))
        py = int(np.ceil(nf/px))

        ## Initate the figure to plot all predicted features
        if CalcParameters["ThesisMode"] == False:
            fig, axs = plt.subplots(px, py, figsize=(4*px,4*py))
            fig_kde, axs_kde = plt.subplots(px, py, figsize=(4*px,4*py))
            fig_kde_log, axs_kde_log = plt.subplots(px, py, figsize=(4*px,4*py))

        else:
            fig, axs = plt.subplots(px, py, figsize=(7.47, 7.47))
            fig_kde, axs_kde = plt.subplots(px, py, figsize=(7.47, 7.47))
            fig_kde_log, axs_kde_log = plt.subplots(px, py, figsize=(7.47, 7.47))

        fig.suptitle("Feature Distribution")
        fig_kde.suptitle("Feature Distribution using KDE")
        fig_kde_log.suptitle("Feature Distribution using KDE")

        ## Initate the figure to plot all predicted features
        if Mask_Flag == True:
            if CalcParameters["ThesisMode"] == False:
                fig_tf, axs_tf = plt.subplots(px, py, figsize=(4*px,4*py))
                fig_tf_kde, axs_tf_kde = plt.subplots(px, py, figsize=(4*px,4*py))
                fig_tf_kde_log, axs_tf_kde_log = plt.subplots(px, py, figsize=(4*px,4*py))
            
            else:
                fig_tf, axs_tf = plt.subplots(px, py, figsize=(4*px,4*py))
                fig_tf_kde, axs_tf_kde = plt.subplots(px, py, figsize=(4*px,4*py))
                fig_tf_kde_log, axs_tf_kde_log = plt.subplots(px, py, figsize=(4*px,4*py))
        
            fig_tf.suptitle("Feature Distribution")
            fig_tf_kde.suptitle("Feature Distribution using KDE")
            fig_tf_kde_log.suptitle("Feature Distribution using KDE")


        ## Generate 2D distribution overview figure, if there is more than just one figure/feature
        if nf != 1:
            if CalcParameters["ThesisMode"] == False:
                fig_o_kde, ax_o_kde = plt.subplots(nf, nf, figsize=(4*nf+2,4*nf))
            else:
                fig_o_kde, ax_o_kde = plt.subplots(nf, nf, figsize=(7.47, 7.47))

            fig_o_kde.suptitle("2D Feature Distribution using KDE")
            if Mask_Flag == True:
                if CalcParameters["ThesisMode"] == False:
                    fig_o_tf_kde, ax_o_tf_kde = plt.subplots(nf, nf, figsize=(4*nf+2,4*nf))
                else:
                    fig_o_tf_kde, ax_o_tf_kde = plt.subplots(nf, nf, figsize=(7.47, 7.47))

                fig_o_tf_kde.suptitle("2D Feature Distribution using KDE")

        ## Bin all features
        for i in range(nf):

            ## Get the plot range
            vmin = CalcParameters["pedo"][i]
            vmax = CalcParameters["peup"][i] + dbx

            vmini = np.where(df_count_temp.index <= vmin)[0][-1]
            vmaxi = np.where(df_count_temp.index <= vmax)[0][-1]

            x1_temp = df_count["Feature %i POC" %(i+1)].iloc[vmini:vmaxi]
            x2_temp = df_count["Feature %i PNC" %(i+1)].iloc[vmini:vmaxi]

            if Mask_Flag == True:
                x3_temp = df_count["Feature %i MOC" %(i+1)].iloc[vmini:vmaxi]
                x4_temp = df_count["Feature %i MNC" %(i+1)].iloc[vmini:vmaxi]
                x5_temp = df_count["Feature %i TOC" %(i+1)].iloc[vmini:vmaxi]
                x6_temp = df_count["Feature %i TNC" %(i+1)].iloc[vmini:vmaxi]
                x7_temp = df_count["Feature %i FOC" %(i+1)].iloc[vmini:vmaxi]
                x8_temp = df_count["Feature %i FNC" %(i+1)].iloc[vmini:vmaxi]

            ## Generate x-positions
            xvals = np.linspace(vmin-10*dbx, vmax+10*dbx, 1000, True)
            bw = (xvals.max() - xvals.min()) / 100

            ## If the temp array isn't empty, fit gaussian kde to the data; else return zero array
            if not np.all(x1_temp == 0):
                x1kde = x1_temp.loc[~(x1_temp==0)]
                try:
                    kdex1 = KernelDensity(kernel="gaussian", bandwidth=abs(bw)).fit(X=np.array(x1kde.index).reshape(-1,1), sample_weight=x1kde)
                except:
                    kdex1 = KernelDensity(kernel="gaussian", bandwidth=abs(2*dbx)).fit(X=np.array(x1kde.index).reshape(-1,1), sample_weight=x1kde)
                dex1 = np.exp(kdex1.score_samples(xvals.reshape(-1,1)))
            else:
                dex1 = np.zeros_like(xvals)

            if not np.all(x2_temp == 0):
                x2kde = x2_temp.loc[~(x2_temp==0)]
                try:
                    kdex2 = KernelDensity(kernel="gaussian", bandwidth=abs(bw)).fit(X=np.array(x2kde.index).reshape(-1,1), sample_weight=x2kde)
                except:
                    kdex2 = KernelDensity(kernel="gaussian", bandwidth=abs(2*dbx)).fit(X=np.array(x2kde.index).reshape(-1,1), sample_weight=x2kde)
                dex2 = np.exp(kdex2.score_samples(xvals.reshape(-1,1)))
            else:
                dex2 = np.zeros_like(xvals)

            if Mask_Flag == True:
                if not np.all(x3_temp == 0):
                    x3kde = x3_temp.loc[~(x3_temp==0)]
                    try:
                        kdex3 = KernelDensity(kernel="gaussian", bandwidth=abs(bw)).fit(X=np.array(x3kde.index).reshape(-1,1), sample_weight=x3kde)
                    except:
                        kdex3 = KernelDensity(kernel="gaussian", bandwidth=abs(2*dbx)).fit(X=np.array(x3kde.index).reshape(-1,1), sample_weight=x3kde)
                    dex3 = np.exp(kdex3.score_samples(xvals.reshape(-1,1)))
                else:
                    dex3 = np.zeros_like(xvals)

                if not np.all(x4_temp == 0):
                    x4kde = x4_temp.loc[~(x4_temp==0)]
                    try:
                        kdex4 = KernelDensity(kernel="gaussian", bandwidth=abs(bw)).fit(X=np.array(x4kde.index).reshape(-1,1), sample_weight=x4kde)
                    except:
                        kdex4 = KernelDensity(kernel="gaussian", bandwidth=abs(2*dbx)).fit(X=np.array(x4kde.index).reshape(-1,1), sample_weight=x4kde)
                    dex4 = np.exp(kdex4.score_samples(xvals.reshape(-1,1)))
                else:
                    dex4 = np.zeros_like(xvals)

                if not np.all(x5_temp == 0):
                    x5kde = x5_temp.loc[~(x5_temp==0)]
                    try:
                        kdex5 = KernelDensity(kernel="gaussian", bandwidth=abs(bw)).fit(X=np.array(x5kde.index).reshape(-1,1), sample_weight=x5kde)
                    except:
                        kdex5 = KernelDensity(kernel="gaussian", bandwidth=abs(2*dbx)).fit(X=np.array(x5kde.index).reshape(-1,1), sample_weight=x5kde)
                    dex5 = np.exp(kdex5.score_samples(xvals.reshape(-1,1)))
                else:
                    dex5 = np.zeros_like(xvals)

                if not np.all(x6_temp == 0):
                    x6kde = x6_temp.loc[~(x6_temp==0)]
                    try:
                        kdex6 = KernelDensity(kernel="gaussian", bandwidth=abs(bw)).fit(X=np.array(x6kde.index).reshape(-1,1), sample_weight=x6kde)
                    except:
                        kdex6 = KernelDensity(kernel="gaussian", bandwidth=abs(2*dbx)).fit(X=np.array(x6kde.index).reshape(-1,1), sample_weight=x6kde)
                    dex6 = np.exp(kdex6.score_samples(xvals.reshape(-1,1)))
                else:
                    dex6 = np.zeros_like(xvals)

                if not np.all(x7_temp == 0):
                    x7kde = x7_temp.loc[~(x7_temp==0)]
                    try:
                        kdex7 = KernelDensity(kernel="gaussian", bandwidth=abs(bw)).fit(X=np.array(x7kde.index).reshape(-1,1), sample_weight=x7kde)
                    except:
                        kdex7 = KernelDensity(kernel="gaussian", bandwidth=abs(2*dbx)).fit(X=np.array(x7kde.index).reshape(-1,1), sample_weight=x7kde)
                    dex7 = np.exp(kdex7.score_samples(xvals.reshape(-1,1)))
                else:
                    dex7 = np.zeros_like(xvals)

                if not np.all(x8_temp == 0):
                    x8kde = x8_temp.loc[~(x8_temp==0)]
                    try:
                        kdex8 = KernelDensity(kernel="gaussian", bandwidth=abs(bw)).fit(X=np.array(x8kde.index).reshape(-1,1), sample_weight=x8kde)
                    except:
                        kdex8 = KernelDensity(kernel="gaussian", bandwidth=abs(2*dbx)).fit(X=np.array(x8kde.index).reshape(-1,1), sample_weight=x8kde)
                    dex8 = np.exp(kdex8.score_samples(xvals.reshape(-1,1)))
                else:
                    dex8 = np.zeros_like(xvals)

            ## Rescale the x-positions to the original picture
            xvals = xvals*sc_stds[i]+sc_means[i]
                
            ## Get the relevant axis
            if nf == 1:
                axs_kde_t = axs_kde
                axs_kde_log_t = axs_kde_log
                if Mask_Flag == True:
                    axs_tf_kde_t = axs_tf_kde
                    axs_tf_kde_log_t = axs_tf_kde_log
            elif py == 1:
                axs_kde_t = axs_kde[i]
                axs_kde_log_t = axs_kde_log[i]
                if Mask_Flag == True:
                    axs_tf_kde_t = axs_tf_kde[i]
                    axs_tf_kde_log_t = axs_tf_kde_log[i]
            else:
                axs_kde_t = axs_kde[i%px, int(np.floor(i/px))]
                axs_kde_log_t = axs_kde_log[i%px, int(np.floor(i/px))]
                if Mask_Flag == True:
                    axs_tf_kde_t = axs_tf_kde[i%px, int(np.floor(i/px))]
                    axs_tf_kde_log_t = axs_tf_kde_log[i%px, int(np.floor(i/px))]

            ## Plot the histograms and color the area below them
            axs_kde_t.plot(xvals, dex1*100/sc_stds[i], color="blue", linestyle="-", label="Pred OF")
            axs_kde_t.plot(xvals, dex2*100/sc_stds[i], color="red", linestyle="-", label="Pred NF")

            axs_kde_log_t.plot(xvals, dex1*100/sc_stds[i], color="blue", linestyle="-", label="Pred OF")
            axs_kde_log_t.plot(xvals, dex2*100/sc_stds[i], color="red", linestyle="-", label="Pred NF")

            if Mask_Flag == True:
                axs_kde_t.plot(xvals, dex3*100/sc_stds[i], color="green", linestyle="-", label="Mask OF")
                axs_kde_t.plot(xvals, dex4*100/sc_stds[i], color="yellow", linestyle="-", label="Mask NF")

                axs_kde_log_t.plot(xvals, dex3*100/sc_stds[i], color="green", linestyle="-", label="Mask OF")
                axs_kde_log_t.plot(xvals, dex4*100/sc_stds[i], color="yellow", linestyle="-", label="Mask NF")

                axs_tf_kde_t.plot(xvals, dex5*100/sc_stds[i], color="blue", linestyle="-", label="True OF")
                axs_tf_kde_t.plot(xvals, dex6*100/sc_stds[i], color="red", linestyle="-", label="True NF")
                axs_tf_kde_t.plot(xvals, dex7*100/sc_stds[i], color="green", linestyle="-", label="False OF")
                axs_tf_kde_t.plot(xvals, dex8*100/sc_stds[i], color="yellow", linestyle="-", label="False NF")

                axs_tf_kde_log_t.plot(xvals, dex5*100/sc_stds[i], color="blue", linestyle="-", label="True OF")
                axs_tf_kde_log_t.plot(xvals, dex6*100/sc_stds[i], color="red", linestyle="-", label="True NF")
                axs_tf_kde_log_t.plot(xvals, dex7*100/sc_stds[i], color="green", linestyle="-", label="False OF")
                axs_tf_kde_log_t.plot(xvals, dex8*100/sc_stds[i], color="yellow", linestyle="-", label="False NF")

            ## Set the axis ranges, names, title and legend
            axs_kde_std = axs_kde_t.twiny()
            ax1, ax2 = axs_kde_t.get_xlim()
            axs_kde_std.set_xlim((ax1-sc_means[i])/sc_stds[i], (ax2-sc_means[i])/sc_stds[i])
            axs_kde_std.set_xlabel("Feature value (standarized)")

            axs_kde_t.set_xlabel("Feature value (original)")
            axs_kde_t.set_ylabel("Distribution in %")
            axs_kde_t.set_title("Feature %i" %(i+1))
            axs_kde_t.legend()
            axs_kde_t.grid()


            axs_kde_log_std = axs_kde_log_t.twiny()
            ax1, ax2 = axs_kde_log_t.get_xlim()
            axs_kde_log_std.set_xlim((ax1-sc_means[i])/sc_stds[i], (ax2-sc_means[i])/sc_stds[i])
            axs_kde_log_std.set_xlabel("Feature value (standarized)")

            axs_kde_log_t.set_xlabel("Feature value (original)")
            axs_kde_log_t.set_ylabel("Distribution in %")
            axs_kde_log_t.set_title("Feature %i" %(i+1))
            axs_kde_log_t.legend()
            axs_kde_log_t.grid()    
            axs_kde_log_t.set_yscale("log")
            if Mask_Flag != True:
                axs_kde_log_t.set_ylim(bottom=1e-5, top=10**np.ceil(np.log10(max(dex1.max(), dex2.max())*100)))
            elif Mask_Flag == True:
                axs_kde_log_t.set_ylim(bottom=1e-5, top=10**np.ceil(np.log10(max(dex1.max(), dex2.max(), dex3.max(), dex4.max())*100)))

            if Mask_Flag == True:
                axs_tf_kde_std = axs_tf_kde_t.twiny()
                ax1, ax2 = axs_tf_kde_t.get_xlim()
                axs_tf_kde_std.set_xlim((ax1-sc_means[i])/sc_stds[i], (ax2-sc_means[i])/sc_stds[i])
                axs_tf_kde_std.set_xlabel("Feature value (standarized)")

                axs_tf_kde_t.set_xlabel("Feature value (original)")
                axs_tf_kde_t.set_ylabel("Distribution in %")
                axs_tf_kde_t.set_title("Feature %i" %(i+1))
                axs_tf_kde_t.legend()
                axs_tf_kde_t.grid()


                axs_tf_kde_log_std = axs_tf_kde_log_t.twiny()
                ax1, ax2 = axs_tf_kde_log_t.get_xlim()
                axs_tf_kde_log_std.set_xlim((ax1-sc_means[i])/sc_stds[i], (ax2-sc_means[i])/sc_stds[i])
                axs_tf_kde_log_std.set_xlabel("Feature value (standarized)")

                axs_tf_kde_log_t.set_xlabel("Feature value (original)")
                axs_tf_kde_log_t.set_ylabel("Distribution in %")
                axs_tf_kde_log_t.set_title("Feature %i" %(i+1))
                axs_tf_kde_log_t.legend()
                axs_tf_kde_log_t.grid()
                axs_tf_kde_log_t.set_yscale("log")
                axs_tf_kde_log_t.set_ylim(bottom=1e-5, top=10**np.ceil(np.log10(max(dex5.max(), dex6.max(), dex7.max(), dex8.max())*100)))

        ## Remove empty subplots
        for i in range(nf, px*py):
            fig.delaxes(axs[-1, int(np.floor(i/px))])
            fig_kde.delaxes(axs_kde[-1, int(np.floor(i/px))])
            fig_kde_log.delaxes(axs_kde_log[-1, int(np.floor(i/px))])

            if Mask_Flag == True:
                fig_tf.delaxes(axs_tf[-1, int(np.floor(i/px))])
                fig_tf_kde.delaxes(axs_tf_kde[-1, int(np.floor(i/px))])
                fig_tf_kde_log.delaxes(axs_tf_kde_log[-1, int(np.floor(i/px))])

        fig_kde.tight_layout()
        fig_kde_log.tight_layout()
            
        if Mask_Flag == True:
            fig_tf_kde.tight_layout()
            fig_tf_kde_log.tight_layout()

        ## Save images
        if mode == "full":
            fig_kde.savefig("%sFeature_Distribution_Predict_kde.pdf" %(CalcParameters['PlotPath']), dpi='figure', format="pdf",
                        metadata=None, bbox_inches="tight", pad_inches=0.1, backend=None)
            fig_kde_log.savefig("%sFeature_Distribution_Predict_kde_log.pdf" %(CalcParameters['PlotPath']), dpi='figure', format="pdf",
                        metadata=None, bbox_inches="tight", pad_inches=0.1, backend=None)

            if Mask_Flag == True:
                fig_tf_kde.savefig("%sFeature_Distribution_TrueFalse_kde.pdf" %(CalcParameters['PlotPath']), dpi='figure', format="pdf",
                            metadata=None, bbox_inches="tight", pad_inches=0.1, backend=None)
                fig_tf_kde_log.savefig("%sFeature_Distribution_TrueFalse_kde_log.pdf" %(CalcParameters['PlotPath']), dpi='figure', format="pdf",
                            metadata=None, bbox_inches="tight", pad_inches=0.1, backend=None)

        elif mode == "test":
            fig_kde.savefig("%sTest-Mode_Feature_Distribution_Predict_kde.pdf" %(CalcParameters['PlotPath']), dpi='figure', format="pdf",
                        metadata=None, bbox_inches="tight", pad_inches=0.1, backend=None)
            fig_kde_log.savefig("%sTest-Mode_Feature_Distribution_Predict_kde_log.pdf" %(CalcParameters['PlotPath']), dpi='figure', format="pdf",
                        metadata=None, bbox_inches="tight", pad_inches=0.1, backend=None)

            if Mask_Flag == True:
                fig_tf_kde.savefig("%sTest-Mode_Feature_Distribution_TrueFalse_kde.pdf" %(CalcParameters['PlotPath']), dpi='figure', format="pdf",
                            metadata=None, bbox_inches="tight", pad_inches=0.1, backend=None)
                fig_tf_kde_log.savefig("%sTest-Mode_Feature_Distribution_TrueFalse_kde_log.pdf" %(CalcParameters['PlotPath']), dpi='figure', format="pdf",
                            metadata=None, bbox_inches="tight", pad_inches=0.1, backend=None)

        plt.close("all")


    ## Plot the 2D feature distribution, not as well commented as the code for single cubes but almost identical
    if CalcParameters["CHECK_FV2D"] == True:

        ## Generate overview figure, if there is more than just one figure
        if nf != 1:
            if CalcParameters["ThesisMode"] == False:
                fig_o_p_kde, ax_o_p_kde = plt.subplots(nf, nf, figsize=(4*nf+2,4*nf))
            else:
                fig_o_p_kde, ax_o_p_kde = plt.subplots(nf, nf, figsize=(7.47))
            
            if Mask_Flag == True:
                if CalcParameters["ThesisMode"] == False:
                    fig_o_m_kde, ax_o_m_kde = plt.subplots(nf, nf, figsize=(4*nf+2,4*nf))
                    fig_o_f_kde, ax_o_f_kde = plt.subplots(nf, nf, figsize=(4*nf+2,4*nf))
                else:
                    fig_o_m_kde, ax_o_m_kde = plt.subplots(nf, nf, figsize=(7.47, 7.47))
                    fig_o_f_kde, ax_o_f_kde = plt.subplots(nf, nf, figsize=(7.47, 7.47))

        ## Iterate over all feature combinations
        for i in range(1,nf+1):

            for ii in range(1,nf+1):

                ## Extend in Fi and Fii direction
                xmi = CalcParameters["pedo"][i-1]
                xma = CalcParameters["peup"][i-1]
                ymi = CalcParameters["pedo"][ii-1]
                yma = CalcParameters["peup"][ii-1]

                ## xg/yg grid
                xg = np.linspace(xmi, xma, xn)
                yg = np.linspace(ymi, yma, yn)

                xm, ym = np.meshgrid(xg, yg, indexing="ij")

                mg = np.array([xm.reshape(-1), ym.reshape(-1)]).T

                ## Read out the summed up grids and averaging them
                if i == ii:
                    dexppo = CalcParameters["FD2D_pred_F%i_o" %(i)] / CalcParameters["FD2S_dp"]
                    if Mask_Flag == True:
                        dexpmo = CalcParameters["FD2D_mask_F%i_o" %(i)] / CalcParameters["FD2S_dp"]
                        dexpfp = CalcParameters["FD2D_false_F%i_o" %(i)] / CalcParameters["FD2S_dp"]
                                
                    dexppn = CalcParameters["FD2D_pred_F%i_n" %(i)] / CalcParameters["FD2S_dp"]

                    if Mask_Flag == True:
                        dexpmn = CalcParameters["FD2D_mask_F%i_n" %(i)] / CalcParameters["FD2S_dp"]
                        dexpfn = CalcParameters["FD2D_false_F%i_n" %(i)] / CalcParameters["FD2S_dp"]

                else:
                    dexpo = CalcParameters["FD2D_pred_F%ivsF%i" %(i, ii)] / CalcParameters["FD2S_dp"]
                    if Mask_Flag == True:
                        dexpm = CalcParameters["FD2D_mask_F%ivsF%i" %(i, ii)] / CalcParameters["FD2S_dp"]
                        dexpf = CalcParameters["FD2D_false_F%ivsF%i" %(i, ii)] / CalcParameters["FD2S_dp"]

                ## Plot the diagonal entries
                if i == ii:
                    if nf == 1:
                        ax_o_p_kde.plot(xg, dexppo, label="Outflow", color="red")
                        ax_o_p_kde.plot(xg, dexppn, label="Non-Outflow", color="blue")
                        ax_o_p_kde.legend()


                        if Mask_Flag == True:
                            ax_o_m_kde.plot(xg, dexpmo, label="Outflow", color="red")
                            ax_o_m_kde.plot(xg, dexpmn, label="Non-Outflow", color="blue")
                            ax_o_m_kde.legend()
                            ax_o_f_kde.plot(xg, dexpfp, label="FP", color="red")
                            ax_o_f_kde.plot(xg, dexpfn, label="FN", color="blue")
                            ax_o_f_kde.legend()
                    
                    else:
                        ax_o_p_kde[ii-1,i-1].plot(xg, dexppo, label="Outflow", color="red")
                        ax_o_p_kde[ii-1,i-1].plot(xg, dexppn, label="Non-Outflow", color="blue")
                        ax_o_p_kde[ii-1,i-1].legend()

                        if Mask_Flag == True:
                            ax_o_m_kde[ii-1,i-1].plot(xg, dexpmo, label="Outflow", color="red")
                            ax_o_m_kde[ii-1,i-1].plot(xg, dexpmn, label="Non-Outflow", color="blue")
                            ax_o_m_kde[ii-1,i-1].legend()
                            ax_o_f_kde[ii-1,i-1].plot(xg, dexpfp, label="FP", color="red")
                            ax_o_f_kde[ii-1,i-1].plot(xg, dexpfn, label="FN", color="blue")
                            ax_o_f_kde[ii-1,i-1].legend()

                ## For more than 2 features plot the off diagonal entries
                elif nf >= 2:

                    if i >= ii:
                        color="red"
                        color2d="Reds"

                    else:
                        color="blue"
                        color2d="Blues"

                    fxmin = xg[0] - (xg[1]-xg[0])/2
                    fxmax = xg[-1] + (xg[1]-xg[0])/2
                    fymin = yg[0] - (yg[1]-yg[0])/2
                    fymax = yg[-1] + (yg[1]-yg[0])/2

                    im_o_p_kde = ax_o_p_kde[ii-1,i-1].imshow(dexpo, aspect="auto", origin="lower", cmap=color2d, extent=[fxmin, fxmax, fymin, fymax])
                    
                    ### Add colorbar     
                    im_ratio = dexpo.shape[0]/dexpo.data.shape[1] 
                    cbar_o_p_kde = plt.colorbar(im_o_p_kde, fraction=0.047*im_ratio) 
            
                    if Mask_Flag == True:

                        im_o_m_kde = ax_o_m_kde[ii-1,i-1].imshow(dexpm, aspect="auto", origin="lower", cmap=color2d, extent=[fxmin, fxmax, fymin, fymax])
                        
                        ### Add colorbar     
                        cbar_o_m_kde = plt.colorbar(im_o_m_kde, fraction=0.047*im_ratio)

                        im_o_f_kde = ax_o_f_kde[ii-1,i-1].imshow(dexpf, aspect="auto", origin="lower", cmap=color2d, extent=[fxmin, fxmax, fymin, fymax])
                        
                        ### Add colorbar     
                        cbar_o_f_kde = plt.colorbar(im_o_f_kde, fraction=0.047*im_ratio)


                    if nf == 2:

                        ## Determine contour levels
                        levels = (50)

                        if dexpo.sum() >= 0:
                            dexpos = sorted(dexpo.reshape(-1)/dexpo.sum())

                            plevels = ()

                            for level in levels:
                                ind = np.where(np.cumsum(dexpos) <= level/100)[0][-1]
                                plevels = np.append(plevels, dexpos[ind]*dexpo.sum())

                            plevels = list(dict.fromkeys(plevels))

                        else:
                            plevels = (-1,0,1)

                        ## Plot the contours
                        ax_o_p_kde[ii-1,i-1].contour(dexpo, levels=plevels, origin="lower", colors=color, extent=[fxmin, fxmax, fymin, fymax])
                        cbar_o_p_kde.set_label("Classification")
                            
                        ax_o_p_kde[i-1,ii-1].contour(dexpo.T, levels=plevels, origin="lower", colors=color, extent=[fymin, fymax, fxmin, fxmax])

                        if Mask_Flag == True:
    
                            if dexpm.sum() >= 0:
                                dexpms = sorted(dexpm.reshape(-1)/dexpm.sum())

                                mlevels = ()

                                for level in levels:
                                    ind = np.where(np.cumsum(dexpms) <= level/100)[0][-1]
                                    mlevels = np.append(mlevels, dexpms[ind]*dexpm.sum())

                                mlevels = list(dict.fromkeys(mlevels))

                            else:
                                mlevels = (-1,0,1)

                            if dexpm.sum() >= 0:
                                dexpfs = sorted(dexpf.reshape(-1)/dexpf.sum())

                                flevels = ()

                                for level in levels:
                                    ind = np.where(np.cumsum(dexpfs) <= level/100)[0][-1]
                                    flevels = np.append(flevels, dexpfs[ind]*dexpf.sum())

                                flevels = list(dict.fromkeys(flevels))

                            else:
                                flevels = (-1,0,1)

                            ax_o_m_kde[ii-1,i-1].contour(dexpm, levels=mlevels, origin="lower", colors=color, extent=[fxmin, fxmax, fymin, fymax])
                            cbar_o_m_kde.set_label("Classification")
                            
                            ax_o_m_kde[i-1,ii-1].contour(dexpm.T, levels=mlevels, origin="lower", colors=color, extent=[fymin, fymax, fxmin, fxmax])

                            ax_o_f_kde[ii-1,i-1].contour(dexpf, levels=flevels, origin="lower", colors=color, extent=[fxmin, fxmax, fymin, fymax])
                            cbar_o_f_kde.set_label("Classification")
                            
                            ax_o_f_kde[i-1,ii-1].contour(dexpf.T, levels=flevels, origin="lower", colors=color, extent=[fymin, fymax, fxmin, fxmax])

                    ## For more features calculate more contours
                    if nf >= 3:
                        
                        levels = ( 5, 25, 50, 75, 95)

                        if dexpo.sum() >= 0:
                                
                            dexpos = sorted(dexpo.reshape(-1)/dexpo.sum())

                            plevels = ()

                            for level in levels:
                                ind = np.where(np.cumsum(dexpos) <= level/100)[0][-1]
                                plevels = np.append(plevels, dexpos[ind]*dexpo.sum())

                        else:
                            plevels = (-1,0,1)

                        ax_o_p_kde[ii-1,i-1].contour(dexpo, levels=plevels, origin="lower", colors=color, extent=[fxmin, fxmax, fymin, fymax])
                        cbar_o_p_kde.set_label("Probability distribution in %")
                            
                        ax_o_p_kde[i-1,ii-1].contour(dexpo.T, levels=plevels, origin="lower", colors=color, extent=[fymin, fymax, fxmin, fxmax])

                        if Mask_Flag == True:
    
                            if dexpm.sum() >= 0:
                                dexpms = sorted(dexpm.reshape(-1)/dexpm.sum())

                                mlevels = ()

                                for level in levels:
                                    ind = np.where(np.cumsum(dexpms) <= level/100)[0][-1]
                                    mlevels = np.append(mlevels, dexpms[ind]*dexpm.sum())

                            else:
                                mlevels = (-1,0,1)
                                
                            if dexpf.sum() >= 0:
                                dexpfs = sorted(dexpf.reshape(-1)/dexpf.sum())

                                flevels = ()

                                for level in levels:
                                    ind = np.where(np.cumsum(dexpfs) <= level/100)[0][-1]
                                    flevels = np.append(flevels, dexpfs[ind]*dexpf.sum())

                            else:
                                flevels = (-1,0,1)

                            ax_o_m_kde[ii-1,i-1].contour(dexpm, levels=mlevels, origin="lower", colors=color, extent=[fxmin, fxmax, fymin, fymax])
                            cbar_o_m_kde.set_label("Probability distribution in %")
                            
                            ax_o_m_kde[i-1,ii-1].contour(dexpm.T, levels=mlevels, origin="lower", colors=color, extent=[fymin, fymax, fxmin, fxmax])

                            ax_o_f_kde[ii-1,i-1].contour(dexpf, levels=flevels, origin="lower", colors=color, extent=[fxmin, fxmax, fymin, fymax])
                            cbar_o_f_kde.set_label("Probability distribution in %")
                            
                            ax_o_f_kde[i-1,ii-1].contour(dexpf.T, levels=flevels, origin="lower", colors=color, extent=[fymin, fymax, fxmin, fxmax])


                ## Adjust the x and y_distr axis                    
                if nf == 1:
                    ax_o_p_kde.set_ylabel(r"Classification")
                    ax_o_p_kde.set_xlabel(r"Feature %i [Standarized]" %(ii))
                    if Mask_Flag == True:
                        ax_o_m_kde.set_xlabel(r"Feature %i [Standarized]" %(ii))
                        ax_o_m_kde.set_ylabel(r"Classification")
                        ax_o_f_kde.set_xlabel(r"Feature %i [Standarized]" %(ii))
                        ax_o_f_kde.set_ylabel(r"Classification")

                elif i == ii:
                    ax_o_p_kde[ii-1,i-1].set_xlabel(r"Feature %i [Standarized]" %(ii))
                    ax_o_p_kde[ii-1,i-1].set_ylabel(r"Probability distribution in %")

                    if Mask_Flag == True:
                        ax_o_m_kde[ii-1,i-1].set_xlabel(r"Feature %i [Standarized]" %(ii))
                        ax_o_m_kde[ii-1,i-1].set_ylabel(r"Probability distribution in %")
                        ax_o_f_kde[ii-1,i-1].set_xlabel(r"Feature %i [Standarized]" %(ii))
                        ax_o_f_kde[ii-1,i-1].set_ylabel(r"Probability distribution in %")
                        
                else:
                    ax_o_p_kde[ii-1,i-1].set_xlabel(r"Feature %i [Standarized]" %(i))
                    ax_o_p_kde[ii-1,i-1].set_ylabel(r"Feature %i [Standarized]" %(ii))

                    if Mask_Flag == True:
                        ax_o_m_kde[ii-1,i-1].set_xlabel(r"Feature %i [Standarized]" %(i))
                        ax_o_m_kde[ii-1,i-1].set_ylabel(r"Feature %i [Standarized]" %(ii))

                        ax_o_f_kde[ii-1,i-1].set_xlabel(r"Feature %i [Standarized]" %(i))
                        ax_o_f_kde[ii-1,i-1].set_ylabel(r"Feature %i [Standarized]" %(ii))


                if nf == 1:
                    ax_o_p_kde.grid()

                    if Mask_Flag == True:
                        ax_o_m_kde.grid()
                        ax_o_f_kde.grid()

                else:
                    ax_o_p_kde[ii-1,i-1].grid()

                    if Mask_Flag == True:
                        ax_o_m_kde[ii-1,i-1].grid()
                        ax_o_f_kde[ii-1,i-1].grid()

        ## Save the image
        fig_o_p_kde.tight_layout()
        fig_o_p_kde.savefig("%sFeatureDistributionGrid_pred_kde_Overview.pdf" %(CalcParameters['PlotPath']), dpi='figure', format="pdf",
                    metadata=None, bbox_inches="tight", pad_inches=0.1, backend=None)

        if Mask_Flag == True:
            fig_o_m_kde.tight_layout()
            fig_o_m_kde.savefig("%sFeatureDistributionGrid_mask_kde_Overview.pdf" %(CalcParameters['PlotPath']), dpi='figure', format="pdf",
                        metadata=None, bbox_inches="tight", pad_inches=0.1, backend=None)
            fig_o_f_kde.tight_layout()
            fig_o_f_kde.savefig("%sFeatureDistributionGrid_false_kde_Overview.pdf" %(CalcParameters['PlotPath']), dpi='figure', format="pdf",
                        metadata=None, bbox_inches="tight", pad_inches=0.1, backend=None)

        plt.close("all")        


    ## Plot the confusion matrix
    if CalcParameters["Plot_CM"] == True and CalcParameters["MASK"] == True:

        ## Define Class Names
        class_label = ["Non-OF Voxel", "OF Voxel"]

        ## Plot the confusion matrix
        if mode == "full":
            Plot_Confusion_Matrix(cm_comp, percent_mode=False, ThesisMode=CalcParameters["ThesisMode"], class_label=class_label, ppath=CalcParameters["PlotPath"], pname="Confusion_Matrix.pdf")
            try:
                Plot_Confusion_Matrix(cm_comp_p/(i_set+1), percent_mode=True, ThesisMode=CalcParameters["ThesisMode"], class_label=class_label, ppath=CalcParameters["PlotPath"], pname="Confusion_Matrix_p.pdf")
            except:
                print("Plot Confusion Matrix Percent Failed...")

        elif mode == "test":
            Plot_Confusion_Matrix(cm_comp, percent_mode=False, ThesisMode=CalcParameters["ThesisMode"], class_label=class_label, ppath=CalcParameters["PlotPath"], pname="Test-Mode_Confusion_Matrix.pdf")
            try:
                Plot_Confusion_Matrix(cm_comp_p/(i_set+1), percent_mode=True, ThesisMode=CalcParameters["ThesisMode"], class_label=class_label, ppath=CalcParameters["PlotPath"], pname="Test-Mode_Confusion_Matrix_p.pdf")
            except:
                print("Plot Confusion Matrix Percent Failed...")
                

    ## Save the updated database
    if mode == "full" and Mask_Flag ==  True:
        data_pd.to_csv("%s%s" %(CalcParameters["DatePath"], CalcParameters["database_short"]))
    
    elif mode == "test" and Mask_Flag ==  True:
        data_pd.to_csv("%sTest-Mode_%s" %(CalcParameters["DatePath"], CalcParameters["database_short"]))
    
    ## Return nothing
    return 