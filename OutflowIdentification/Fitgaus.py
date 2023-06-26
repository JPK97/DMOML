# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 18:15:18 2018

@author: Phillip
"""

## Import packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import odr

def fit_gaus(xdata, ydata, xerr=0, yerr=0, name=False, plot=True):

    ## Define fit function, here Gaussian, has to be identical but for two seperate fit functions    
    def func0(x, a, b, c, d):
        return a * np.exp(-(x-b)**2/(2*c**2)) + d

    def func1(p, x):
        a, b, c, d = p
        return a * np.exp(-(x-b)**2/(2*c**2)) + d
    
    # Model object
    quad_model = odr.Model(func1)

    # Error-arrays
    if type(xerr) == int or type(xerr) == float:
        xerr = np.repeat(xerr,len(xdata))
        
    if type(yerr) == int or type(yerr) == float:
        yerr = np.repeat(yerr,len(ydata))

    ## Initial guess for fit
    a0 = np.nanmax(ydata)-np.nanmean(ydata)
    b0 = xdata[np.where(ydata==np.nanmax(ydata))[0][0]]
    d0 = np.nanmean(ydata)
        
    try:
        above45i = np.where(ydata >= .45*(a0-d0))[0]
        c0 = (xdata[above45i[-1]] - xdata[above45i[0]]) / (2*np.sqrt(2*np.log(2)))
            
    except:
        c0 = 1
        
    popt0 = (a0,b0,c0,d0) # a guess for slope, intercept, slope, intercept 
    
    # First fit
    try:
        if yerr.any() != 0:
            popt, pcov = curve_fit(func0, xdata, ydata, sigma=yerr, p0=popt0)

        else:
            popt, pcov = curve_fit(func0, xdata, ydata, p0=popt0)

        # Parameter
        perr = np.sqrt(np.diag(pcov))

    ## If it fails, take the initial guess as fit parameters
    except:
        popt = popt0

    ## Second fit
    try:
        # Create a RealData object
        if xerr.all() == 0 and yerr.all() == 0:
            data = odr.RealData(xdata, ydata)
        
        elif xerr.any() != 0 and yerr.all() == 0:
            data = odr.RealData(xdata, ydata, sx=xerr)
        
        elif xerr.all() == 0 and yerr.any() != 0:
            data = odr.RealData(xdata, ydata, sy=yerr)
        
        elif xerr.any() != 0 and yerr.any() != 0:
            data = odr.RealData(xdata, ydata, sx=xerr, sy=yerr)
    
        # Set up ODR with the model and data.
        Odr = odr.ODR(data, quad_model, beta0=popt)
    
        # Run the regression.
        out = Odr.run()
    
        # Fitted parameters
        popt = out.beta     # Parameters
        perr = out.sd_beta  # Errors
    
        xmi = min(xdata)
        xma = max(xdata)
    
        x_fit = np.linspace(xmi - (xma-xmi)*.125, xma + (xma-xmi)*.125, 100)
        fit = func1(popt, x_fit)

    ## If it fails, take the results of the first fit
    except:
        continue
    

    # Plot the results
    if plot == True or name != False:
        fig, ax = plt.subplots(1)

        ## Plot the input data and their error bars
        if xerr.all() == 0 and yerr.all() == 0:
            line,caps,bars=plt.errorbar(xdata, ydata,fmt="r+", linewidth=3, elinewidth=.5, ecolor='k', capsize=5, capthick=.5, label="data")
        
        elif xerr.any() != 0 and yerr.all() == 0:
            line,caps,bars=plt.errorbar(xdata, ydata, xerr=xerr, fmt="r+", linewidth=3, elinewidth=.5, ecolor='k', capsize=5, capthick=.5, label="data")
        
        elif xerr.all() == 0 and yerr.any() != 0:
            line,caps,bars=plt.errorbar(xdata, ydata, yerr=yerr, fmt="r+", linewidth=3, elinewidth=.5, ecolor='k', capsize=5, capthick=.5, label="data")
        
        elif xerr.any() != 0 and yerr.any() != 0:
            line,caps,bars=plt.errorbar(xdata, ydata, xerr=xerr, yerr=yerr, fmt="r+", linewidth=3, elinewidth=.5, ecolor='k', capsize=5, capthick=.5, label="data")
    
        ## Set figure descirptions
        plt.xlabel("Velocity Channels")
        plt.ylabel("Mean Intensity")
        plt.title("Fitted gaussian function")
        plt.plot(x_fit, fit, "g", lw=2, label="fit")
        plt.legend()
        plt.grid()
        plt.tight_layout()

        ## Save figure and fit parameters if requested
        if name != False:
            plt.savefig('./../Plots/%s.pdf' %(name),
                    transparent=True, pad_inches=0.0, orientation='portrait',
                    format="pdf")
            plt.close()

            results = "Line function:\n\nf(c) = a*x + b\na = %.3E +- %.3E\nb = %.3E +- %.3E" %(popt[0], perr[0], popt[1], perr[1])
        
            ## Saving the results into a text document
            file = open('./../Results/Fit%s.txt' %(name), "w") 
        
            file.write(results)
        
        ## Otherwise show or just close the fit
        if plot == False:
            plt.close()

        else:
            plt.show()

    ## Return fit parameters
    return popt, perr
