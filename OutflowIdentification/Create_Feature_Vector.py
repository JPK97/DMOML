# -*- coding: iso-8859-1 -*-

"""
Created on Thu Jun 17 16:39:25 2021

@author: Phillip
"""

##############################################################################
### Include some packages

## Import existing packages
import astropy.io.fits as fits
from scipy.ndimage import convolve
from astropy.stats import sigma_clip
from datetime import datetime as datetime
import numpy as np
from skimage.filters import difference_of_gaussians as DoG
from spectral_cube import SpectralCube as sc
import pandas as pd
import warnings

## Import self written packages
from Fitgaus import fit_gaus as fg

def Feature_Extractor(CalcParameters, FILE_CUBE, FILE_MASK="NoMask"):
                    
    ## Try to derive Feature Vectors
    try:
        ## Read out data
        data = sc.read(FILE_CUBE).to("K").hdu.data

        ## Check if all entries are nan or inf
        if not np.isnan(data).all() and np.isinf(data).all():

            ## Create original copy of data
            data_o = np.copy(data)

            ## Shape of data cube
            lz, lx, ly = np.shape(data)

            ## Read out mask
            if FILE_MASK != "NoMask":
                mask = sc.read(FILE_MASK).hdu.data
            else:
                mask = np.zeros_like(data)

            ## Calculate mean of each slice and std of the background using sigma clipping
            slmean = np.nanmean(data, axis=(1,2)).reshape(lz,1,1)
            data_sc = sigma_clip(data, sigma=2, maxiters=None, cenfunc="mean")
            scstd = np.nanstd(data_sc)

            ## Avoid too small or nan mean and std values
            slmean = np.ma.fix_invalid(slmean, fill_value=1e-5)
            slmean = np.maximum(slmean, 1e-5)
            scstd = np.maximum(scstd, 1e-5)

            ## Read out important header parameters
            header = fits.getheader(FILE_CUBE)
            crval3 = header['CRVAL3']
            cdelt3 = header['CDELT3']
            crpix3 = header['CRPIX3']
            cunit3 = header['CUNIT3']
            cdelt1 = header['CDELT1']
            cdelt2 = header['CDELT2']
            BMAJ = header['BMAJ']

            ## Subcube extend in spatial dimension; defined via SCS times the beam size
            npsc1 = max(int(np.ceil(BMAJ*CalcParameters["SigmaSC"]*0.5/cdelt1 - 0.5)) * 2 + 1, 3)
            npsc2 = max(int(np.ceil(BMAJ*CalcParameters["SigmaSC"]*0.5/cdelt2 - 0.5)) * 2 + 1, 3)

            ## Subcube extend in velocity dimension; defined via 5 km/s
            if cunit3 == "kms-1" or cunit3 == "km/s" or cunit3 == "km s-1" or cunit3 == "km / s":
                nvsc = max(int(np.ceil(CalcParameters["VelocitySC"]*0.5/cdelt3 - 0.5)) * 2 + 1, 7)
            elif cunit3 == "ms-1" or cunit3 == "m/s" or cunit3 == "m s-1" or cunit3 == "m / s":
                nvsc = max(int(np.ceil(CalcParameters["VelocitySC"]*1000*0.5/cdelt3 - 0.5)) * 2 + 1, 7)
            else:
                print("The velocity unit is unknown. Assuming it is in km/s.")
                nvsc = max(int(np.ceil(CalcParameters["VelocitySC"]*0.5/cdelt3 - 0.5)) * 2 + 1, 7)

            ## Get "radius" of subcube
            rpsc1 = int((npsc1-1)/2)
            rpsc2 = int((npsc2-1)/2)
            rpsc3 = int((nvsc-1)/2)

            ## Swapp the invalid voxels with the mean of the sc around it
            inv_ind = np.array(np.where(np.logical_or(np.isnan(data), np.isinf(data)) == True)).T
            for zi, xi, yi in inv_ind:
                sub_cube = data_o[max(zi-rpsc3,0):min(zi-rpsc3,lz), max(xi-rpsc1,0):min(xi-rpsc1,lx), max(yi-rpsc2,0):min(yi-rpsc2,ly)]
                if not np.isnan(np.nanmean(sub_cube)):
                    data[zi, xi, yi] = np.nanmean(sub_cube)
                else:
                    data[zi, xi, yi] = np.nanmean(data_o)

            ## Place-holder for the signal-to-noise value sigma
            sigma = np.zeros_like(data, dtype=int)

            ## Create enlarged x, y, z arrays
            x = np.linspace(0,lx-1,lx, dtype=int)
            y = np.linspace(0,ly-1,ly, dtype=int)
            z = np.linspace(0,lz-1,lz, dtype=int)
            zm, xm, ym = np.meshgrid(z, x, y, indexing="ij")

            ## Calculate the average brightnes of the subcube
            weights_sc = np.ones((nvsc, npsc1, npsc2)) / (nvsc*npsc1*npsc2)
            T_sub_cube = convolve(data, weights_sc, mode="constant", cval=0)

            ## Calculate the maximal spatial temperatures
            T_spat = np.max(data, axis=0)
            weights_ts = np.ones((npsc1, npsc2)) / (npsc1*npsc2)
            T_spat = convolve(T_spat, weights_ts, mode="constant", cval=0)
            T_spat = T_spat.reshape(1,lx,ly)

            ## Voxel velocity
            v_voxel = velo(zm, crval3, cdelt3, crpix3)

            ## Slope of subcube, solved analyticly
            weights_sl = np.ones((nvsc, npsc1, npsc2)) / (npsc1*npsc2)
            slope_x  = convolve(v_voxel, weights_sl, mode="constant", cval=0)
            slope_y  = convolve(data, weights_sl, mode="constant", cval=0)
            slope_xx = convolve(v_voxel*v_voxel, weights_sl, mode="constant", cval=0)
            slope_xy = convolve(v_voxel*data, weights_sl, mode="constant", cval=0)
            slope    = (nvsc * slope_xy - slope_x * slope_y) / (nvsc * slope_xx - slope_x * slope_x)
    
            ## Line width
            line_width = sc.read(FILE_CUBE).linewidth_sigma().value

            ## System velocity; fit the average spectrum
            fit_x = np.linspace(0,lz-1,lz, dtype=int)
            fit_y = np.nanmean(np.nanmean(data, axis=1), axis=1)
    
            ## Remove nans and infs
            if np.isnan(fit_y).sum() != 0:        
                fit_x = np.delete(fit_x, np.isnan(fit_y))
                fit_y = np.delete(fit_y, np.isnan(fit_y))
        
            if np.isinf(fit_y).sum() != 0:        
                fit_x = np.delete(fit_x, np.isinf(fit_y))
                fit_y = np.delete(fit_y, np.isinf(fit_y))

            ## Executing the fit (self-written) and extract the system velocity
            try:
                p_rest = fg(fit_x, fit_y,plot=False)[0][1]

            except:
                p_rest = fit_x[np.where(fit_y==np.nanmax(fit_y))[0][0]]
            v_cloud = velo(p_rest, crval3, cdelt3, crpix3)

            ## Prepare DoG
            DoGarrayint = np.zeros_like(data)
            for i in np.linspace(-(nvsc-1)/2, (nvsc-1)/2, nvsc, dtype=int):
                DoGarrayint[max(i, 0):min(lz+i, lz)] += data[max(-i, 0):min(lz-i, lz)]

            ## Calculating DoG
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                DoGarray = DoG(DoGarrayint, 1*BMAJ/max(cdelt1,cdelt2), 2*BMAJ/max(cdelt1,cdelt2))


            ### Calculate the features
            ### Feature 1
            feature_1 = (T_sub_cube - slmean) / scstd

            ### Feature 2
            feature_2 = T_sub_cube / T_spat

            ### Feature 3
            feature_3 = abs(v_voxel - v_cloud) / line_width

            ### Feature 4
            feature_4 = slope * line_width / T_spat * np.sign(v_voxel - v_cloud)

            ### Feature 5
            feature_5 = DoGarray


            ### Produce Feature Vectors out of these arrays, turn array shape into 1d
            nfv = lx*ly*lz
            xm_l = xm.reshape(nfv)          # X-value
            ym_l = ym.reshape(nfv)          # Y-value
            zm_l = zm.reshape(nfv)          # Z-value
            si_l = sigma.reshape(nfv)       # Sigma to noise value to filter input; outdated but could become relevant again
            ma_l = mask.reshape(nfv)        # Outflow mask (0 --> nOF, 1 --> OF)
            f1_l = feature_1.reshape(nfv)   # Feature 1
            f2_l = feature_2.reshape(nfv)   # Feature 2
            f3_l = feature_3.reshape(nfv)   # Feature 3
            f4_l = feature_4.reshape(nfv)   # Feature 4
            f5_l = feature_5.reshape(nfv)   # Feature 5

            ## Group data and generate pandas DataFrame
            data = np.array((zm_l, xm_l, ym_l, si_l, ma_l, f1_l, f2_l, f3_l, f4_l, f5_l))

            header = ["z position", "x position", "y position", "Sigma Value", "OF Pixel"]
            for i in range(len(data)-len(header)):
                header = np.append(header, "Feature %i" %(i+1))

            df = pd.DataFrame(data=data.T, columns=header, index=None)
    
            return df

        ## Return warning if the cube is invalid
        else:
            print("\nWarning: The array %i contains just nans. It will be ignored for further calculations and removed from the database." %(i_set))
        
            return False

    ## Return warning if the cube is invalid
    except Exception as e:
        exception_type, exception_object, exception_traceback = sys.exc_info()
        filename = exception_traceback.tb_frame.f_code.co_filename
        line_number = exception_traceback.tb_lineno
        
        print("\nWarning: Failed to derive Feature Vectors from array %i. It will be ignored for further calculations and removed from the database." %(i_set))
        print("Exception type:\t%s" %(exception_type))
        print("File name:\t%s" %(filename))
        print("Line number:\t%s" %(line_number))
        print("The error itself:\n%s\n\n" %(e))

        return False
            
## Create a function to calculate the velocities of all values 
def velo(x, crval3, cdelt3, crpix3):
    return -crpix3*cdelt3+crval3+cdelt3*(x+1)        


##############################################################################
### Function that checks if the database is valid and if it contains a mask, saves a copy afterwards
def check_copy_database(CalcParameters):

    ## Read the database header
    database = pd.read_csv("%s%s" %(CalcParameters["DatePath"], CalcParameters["database_file"]), index_col=False)

    ## Check if mandatory column entries are in the header
    if "Parent dir" in database.columns and "Cube name" in database.columns:
        
        ## Drop unnamed lines
        database.drop(database.filter(regex="Unname").columns, axis=1, inplace=True)


        ## If the Mask name entry doesn't exist, set the MASK flag to False
        if "Mask name" not in database.columns and CalcParameters["MASK"] == True:
            print("Warning, the database has no information on the masks. Please check the database.\nTurning the MASK flag to 'False'.")
            CalcParameters["MASK"] = False

    ## In case of an invalid header --> count the number of columns and set a matching header
    else:

        ## 2 columns --> dir, cube
        if len(database.columns) == 2:
            header = ["Parent dir", "Cube name"]
            
            ## Turn the MASK flag to False if it's True
            if CalcParameters["MASK"] == True:
                print("Warning, the database has no information on the masks. Please check the database.\nTurning the MASK flag to 'False'.")
                CalcParameters["MASK"] = False
                
        ## 3 columns --> dir, cube, mask
        elif len(database.columns) == 3:
            header = ["Parent dir", "Cube name", "Mask name"]

        ## else the database has a wrong format
        else:
            error_text_line_1 = "The database header doesn't contain the two requested arguements 'Parent dir' and 'Cube name' and, therefore, is invalid.\n"
            error_text_line_2 = "In case the database would have a header, it would be:\n%s\n" %(database.columns)
            error_text_line_3 = "Furthermore, the database has an invalid shape. For non-header data, only data sets with 2 or 3 columns are accepted.\n"
            error_text_line_4 = "This database has %i columns.\n" %(len(database.columns))
            error_text_line_5 = "Please correct the database and/or its header try again.\n"

            error_text = error_text_line_1 + error_text_line_2 + error_text_line_3 + error_text_line_4 + error_text_line_5
            
            raise Exception(error_text)

        ## Read in the data but this time with a header
        database = pd.read_csv("%s%s" %(CalcParameters["DatePath"], CalcParameters["database_file"]), columns=header, index_col=False)

    ## Drop duplicates
    database = database.drop_duplicates()

    ## Drop lines containing the trainings/a date
    database.drop(database.filter(regex="Date").columns, axis=1, inplace=True)


    ## Drop constant values excluding model/dir/cube/mask names
    rel_index = (database != database.iloc[0]).any()

    rel_index["Parent dir"] = True
    rel_index["Cube name"] = True
    
    if "Mask name" in database.columns:
        rel_index["Mask name"] = True

    if "FV name" in database.columns:
        rel_index["FV name"] = True

    database = database.loc[:, rel_index]

    ## Save a copy of the database in the DatePath
    database.to_csv("%s%s" %(CalcParameters["DatePath"], CalcParameters["database_short"]))

    return CalcParameters["MASK"]


###############################################################################
### Function that calculates the features of all pixels
def create_feature_vector(CalcParameters, data_set=None):
    
    ## Generate FVs for the whole database if no cube is declared
    if data_set is None:

        ## Read the database
        data_pd = pd.read_csv("%s%s" %(CalcParameters["DatePath"], CalcParameters["database_short"]), index_col=0)

        Parent_Dirs = data_pd["Parent dir"]
        Cube_Names = data_pd["Cube name"]

        if "Mask name" in data_pd.columns and CalcParameters["MASK"] == True:
            Mask_Names = data_pd["Mask name"]

        ## Add a column for the FV dirs and names, if not existing
        if not "FV name" in data_pd.columns:
            data_pd["FV name"] = ""

    ## Or use the handed over data set
    else:
        Parent_Dirs = data_set["Parent dir"]
        Cube_Names = data_set["Cube name"]
        FV_name = data_set["FV name"]

        if "Mask name" in data_set.columns and CalcParameters["MASK"] == True:
            Mask_Names = data_set["Mask name"]

    ## List to catch the indexes of failed feature extraction tries
    inval_cubes = []

    ## Loop over all data sets
    for i_set in range(len(Parent_Dirs)):

        ## Return a status update
        if not i_set+1 == len(Parent_Dirs):
            print("Creating the FV of set %i of %i." %(i_set+1, len(Parent_Dirs)), end="\r")
        else:
            print("Creating the FV of set %i of %i." %(i_set+1, len(Parent_Dirs)))

        ## Get the cube name inkl its dir
        Parent_Dir = Parent_Dirs[i_set]
        Cube_Name = Cube_Names[i_set]
        FILE_CUBE = "%s%s" %(Parent_Dir, Cube_Name)

        ## Get the mask name inkl its dir, if MASK is Ture
        if CalcParameters["MASK"] == True:
            Mask_Name = Mask_Names[i_set]
            FILE_MASK = "%s%s" %(Parent_Dir, Mask_Name)

        else:
            FILE_MASK = "NoMask"

        ## Calculate Feature Vectors
        FV_dset = Feature_Extractor(CalcParameters, FILE_CUBE, FILE_MASK)

        ## Check if run was successful
        if isinstance(FV_dset, pd.DataFrame):

            ## Create a database out of the data set
            if data_set == None:
                FV_name = "%s_FV.pkl" %(".".join(Cube_Name.split(".")[:-1]))

            FV_dset.to_pickle("%s%s" %(CalcParameters["FVPath"], FV_name))

            ## Add FV name to database
            if data_set == None:
                data_pd.loc[i_set, "FV name"] = FV_name

        ## If the run failed, add the index to the list
        else:
            inval_cubes = np.append(inval_cubes, i_set)

    ## Remove all invalid data sets, if neccessary
    if not inval_cubes:
        data_pd = data_pd.drop(index=inval_cubes)

    ## Update the database
    if data_set == None:
        data_pd.to_csv("%s%s" %(CalcParameters["DatePath"], CalcParameters["database_short"]))

    ## Return whole CalcParameters or just the data set
    if data_set is not None:
        return data_set

    else:
        return