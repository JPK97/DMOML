#!/usr/bin/python3
# -*- coding: utf-8 -*-
##********************************************************************************************************************************************************
##
##  Collect results of envelope fits for selected cores
##
##
##  Usage:
##
##      python3 ComputeMolfitFile_for_Outflow.py
##
##
##  Command linie parameter:
##
##      - None
##
##
##
##
##  Copyright (C) 2022
##
##  I. Physikalisches Institut, University of Cologne
##
##
##
##  The following functions are included in this module:
##
##      - function MolfitGenerator.__init__:                        initialize class MolfitGenerator
##      - function MolfitGenerator.CreateCoordinateGrid:            define function to create the coordinate grid
##      - function MolfitGenerator.ComputeOutflowShape:             define function to compute the outflow shape (Raga)
##      - function MolfitGenerator.ComputeX1AndDistance:            define function to compute the x1 and outflow distance (Raga)
##      - function MolfitGenerator.ComputeVelocityMolecule:         define function to compute velocity
##      - function MolfitGenerator.ComputeDensityMolecule:          define function to compute density
##      - function MolfitGenerator.MoleculeDensity:                 calculate the molecular density for a given vector
##      - function MolfitGenerator.ComputeTemperatureMolecule:      define function to compute temperature
##      - function MolfitGenerator.ComputeSourceSizeMolecule:       define function to compute source size
##      - function MolfitGenerator.ComputeLineWidthMolecule:        define function to compute line width
##      - function MolfitGenerator.qTrapz3D:                        integrates function Func(x,y,z) in the
##                                                                  cuboid [ax,bx] * [ay,by] * [az,bz]
##      - function MolfitGenerator.CalcMolfitParameters:            calculate molfit parameters for current distance
##      - function CreateFitsCube:                                  define function to create fits cubes for all parameters
##      - function Start:                                           define function which is called from the CubeFit function
##      - function main:                                            main program (only used for debugging)
##
##
##
##  Versions of the program:
##
##  Who             When            What
##
##  T. Moeller      2022-03-09      initial version
##
##  P. Kahl         2023-06-27      include synthetic outflows
##
##********************************************************************************************************************************************************


##********************************************************************* load packages ********************************************************************
from astropy import constants as const                                                      ## import astropy constants package
from astropy import units as u                                                              ## import astropy units package
from astropy.coordinates import SpectralCoord                                               ## import spectral coordinate package
from astropy.io import fits                                                                 ## import astropy fits package
from astropy.visualization.wcsaxes import add_scalebar                                      ## import astropy add_scalebar package
from astropy.wcs import WCS                                                                 ## import astropy WCS package
import datetime                                                                             ## import datetime package
from distutils.util import strtobool                                                        ## import distutils strtobool package
import itertools                                                                            ## import itertools package
import matplotlib                                                                           ## import matplotlib package
import multiprocessing as mp                                                                ## import multiprocessing package
import numpy                                                                                ## import numpy package
import numpy.ma as ma                                                                       ## import numpys ma/mask package
import os                                                                                   ## import os package
import pandas                                                                               ## import pandas package
from pathlib import Path                                                                    ## import pathlib Path package
import platform                                                                             ## import platform package
import pylab                                                                                ## import pylab
import radio_beam                                                                           ## import radio_beam
from scipy.optimize import minimize                                                         ## import scipy minimize package
from scipy.interpolate import interp1d, RegularGridInterpolator                             ## import interpol1d function from scipy
from spectral_cube import SpectralCube                                                      ## import spectral cube package
import sys                                                                                  ## import sys package
import time                                                                                 ## import time package
import warnings                                                                             ## import warnings package
import xml.etree.ElementTree as ET                                                          ## import xml package
from __future__ import print_function                                                       ## for python 2 usage

##--------------------------------------------------------------------------------------------------------------------------------------------------------


##--------------------------------------------------------------------------------------------------------------------------------------------------------
## Class MolfitGenerator
class MolfitGenerator():


    ##----------------------------------------------------------------------------------------------------------------------------------------------------
    ## initialize main class
    def __init__(self, CalcParameters):
        """

    input parameters:
    -----------------

        - CalcParameters            dictionary containing model parameters


    output parameters:
    ------------------

        - None
        """

        self.debug = CalcParameters["Debug"]
        self.plot = CalcParameters["Plot"]


        ##------------------------------------------------------------------------------------------------------------------------------------------------
        ## copy input parameters
        self.CalcParameters = CalcParameters


        ##------------------------------------------------------------------------------------------------------------------------------------------------
        ## define some constants
        self.G = const.G.value                                                     ## unit:  m^3 / (kg * s^2)
        self.MSun = const.M_sun.value                                              ## unit:  kg
        self.StefanBoltzman = const.sigma_sb.value                                 ## Stefan-Boltzmann constant (W m-2 K-4)
        self.pc2AU = const.pc.to("au").value                                       ## convert pc to au
        self.AU2Meter = const.au.value                                             ## convert au to m
        self.pc2Meter = const.pc.value                                             ## convert pc to m
        self.m3TOcm3 = 1.e6                                                        ## convert m3 to cm3
        self.cm3TOm3 = 1.e-6                                                       ## convert cm3 to m3
        self.m2TOcm2 = 1.e4                                                        ## convert m2 to cm2
        self.m3TOcm2 = 1.e4                                                        ## convert m3 to cm2
        self.msTOkms = 1.e-3                                                       ## convert m to km


        # Debug:
        if self.debug == True:
            print ("\nself.G = ", self.G)
            print ("self.MSun = ", self.MSun)
            print ("self.StefanBoltzman = ", self.StefanBoltzman)
            print ("self.pc2AU = ", self.pc2AU)
            print ("self.AU2Meter = ", self.AU2Meter)
            print ("self.pc2Meter = ", self.pc2Meter)
            print ("self.m3TOcm3 = ", self.m3TOcm3)
            print ("self.cm3TOm3 = ", self.cm3TOm3)
            print ("self.m2TOcm2 = ", self.m2TOcm2)
            print ("self.m3TOcm2 = ", self.m3TOcm2)
            print ("self.msTOkms = ", self.msTOkms)


        ##------------------------------------------------------------------------------------------------------------------------------------------------
        ## define grid

        ## get grid parameters from input dictionary
        self.MaxResolution = self.CalcParameters['MaxResolution']                                         ## get maximal resolution along RA/DEC direction
        self.MaxXYSteps = self.CalcParameters['MaxXYSteps']                                         ## get maximal steps in x and y direction

        self.NumberOfCellsAlongLineOfSight = self.CalcParameters['NumberOfCellsAlongLineOfSight']    ## get the number of cells along the line of sight

        self.DistanceToSource = self.CalcParameters['DistanceToSource']                     ## get distance to SgrB2 in pc
        self.DistanceToSourceMeter = self.DistanceToSource * self.pc2Meter                  ## convert pc to meter

        # Debug:
        if self.debug == True:
            print ("\nresolution = ", self.resolution)
            print ("MaxXYSteps = ", self.MaxXYSteps)
            print ("\nNumberOfCellsAlongLineOfSight = ", self.NumberOfCellsAlongLineOfSight)
            print("DistanceToSource [m] = ", self.DistanceToSourceMeter)


        ##------------
        ## get other parameters
        self.ssComp = self.CalcParameters['ssComp']                                         ## Source size
        self.vWidthComp = self.CalcParameters['vWidthComp']                                 ## line width


        ##------------------------------------------------------------------------------------------------------------------------------------------------
        ## get rotation parameters from input dictionary


        ## get rotation angles
        self.ThetaJet = self.CalcParameters['Theta'] % 360                                        ## get rotation angle around x-axis from 0 to 360 deg
        self.ThetaJet = self.ThetaJet * numpy.pi / 180.0                                        ## convert degree to radiant
        self.PhiJet = self.CalcParameters['Phi'] % 360                                        ## get rotation angle around y-axis from 0 to 360 deg
        self.PhiJet = self.PhiJet * numpy.pi / 180.0                                            ## convert degree to radiant

        ## Calculate the unit outflow vector
        ej = self.PolarCoordinate(self.ThetaJet, self.PhiJet)
        self.ej = ej / numpy.linalg.norm(ej)

        ## Define the z unit vector 
        ez = numpy.array([0,0,1])

        ## Check if theta != 0 --> cross product would return [0, 0, 0]
        if self.ThetaJet != 0:
            self.er = numpy.cross(ez, self.ej)
            self.er /= numpy.linalg.norm(self.er)

        else:
            self.er = numpy.array([1,0,0])

        ## Calculate rotation angle
        self.alpha = numpy.arccos(numpy.dot(ez, self.ej))

        ## Calculate the rotation matrix around the rotation axis
        self.R = self.RotationMatrixAxis(self.alpha, self.er)

        if self.debug == True:
            print ("\nThetaJet = ", self.ThetaJet)
            print ("PhiJet = ", self.PhiJet)
            print ("Unit jet vector = ", self.ej)
            print ("Unit rotation vector = ", self.er)
            print ("Rotation angle = ", self.alpha)
            print ("Rotation matrix = ", self.R)
    
            
        ##------------------------------------------------------------------------------------------------------------------------------------------------
        ##read physical parameters
        ## get the background values
        self.nbackcm3 = self.CalcParameters['nbackground']                                  ## Particle background density in cm-3
        self.nbackm3 = self.CalcParameters['nbackground'] / self.cm3TOm3                    ## Particle background density in m-3
        self.Tback = self.CalcParameters['Tbackground']                                         ## Background temperature in T     
        
        ## debug
        if self.debug == True:
            print("\nn (background) [cm-3] = ", self.nbackcm3)
            print("n (background) [m-3] = ", self.nbackm3)
            print("T (background) [T] = ", self.Tback)
        
        ## get model name
        self.model = self.CalcParameters['modelname']

        # Debug:
        if self.debug == True:
            print("\nmodelname = ", self.model)


        ##------------------------------------------------------------------------------------------------------------------------------------------------
        ## get Cabrit parameters from input dictionary

        if self.model == "Cabrit":

            ## general parameters
            self.alpha = self.CalcParameters['alpha']                                       ## velocity exponent
            self.delta = 2 - self.alpha                                                     ## density exponent
            self.gamma = self.CalcParameters['gamma']                                       ## Molecular gas

            ## outflow shape parameters
            self.dmin = self.CalcParameters['dmin'] * self.AU2Meter                         ## Minimal outflow distance in m 
            self.sigmaof = self.CalcParameters['sigmaof']                                   ## Factor that describes the length of the outflow in units of the inner distance
            self.dmax = self.dmin * self.sigmaof                                            ## Maximal Outflow distance in m
            self.thetamax = self.CalcParameters['ThetaMax'] * numpy.pi/180                  ## Maximal opening angle in rad

            ## initial distribution parameters
            self.vmin = self.CalcParameters['vmax'] #  * (self.dmax/self.dmin)**self.alpha     ## initial jet velocity in km/s
            self.nmin = self.CalcParameters['nmin'] / self.cm3TOm3                          ## initial density in m-3
            self.Tmin = self.CalcParameters['Tmin']                                         ## inítial temperature of the mixing-layer in K

            if self.debug == True:
                print("\nalpha = ", self.alpha)
                print("delta = ", self.delta)
                print("dmin [m] = ", self.dmin)
                print("sigmaof = ", self.sigmaof)
                print("dmax [m] = ", self.dmax)
                print("theta_max [rad] = ", self.thetamax)
                print("vmin [km s-1] = ", self.vmin)
                print("nmin [m-3] = ", self.nmin)
                print("Tmin [K] = ", self.Tmin)

        ##------------------------------------------------------------------------------------------------------------------------------------------------
        ## get Raga parameters from input dictionary

        elif self.model == "Raga":

            ## general parameters
            self.epsilon = self.CalcParameters['epsilon']                                   ## Jet efficency
            self.gamma = self.CalcParameters['gamma']                                       ## Molecular gas
            self.M1 = self.CalcParameters['M1']                                             ## Mixing-layer sound speed velocity ratio

            ## outflow shape parameters
            self.beta = self.CalcParameters['beta']                                         ## Shape of the outflow 
            self.r0 = self.CalcParameters['r0'] * self.AU2Meter                             ## Cavity radius at the working surface
            self.z0 = self.CalcParameters['z0'] * self.AU2Meter                             ## Present position of the working surface

            ## initial distribution parameters
            self.vj = self.CalcParameters['vj']                                             ## initial jet velocity in km/s
            self.n0 = self.CalcParameters['n0'] / self.m3TOcm3                              ## initial density in m-3
            self.ne0 = self.nbackm3                                                         ## initial background density in m-3
            self.T1 = self.CalcParameters['T1']                                             ## inítial temperature of the mixing-layer in K

            if self.debug == True:
                print("\nepsilon = ", self.epsilon)
                print("gamma = ", self.gamma)
                print("M1 = ", self.M1)
                print("beta = ", self.beta)
                print("r0 = ", self.r0)
                print("z0 = ", self.z0)
                print("vj = ", self.vj)
                print("n0 = ", self.n0)
                print("ne0 = ", self.ne0)
                print("T1 = ", self.T1)

        self.npro = self.CalcParameters['npro']                                             ## get number of processes

        if self.debug == True:
            print("\nnpro = ", self.npro)

        ## we're done
        return
        ##----------------------------------------------------------------------------------------------------------------------------------------------------


    ##----------------------------------------------------------------------------------------------------------------------------------------------------
    ##
    ## define function to create the coordinate grid and rotate it
    ##
    def CreateCoordinateGrid(self):
        """

    input parameters:
    -----------------

        - None


    used parameters:
    ----------------
    
        - self.dxmax                                        maximal extend along x axis

        - self.dymax                                        maximal extend along z axis

        - self.dzmax                                        maximal extend along z axis

        - self.MaxXYSteps                                   maximal number of cells in x/y direction

        - self.MaxResolution                                maximal resolution in RA/DEC direction

        - self.DistanceToSourceMeter                        distance to the source

        - self.NumberOfCellsAlongLineOfSight                number of cells in z direction

        - self.ej                                           unit jet/outflow vector

        - self.dmax                                         outflow length

    return parameters:
    ------------------

        - self.resolution                                   stepsize along RA/DEC direction

        - self.xStepSize                                    stepsize along x axis

        - self.yStepSize                                    stepsize along y axis

        - self.zStepSize                                    stepsize along z axis

        - self.NumberXCoord                                 number of cells in x direction

        - self.NumberYCoord                                 number of cells in y direction

        - self.pcrit                                        reference point to compare the array results

        - self.XarrayUnRot                                  unrotated x-array

        - self.YarrayUnRot                                  unrotated y-array

        - self.ZarrayUnRot                                  unrotated z-array

        - self.Xarray                                       rotated x-array

        - self.Yarray                                       rotated y-array

        - self.Zarray                                       rotated z-array
        
        - self.d                                            distance to the source array

        - self.z                                            distance to source along z-axis array

        - self.r                                            radial distance to outflow axis array
        
        - self.theta                                        theta value of each point (for spherical coordinates)

        - self.phi                                          phi value of each point (for spherical coordinates)

    output parameters:
    ------------------

        - None
        """

        # Debug:
        if self.debug == True:
            print("dxmax, dymax, dxmax = %.3e, %.3e, %.3e [m]" %(self.dxmax, self.dymax, self.dzmax))
            print("dmax = %.3e [m]" %(self.dmax))
            print("Unit jet vector = %a" %(self.ej))
            print("Max steps x/y = %i" %(self.MaxXYSteps))
            print("Max resolution RA/DEC = %i [deg]" %(self.MaxResolution))
            print("Number of cells along line of sight = %.5e m" %(self.NumberOfCellsAlongLineOfSight))
            print("Distance to source = %.5e m" %(self.DistanceToSourceMeter))


        ## Calculate the resolution [in deg]
        res = 2 * numpy.arcsin(max(self.dxmax/2, self.dymax/2)/(self.DistanceToSourceMeter)) / self.MaxXYSteps * 180 / numpy.pi

        ## Comparing to the maximal resolution and taking the bigger one
        self.resolution = max(abs(res), abs(self.MaxResolution))

        if numpy.shape(self.resolution) != ():
            self.resolution = self.resolution[0]

        ## Calculate the number of steps in x and y direction
        self.NumberXCoord = int(numpy.ceil(2 * numpy.arcsin(self.dxmax/(2*self.DistanceToSourceMeter)) / (self.resolution*numpy.pi/180)))
        self.NumberYCoord = int(numpy.ceil(2 * numpy.arcsin(self.dymax/(2*self.DistanceToSourceMeter)) / (self.resolution*numpy.pi/180)))

        ## Calculating the step size in z direction
        self.zStepSize = self.dzmax / self.NumberOfCellsAlongLineOfSight

        ## define z coordinate of current grid cell center
        xKartIndex = numpy.linspace(0,self.NumberXCoord-1,self.NumberXCoord,dtype=int) - (self.NumberXCoord-1)/2
        yKartIndex = numpy.linspace(0,self.NumberYCoord-1,self.NumberYCoord,dtype=int) - (self.NumberYCoord-1)/2
        zKartIndex = numpy.linspace(0,self.NumberOfCellsAlongLineOfSight-1,self.NumberOfCellsAlongLineOfSight,dtype=int) - (self.NumberOfCellsAlongLineOfSight-1)/2

    
        ## define distance to observer
        self.DistanceObserverArray = self.DistanceToSourceMeter + zKartIndex * self.zStepSize


        ## Create coordinate arrays
        yi_mesh, xi_mesh, zi_mesh = numpy.meshgrid(yKartIndex, xKartIndex, zKartIndex)

        xarray = numpy.sin(xi_mesh * self.resolution*numpy.pi/180) * self.DistanceObserverArray
        yarray = numpy.sin(yi_mesh * self.resolution*numpy.pi/180) * self.DistanceObserverArray
        zarray = zi_mesh * self.zStepSize

        ## "Save" the original grid to self.
        self.XarrayUnRot = xarray
        self.YarrayUnRot = yarray
        self.ZarrayUnRot = zarray

        ## Calculate x and y step sizes, create them in arrays as they are in general not constant
        self.xStepSize = abs(numpy.sin((xi_mesh + 1/2) * self.resolution * numpy.pi / 180.0) - numpy.tan((xi_mesh - 1/2) * self.resolution * numpy.pi / 180.0)) * self.DistanceObserverArray
        self.yStepSize = abs(numpy.sin((yi_mesh + 1/2) * self.resolution * numpy.pi / 180.0) - numpy.tan((yi_mesh - 1/2) * self.resolution * numpy.pi / 180.0)) * self.DistanceObserverArray           


        ## Rotate coordinate grid
        self.Xarray = self.R[0,0]*xarray + self.R[0,1]*yarray + self.R[0,2]*zarray
        self.Yarray = self.R[1,0]*xarray + self.R[1,1]*yarray + self.R[1,2]*zarray
        self.Zarray = self.R[2,0]*xarray + self.R[2,1]*yarray + self.R[2,2]*zarray

        ## define reference pixel to compare array results
        self.pcrit = numpy.array(self.ej*self.dmax*0.25+
                              numpy.array([(self.NumberXCoord-1)/2,(self.NumberYCoord-1)/2,(self.NumberOfCellsAlongLineOfSight-1)/2]),
                              dtype=int)



        ## Calculate distances
        self.d = numpy.sqrt(self.Xarray**2+self.Yarray**2+self.Zarray**2)                   ## Distance to source
        self.r = numpy.sqrt(self.Xarray**2+self.Yarray**2)                                  ## Radius to outflow axis

        ## Avoiding division by zero in arccos
        d0i = numpy.where(self.d == 0)
        dcos = self.d
        dcos[d0i] = 1e-5

        ## Calculate spherical angels
        self.theta = numpy.arccos(self.Zarray/dcos)    
        self.phi = numpy.arctan2(self.Xarray, self.Yarray)

        ## debug:
        if self.debug == True:
            print("\nRA/DEC - resolution = %.3e (%.3e/%.3e)" %(self.resolution, res, self.MaxResolution))
            print("Number of x/y steps = %i/%i\n" %(self.NumberXCoord, res, self.NumberXCoord))
            print("pcrit = %a\n" %(self.pcrit))
            print("x/y/z - StepSize[pcrit] [m] = %.3e/%.3e/.3e\n" %(self.xStepSize[self.pcrit], self.yStepSize[self.pcrit], self.zStepSize))
            print("arrayUnRot[pcrit] [m] = %a" %(numpy.array([self.XarrayUnRot[self.pcrit], self.XarrayUnRot[self.pcrit], self.XarrayUnRot[self.pcrit]])))
            print("array[pcrit] [m] = %a" %(numpy.array([self.Xarray[self.pcrit], self.Xarray[self.pcrit], self.Xarray[self.pcrit]])))
            print("d/z [pcrit] [m] = %.3e/%.3e" %(self.d[self.pcrit], abs(self.Zarray[self.pcrit])))
            print("Spherical coordinates [pcrit] = %.3e, %.3f, %.3f (r, theta, phi)\n" %(self.r[self.pcrit], self.theta[self.pcrit], self.phi[self.pcrit]))

        ## we're done
        return
    
    ##----------------------------------------------------------------------------------------------------------------------------------------------------
    ##
    ## Rotate the coordinate grid along the x axis
    ##
    def rotx(self, x_ang):
        return numpy.array([[1,0,0],
                            [0,numpy.cos(x_ang),-numpy.sin(x_ang)],
                            [0,numpy.sin(x_ang),numpy.cos(x_ang)]])
    
    
    ##----------------------------------------------------------------------------------------------------------------------------------------------------

    
    ##----------------------------------------------------------------------------------------------------------------------------------------------------
    ##
    ## Rotate the coordinate grid along the y axis
    ##
    def roty(self, y_ang):
        return numpy.array([[numpy.cos(y_ang),0,numpy.sin(y_ang)],
                            [0,1,0],
                            [-numpy.sin(y_ang),0,numpy.cos(y_ang)]])
    
    
    ##----------------------------------------------------------------------------------------------------------------------------------------------------

    
    ##----------------------------------------------------------------------------------------------------------------------------------------------------
    ##
    ## Rotate the coordinate grid along the z axis
    ##
    def rotz(self, z_ang):
        return numpy.array([[numpy.cos(z_ang),-numpy.sin(z_ang),0],
                            [numpy.sin(z_ang),numpy.cos(z_ang),0],
                            [0,0,1]])

    ##----------------------------------------------------------------------------------------------------------------------------------------------------

    
    ##----------------------------------------------------------------------------------------------------------------------------------------------------
    ##
    ## Create the rotational matrix
    ##
    def RotationMatrix(self, x_ang, y_ang, z_ang):
        return numpy.matmul(numpy.matmul(self.rotz(z_ang), self.roty(y_ang)), self.rotx(x_ang))

    ##----------------------------------------------------------------------------------------------------------------------------------------------------

    
    ##----------------------------------------------------------------------------------------------------------------------------------------------------
    ##
    ## Solve the rotation matrix for the three rotation axis
    ##
    def SolveRotationMatrix(self, ang, theta, phi):
        return 1e10*(numpy.matmul(self.RotationMatrix(ang[0], ang[1], ang[2]), [0,0,1]) - self.PolarCoordinate(theta, phi))

    ##----------------------------------------------------------------------------------------------------------------------------------------------------

    
    ##----------------------------------------------------------------------------------------------------------------------------------------------------
    ##
    ## Calculate the polar coordinate
    ##
    def PolarCoordinate(self, theta, phi, r=1):
            return r*numpy.array([numpy.cos(phi)*numpy.sin(theta), numpy.sin(phi)*numpy.sin(theta), numpy.cos(theta)])


    ##----------------------------------------------------------------------------------------------------------------------------------------------------
    ##
    ## define function to compute a Gaussian 
    ##
    def Gaus(self, z, rmin):
        return self.A0 * numpy.sqrt(2/numpy.pi/self.sigma**2) * numpy.exp(-rmin**2/(2*self.sigma**2)) * (z/self.z0)**(-self.beta*5)
    
    
    ##----------------------------------------------------------------------------------------------------------------------------------------------------

    
    ##----------------------------------------------------------------------------------------------------------------------------------------------------
    ##
    ## define function to compute the environmental density
    ##
    def ne(self, z):
        return self.ne0 * (z/self.z0)**self.beta
    
    
    ##----------------------------------------------------------------------------------------------------------------------------------------------------


    ##----------------------------------------------------------------------------------------------------------------------------------------------------
    ##
    ## define function to compute the origin mixing layer density
    ##
    def n1(self):
        return self.epsilon * self.ne(abs(self.Zarray)) * 100
    
    
    ##----------------------------------------------------------------------------------------------------------------------------------------------------


    ##----------------------------------------------------------------------------------------------------------------------------------------------------
    ##
    ## define rotation matrix in polar coordinates
    ##
    def RotationMatrixAxis(self, theta, er):
        return numpy.array([[numpy.cos(theta) + er[0]**2*(1-numpy.cos(theta)), er[0]*er[1]*(1-numpy.cos(theta)) - er[2]*numpy.sin(theta), er[0]*er[2]*(1-numpy.cos(theta)) + er[1]*numpy.sin(theta)],
                         [er[1]*er[0]*(1-numpy.cos(theta)) + er[2]*numpy.sin(theta), numpy.cos(theta) + er[1]**2*(1-numpy.cos(theta)), er[1]*er[2]*(1-numpy.cos(theta)) - er[0]*numpy.sin(theta)],
                         [er[2]*er[0]*(1-numpy.cos(theta)) - er[1]*numpy.sin(theta), er[2]*er[1]*(1-numpy.cos(theta)) + er[0]*numpy.sin(theta), numpy.cos(theta) + er[2]**2*(1-numpy.cos(theta))]])
    
    
    ##----------------------------------------------------------------------------------------------------------------------------------------------------


    ##----------------------------------------------------------------------------------------------------------------------------------------------------
    ##
    ## define function to compute the shape of the outflow, only important for Raga model
    ##
    def ComputeOutflowShape(self):
        """
        TBD:
        - Modify this function so that it calculates the outflow ranges


    input parameters:
    -----------------

        - None

    used parameters:
    -----------------

        - CalcParameters["z0"]:                             distance of the working surface

        - CalcParameters["r0"]:                             radius at the working surface

        - CalcParameters["NumberOfCellsAlongLineOfSight"]:  numbers of cells along the line of sight
        
        - CalcParameters["beta"]:                           beta value --> shape of the outflow


    output parameters:
    ------------------

        - rc:                                               interpolated function to calculate the outflow radius at a given distance (Raga)
                                                            or return -1 (other models)
        """

        if self.model == "Raga":
 
            zi = max([self.NumberOfCellsAlongLineOfSight, 1000])

            # Debug:
            if self.CalcParameters["Debug"] == True:
                print ("\nz0 = ", self.z0)
                print ("r0 = ", self.r0)
                print ("NumberOfCellsAlongLineOfSight = ", self.NumberOfCellsAlongLineOfSight)
                print ("zi = ", zi)
                print ("beta = ", self.beta) 
        
            ## Define some local functions
            ##----------------------------------------------------------------------------------------------------------------------------------------------------
            ##
            ## define function to compute the shape of the outflow, is required for ComputeOutflowShape
            ##           
            def f(r, z, z0, beta, r0):
        
                # Finding the position on the z-axis in contrast to z0
                if z < z0:
                    # Returning the correct equation
                    return ((numpy.pi-numpy.arctan(r/(z0-z)))*r0*(z/z0)**(-beta/2) - r*numpy.pi/2)**2
        
                elif z == z0:
                    return (numpy.pi/2*r0*(z/z0)**(-beta/2) - r*numpy.pi/2)**2
            
                elif z > z0:
                    return (numpy.arctan(r/(z-z0))*r0*(z/z0)**(-beta/2) - r*numpy.pi/2)**2
        
            ##----------------------------------------------------------------------------------------------------------------------------------------------------


            ##----------------------------------------------------------------------------------------------------------------------------------------------------
            ##
            ## define function to calculate the outflows close limit, is required for ComputeOutflowShape
            ##

            def fclose(r, z, z0, beta, r0):
    
                if z < z0:
                    return (numpy.pi*r0*(z/z0)**(-beta/2) - r*numpy.pi/2)**2
        
                elif z == z0:
                    return (numpy.pi*r0*(z/z0)**(-beta/2) - r*numpy.pi/2)**2
            
                elif z > z0:
                    return (numpy.pi*r0*(z/z0)**(-beta/2) - r*numpy.pi/2)**2
        
            ##----------------------------------------------------------------------------------------------------------------------------------------------------


            ##----------------------------------------------------------------------------------------------------------------------------------------------------
            ##
            ## define function to calculate the outflows far limit, is required for ComputeOutflowShape
            ##

            def ffar(r, z, z0, r0):
    
                if z < z0:
                    return ((numpy.pi-numpy.arctan(r/(z0-z)))*r0 - r*numpy.pi/2)**2
        
                elif z == z0:
                    return (numpy.pi/2*r0 - r*numpy.pi/2)**2
            
                elif z > z0:
                    return (numpy.arctan(r/(z-z0))*r0 - r*numpy.pi/2)**2
        
            ##----------------------------------------------------------------------------------------------------------------------------------------------------


            ##----------------------------------------------------------------------------------------------------------------------------------------------------
            ##
            ## define function to calculate the outflows far limit, is required for ComputeOutflowShape
            ##

            def rmaxi(d, rc, ej, R, com):

                if com == "x":
                    er = numpy.array([1,0,0])
                    en = 0
                elif com == "y":
                    er = numpy.array([0,1,0])
                    en = 1
                elif com == "z":
                    er = numpy.array([0,0,1])
                    en = 2

                if 1 - abs(numpy.dot(ej,er)/(numpy.linalg.norm(er)*numpy.linalg.norm(ej))) <= 1e-3:  
                    err = numpy.array([0,0,0])
                else:
                    err = numpy.matmul(R, er)

                    err /= numpy.linalg.norm(err)

                return (-(numpy.matmul(rc(d).reshape(-1,1),err.reshape(1,-1))+numpy.matmul(d.reshape(-1,1),ej.reshape(1,-1))).T[en])
        
            ##----------------------------------------------------------------------------------------------------------------------------------------------------

            ##================================================================================================================================================
            ## compute outflow shape
            ## define function for source size here (make sure that result is in arcsec)
            ##
            ## Coordinates to calculate the outflow shape
            if self.plot == True:
                zs = numpy.linspace(0, 1.5*self.z0, 1000)
    
                ## First guesses for the solver
                guess = 0
                guessh = 0
                guess1 = 0
                guess2 = 0
    
                ## Some values for beta
                betas = [-0.5,-2,-4]
    
                if not self.beta in betas:
                    betas = sorted(numpy.append(betas, [self.beta]), reverse=True)
    
                ## Creating a figure
                fig, ax = matplotlib.pyplot.subplots(len(betas),figsize=(len(betas)*5,15))
    
                ## Looping over all beta values
                for i, betai in enumerate(betas):
        
                    # Empty root arrays
                    rl = numpy.zeros_like(zs)
                    rlh = numpy.empty_like(zs)
                    rl1 = numpy.zeros_like(zs)
                    rl2 = numpy.zeros_like(zs)
        
                    ## Looping over each position
                    for ii, zi in enumerate(zs): 
            
                        # Finding the local minima
                        sol = minimize(f, guess, args=(zi / self.r0, self.z0 / self.r0, betai, self.r0 / self.r0)) # , method='trust-krylov')
                        solh = minimize(f, guessh, args=(zi / self.r0, self.z0/2 / self.r0, betai, self.r0 / self.r0)) # , method='trust-krylov')
                        sol1 = minimize(fclose, guess1, args=(zi / self.r0, self.z0 / self.r0, betai, self.r0 / self.r0)) # , method='trust-krylov')
                        sol2 = minimize(ffar, guess2, args=(zi / self.r0, self.z0 / self.r0, self.r0 / self.r0)) # , method='trust-krylov')
        
                        rs = sol.x
                        rsh = solh.x
                        rs1 = sol1.x
                        rs2 = sol2.x
            
                        # Updating the first guess for the next iteration
                        guess = rs
                        guessh = rsh
                        guess1 = rs1
                        guess2 = rs2
            
                        # Writing the minima/root to an array
                        rl[ii] = rs * self.r0 if (zi <= self.z0 or rs >= 1e-5) else -self.r0
                        rlh[ii] = rsh * self.r0 if (zi <= self.z0 or rsh >= 1e-5) else -self.r0
                        rl1[ii] = rs1 * self.r0 if (zi <= self.z0 or rs1 >= 1e-5) else -self.r0
                        rl2[ii] = rs2 * self.r0 if (zi <= self.z0 or rs2 >= 1e-5) else -self.r0
            
            
                    # Plotting the results
                    if 1e-3 <= self.z0 / self.r0 <= 1e3:
                        ax[i].plot(zs/self.r0, rl/self.r0, label="z0 = %.3f" %(self.z0 / self.r0))
                        ax[i].plot(zs/self.r0, rlh/self.r0, label="z0 = %.3f" %(self.z0/2 / self.r0), ls="--")

                    else:
                        ax[i].plot(zs/self.r0, rl/self.r0, label="z0 = %.3e" %(self.z0 / self.r0))
                        ax[i].plot(zs/self.r0, rlh/self.r0, label="z0 = %.3e" %(self.z0/2 / self.r0), ls="--")

                    ax[i].plot(zs/self.r0, rl1/self.r0, label="close")
                    ax[i].plot(zs/self.r0, rl2/self.r0, label="far")

                    ax[i].grid()
                    ax[i].set_xlabel(r"Distance z/r$_0$")
                    ax[i].set_ylabel(r"Distance r/r$_0$")
                    ax[i].set_ylim(0,3)
                    ax[i].set_title(r"$\beta$ = %.1f; z0 = %.2e" %(betai, self.z0))
                    ax[i].legend()
        
                    ## Buffering the important beta values
                    if betai == self.beta:
                        ofs = rl
    
                try:
                    matplotlib.pyplot.savefig(self.CalcParameters['PlotPath'] + 'Raga_Shape_beta-%.2f.pdf' %(self.beta),
                            transparent=True, pad_inches=0.0, orientation='portrait', format="pdf")
           
                except Exception as e:
                    exception_type, exception_object, exception_traceback = sys.exc_info()
                    filename = exception_traceback.tb_frame.f_code.co_filename
                    line_number = exception_traceback.tb_lineno
        
                    print("\nSaving the figure failed. Maybe it's still open somewhere.")
                    print("Exception type:\t%s" %(exception_type))
                    print("File name:\t%s" %(filename))
                    print("Line number:\t%s" %(line_number))
                    print("The error itself:\n%s\n\n" %(e))

                matplotlib.pyplot.close()

            else:
                zs = numpy.linspace(0, 1.5*self.z0, 1000)
    
                ## First guesses for the solver
                guess = 0
            
                # Empty root arrays
                ofs = numpy.zeros_like(zs)
        
                ## Looping over each position
                for ii, zi in enumerate(zs): 
            
                    # Finding the local minima
                    sol = minimize(f, guess, args=(zi / self.r0, self.z0 / self.r0, self.beta, self.r0 / self.r0)) # , method='trust-krylov')
                    rs = sol.x

                    # Updating the first guess for the next iteration
                    guess = rs

            
                    # Writing the minima/root to an array
                    ofs[ii] = rs * self.r0 if (zi <= self.z0 or rs >= 1e-5) else -self.r0
          

            ## expand the radii and offsets in the other direction, symmetrical problem
            zsd = numpy.append(-numpy.flip(zs[1:]), zs)
            ofsd = numpy.append(numpy.flip(ofs[1:]), ofs)

            ## interpolate the grid
            rc = interp1d(zsd, ofsd, kind='cubic', bounds_error=False, fill_value=-1e-5)
            rcr0 = interp1d(zsd/self.r0, ofsd/self.r0, kind='cubic', bounds_error=False, fill_value=-1e-5)

            ## Get the z outflow extend in units of r0
            zoi = numpy.where(ofs >= 1e-5)[0][-1] + 1
            zov = zs[zoi]
            zovr0 = zov / self.r0

            ## Simulate smooth transition for outflow head
            sigma = (zov - self.z0) / (4*numpy.sqrt(2*numpy.log(2)))
            A0 = self.n0 / (numpy.sqrt(2/numpy.pi/sigma**2) * (zov/self.z0)**(-5*self.beta))


            ## Buffer the important results
            self.rc = rc
            self.A0 = A0
            self.sigma = sigma
            self.dmax = zov

            ## calculate maximal outflow extend in each direction
            dmax1 = abs(minimize(rmaxi, x0=-zovr0/2, args=(rcr0, self.ej, self.R, "x"), bounds=((-zovr0,0),)).fun)
            dmax2 = abs(minimize(rmaxi, x0=zovr0/2, args=(rcr0, self.ej, self.R, "x"), bounds=((0,zovr0),)).fun)
            dmay1 = abs(minimize(rmaxi, x0=-zovr0/2, args=(rcr0, self.ej, self.R, "y"), bounds=((-zovr0,0),)).fun)
            dmay2 = abs(minimize(rmaxi, x0=zovr0/2, args=(rcr0, self.ej, self.R, "y"), bounds=((0,zovr0),)).fun)
            dmaz1 = abs(minimize(rmaxi, x0=-zovr0/2, args=(rcr0, self.ej, self.R, "z"), bounds=((-zovr0,0),)).fun)
            dmaz2 = abs(minimize(rmaxi, x0=zovr0/2, args=(rcr0, self.ej, self.R, "z"), bounds=((0,zovr0),)).fun)

            ## Buffer results
            self.dxmax = max(dmax1, dmax2) * self.r0 * 2.2
            self.dymax = max(dmay1, dmay2) * self.r0 * 2.2
            self.dzmax = max(dmaz1, dmaz2) * self.r0 * 2.2


        
            ##
            ##================================================================================================================================================

            if self.debug == True:
                print("\nrc = ", self.rc)
                print("A0 = ", self.A0)
                print("sigmal = ", self.sigma)
                print("zov = ", zov)
                print("z0 = ", self.z0)
                print("n0 = ", self.n0)
                print("beta = ", self.beta)
                print("dxmax = %.3e" %(self.dxmax))
                print("dymax = %.3e" %(self.dymax))
                print("dzmax = %.3e" %(self.dzmax))

        else:
            ########

            ## Find the points in the spherical cap of a unit sphere with the largest amount in x/y/z direction

            ## Calculate the angle of the jet vector to the axis
            angjx = numpy.arccos(numpy.dot(self.ej, numpy.array([1,0,0]).T))
            angjy = numpy.arccos(numpy.dot(self.ej, numpy.array([0,1,0]).T))
            angjz = numpy.arccos(numpy.dot(self.ej, numpy.array([0,0,1]).T))

            ## Spherical cap
            a = numpy.sin(self.thetamax)
            h = 1-numpy.cos(self.thetamax)

            ## Generate h-vector and two orthogonal (to ej) ones
            vh = (1-h)*self.ej

            va = a*numpy.array([numpy.sin(self.ThetaJet+numpy.pi/2)*numpy.cos(self.PhiJet),
                                numpy.sin(self.ThetaJet+numpy.pi/2)*numpy.sin(self.PhiJet),
                                numpy.cos(self.ThetaJet+numpy.pi/2)])

            vb = numpy.cross(self.ej, va)
            vb *= a/numpy.linalg.norm(vb)


            ## Function that describes the ring of the spherical cap
            def SphericalSegment(x, axis="point"):
                if type(axis) == int and 0<=axis<len(va):
                    ss = vh[axis] + va[axis]*numpy.cos(x) + vb[axis]*numpy.sin(x)
                    return -abs(ss)
                else:
                    ss = vh + va*numpy.cos(x) + vb*numpy.sin(x)
                    return ss


            ## find the maximal distance in ...
            ## x direction
            ## Check if the ex axis is inside the spherical cap
            if angjx <= self.thetamax:
                ## if yes, the maximal distance lies on the axis
                pox = numpy.array([1,0,0])

            else:
                ## if not, it is on the outer ring; solve for the maximal distance
                solx = minimize(SphericalSegment, 0, args=(0)) # , method='trust-krylov')
                ssx = solx.x

                ## Check if the result is not the minima; it is shifted by pi relative to the maxima
                if abs(SphericalSegment(ssx)[0]) >= abs(SphericalSegment(ssx+numpy.pi)[0]):
                    pox = SphericalSegment(ssx)
                else:
                    pox = SphericalSegment(ssx+numpy.pi)
    

            ## y direction
            if angjy <= self.thetamax:
                poy = numpy.array([0,1,0])

            else:
                soly = minimize(SphericalSegment, 0, args=(1)) # , method='trust-krylov')
                ssy = soly.x
                #poy = SphericalSegment(ssy)
                if abs(SphericalSegment(ssy)[1]) >= abs(SphericalSegment(ssy+numpy.pi)[1]):
                    poy = SphericalSegment(ssy)
                else:
                    poy = SphericalSegment(ssy+numpy.pi)

            ## z direction
            if angjz <= self.thetamax:
                poz = numpy.array([0,0,1])

            else:
                solz = minimize(SphericalSegment, 0, args=(2)) # , method='trust-krylov')
                ssz = solz.x
                #poz = SphericalSegment(ssz)
                if abs(SphericalSegment(ssz)[2]) >= abs(SphericalSegment(ssz+numpy.pi)[2]):
                    poz = SphericalSegment(ssz)
                else:
                    poz = SphericalSegment(ssz+numpy.pi)


            ## Transfering it to the outflow size and enlarge the ranges by 1.1
            self.dxmax = 2 * abs(pox[0]) * self.dmax * 1.1
            self.dymax = 2 * abs(poy[1]) * self.dmax * 1.1
            self.dzmax = 2 * abs(poz[2]) * self.dmax * 1.1

            if self.debug == True:
                print("dxmax = %.3e" %(self.dxmax))
                print("dymax = %.3e" %(self.dymax))
                print("dzmax = %.3e" %(self.dzmax))

            def rc(z):
                return -1

            self.rc = rc
            self.A0 = 0
            self.sigma = 0

        ## we're done
        return
    
    
    ##----------------------------------------------------------------------------------------------------------------------------------------------------


    ##----------------------------------------------------------------------------------------------------------------------------------------------------
    ##
    ## prepare ComputeOutflowShape function by initiating multiprocessing, only important for Raga model
    ##
    def ComputeX1AndDistance(self, xarray, yarray, zarray, prepinter=False):
        """

    input parameters:
    -----------------

        - xarray:                      1-D position x-array

        - yarray:                      1-D position y-array

        - zarray:                      1-D position z-array

    
    used parameters:
    -----------------


    output parameters:
    ------------------

        - x1array:                      origin of the molecules in absolute distance; 1-D array

        - rminarray:                    distance to the outflow surface; 1-D array
        """

        print("Start computing X1 and rmin at %s." %(str(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))))

        ## Define some local functions
        ##----------------------------------------------------------------------------------------------------------------------------------------------------
        ##
        ## define function to compute the z origin of the material
        ##
        def funx1(z, zi, ri, rc, M1):
            return (z - zi + (rc(z) - ri) * M1)**2
    
        ##----------------------------------------------------------------------------------------------------------------------------------------------------

    
        ##----------------------------------------------------------------------------------------------------------------------------------------------------
        ##
        ## define function to compute the distance between the point and surface
        ##

        def dps(z, zi, ri, rc):
            return ((rc(z) - ri)**2 + (z - zi)**2)**2
    
    
        ##----------------------------------------------------------------------------------------------------------------------------------------------------

        # Debug:
        if self.debug == True and zi == abs(self.Zarray[self.pcrit]) and r == self.r[self.pcrit] and self.model == "Raga":
            print ("\nz = ", z)
            print ("r = ", r)
            print ("rc(z) = ", self.rc(z))
            print ("z0 = ", self.z0)
            print ("M1 = ", self.M1)


        ## Ensure that the input data has the right format
        if numpy.shape(xarray) != numpy.shape(xarray.reshape(-1)):
            xarray = xarray.reshape(-1)

        if numpy.shape(yarray) != numpy.shape(yarray.reshape(-1)):
            yarray = yarray.reshape(-1)

        if numpy.shape(zarray) != numpy.shape(zarray.reshape(-1)):
            zarray = zarray.reshape(-1)
            
        ## if neccessary, generate the interpolations in the preperation interpolation mode
        if prepinter == True:

            ## generate grids and extract points from them
            x = numpy.linspace(-self.dxmax,self.dxmax,self.NumberXCoord).reshape(-1)
            y = numpy.linspace(-self.dymax,self.dymax,self.NumberYCoord).reshape(-1)
            z = numpy.linspace(-self.dzmax,self.dzmax,self.NumberOfCellsAlongLineOfSight).reshape(-1)
            points = (x, y, z)
            
            ## create meshgrid from the data and reshape the cubes to 1d
            xa, ya, za = numpy.meshgrid(x, y, z, indexing='ij')

            xa = xa.reshape(-1)
            ya = ya.reshape(-1)
            za = za.reshape(-1)

            points_grid = (xa, ya, za)

            ## create empty arrays
            values_X1 = numpy.zeros_like(xa)
            values_rmin = numpy.zeros_like(xa)

            ## Calculate X1 and rmin for each point
            for i, (xi, yi, zi) in enumerate(zip(xa, ya, za)):
                
                ## Radius
                r = numpy.sqrt(xi**2 + yi**2)

                ##================================================================================================================================================
                ## compute the molecule origin and distance to the working surface
                ##

                ## Check if the pixel is inside the outflow region and then solve the equation
                if r <= self.rc(abs(zi)):
        
                    solx1 = minimize(funx1, abs(zi), args=(abs(zi), r, self.rc, self.M1), bounds=numpy.array([[0,1.5*self.z0]]), method="Powell")
                    solxdm = minimize(dps, 0.1, args=(abs(zi), r, self.rc), bounds=numpy.array([[0,1.5*self.z0]]), method="Powell")

                    values_X1[i] = solx1.x[0]
                    xdmi = solxdm.x[0]
                    values_rmin[i] = numpy.sqrt((self.rc(xdmi) - r)**2 + (xdmi - abs(zi))**2)
        
                ## if not, set the origin to zero and the distance to inf
                else:
                    values_X1[i] = 0
                    values_rmin[i] = numpy.inf

                ##
                ##================================================================================================================================================
                

            ## Save X1 and rmin arrays to fits files
            X1ArraySwapped = numpy.swapaxes(values_X1.reshape(self.NumberXCoord, self.NumberYCoord, self.NumberOfCellsAlongLineOfSight)*1., 2, 0)

            X1Name = "X1.fits"

            hdu = fits.PrimaryHDU(numpy.array(X1ArraySwapped))
            hdu.writeto(self.CalcParameters['CubePath']+X1Name, overwrite=True)

            rminArraySwapped = numpy.swapaxes(values_rmin.reshape(self.NumberXCoord, self.NumberYCoord, self.NumberOfCellsAlongLineOfSight)*1., 2, 0)

            rminName = "rmin.fits"

            hdu = fits.PrimaryHDU(numpy.array(rminArraySwapped))
            hdu.writeto(self.CalcParameters['CubePath']+rminName, overwrite=True)

            ## interpolate the results
            self.x1Inter = RegularGridInterpolator(points=points, values=values_X1.reshape(self.NumberXCoord, self.NumberYCoord, self.NumberOfCellsAlongLineOfSight),
                                                   method="linear", bounds_error=False, fill_value=0)
            self.rminInter = RegularGridInterpolator(points=points, values=values_rmin.reshape(self.NumberXCoord, self.NumberYCoord, self.NumberOfCellsAlongLineOfSight),
                                                     method="linear", bounds_error=False, fill_value=numpy.inf)
        
            ## Interpolate the arrays and save them as fits files to better estimate the interpolations
            x1array_full = self.x1Inter(points_grid)
            rminarray_full = self.rminInter(points_grid)

            X1ArraySwapped = numpy.swapaxes(x1array_full.reshape(self.NumberXCoord, self.NumberYCoord, self.NumberOfCellsAlongLineOfSight)*1., 2, 0)

            X1Name = "X1_full.fits"

            hdu = fits.PrimaryHDU(numpy.array(X1ArraySwapped))
            hdu.writeto(self.CalcParameters['CubePath']+X1Name, overwrite=True)

            rminArraySwapped = numpy.swapaxes(rminarray_full.reshape(self.NumberXCoord, self.NumberYCoord, self.NumberOfCellsAlongLineOfSight)*1., 2, 0)

            rminName = "rmin_full.fits"

            hdu = fits.PrimaryHDU(numpy.array(rminArraySwapped))
            hdu.writeto(self.CalcParameters['CubePath']+rminName, overwrite=True)

        ## get the interpolated results for the input arrays
        pts = numpy.array([xarray, yarray, zarray]).T
        x1array = self.x1Inter(pts)
        rminarray = self.rminInter(pts)
        
        # Debug:
        if self.debug == True and zi == abs(self.Zarray[self.pcrit]) and r == self.r[self.pcrit] and self.model == "Raga":
            print ("x1 = ", x1)
            print ("rmin = ", rmin)
            print("funx1(x1) = ", funx1(x1, abs(z), r, self.rc, self.M1))
            print("funx1(x1-1e-3) = ", funx1(x1-1e-3, abs(z), r, self.rc, self.M1))
            print("funx1(x1+1e-3) = ", funx1(x1+1e-3, abs(z), r, self.rc, self.M1))

        return x1array, rminarray

    ##--------------------------------------------------------------------------------------------------------------------------------------------------------




    ##----------------------------------------------------------------------------------------------------------------------------------------------------
    ##
    ## prepare ComputeOutflowShape function by initiating multiprocessing, only important for Raga model
    ##
    def ComputeX1AndDistanceOld(self, xarray, yarray, zarray):
        """

    input parameters:
    -----------------

        - xarray:                      1-D position x-array

        - yarray:                      1-D position y-array

        - zarray:                      1-D position z-array

    
    used parameters:
    -----------------


    output parameters:
    ------------------

        - x1array:                      origin of the molecules in absolute distance; 1-D array

        - rminarray:                    distance to the outflow surface; 1-D array
        """

        ## create empty arrays
        x1array = numpy.zeros_like(xarray)
        rminarray = numpy.zeros_like(xarray)


        ## Adapt the used number of cores
        npro = min([self.npro, len(xarray)])
        
        ## Create two empty queues
        xyzqueue = mp.Queue()        ## Hand in for the positions
        resqueue= mp.Queue()         ## Return the results
     
        ##----------------------------------------------------------------------------------------------------------------------------------------------------
        ## Get all x-y combinations in the queues
        for i, [xi, yi, zi] in enumerate(zip(xarray, yarray, zarray)):
            xyzqueue.put([xi, yi, zi, i])

        ## Creating processes
        if self.debug == True:
            print("Creating %i processes." %(npro))

        processes = [mp.Process(target=self.CallComputeX1AndDistanceMulti, args=(xyzqueue, resqueue)) for ip in range(npro)]

        for p in processes:
            p.start()

        if self.debug == True:
            print("Let's go!!!!")

        ## For time measurement and expected run time
        tst = time.time()

        ## Number of total and done iteration
        ittotal = len(xarray)
        idonep = -1
        idone = 0

        print(ittotal)
         
        ## Iterate until all pixels are handled 
        while idone != ittotal:
    
            idone = resqueue.qsize()
        
            ## Run time and predicted run time
            if idone != idonep and self.debug == True:
                tc = time.time()
                th = numpy.floor((tc-tst)/3600)
                tm = numpy.floor((tc-tst-3600*th)/60)
                ts = tc-tst-3600*th-tm*60
            
                tp = (tc-tst) * ittotal/idone if not idone == 0 else 0
                tph = numpy.floor((tp)/3600)
                tpm = numpy.floor((tp-3600*tph)/60)
                tps = tp-3600*tph-tpm*60
            
                print("Iterations done: %i of %i (%.2f%%)\nIterating time: %ih %2.imin %2.2fs\nPredicted run time: %ih %2.imin %2.2fs\n"
                        %(idone, ittotal, idone/ittotal*100, th, tm, ts, tph, tpm, tps))    
    
                idonep = idone
                
            time.sleep(5)
            
    
        if self.debug == True:
            print("\nIterations done: %i of %i (%.2f%%)" %(idone, ittotal, idone/ittotal*100))
            print("Iterating time: %ih %2.imin %2.2fs\n\n" %(th, tm, ts))

        ## Write the output to arrays
        for i in range(ittotal):
            [i, x1i, rmini] = resqueue.get()
        
            x1array[i] = x1i
            rminarray[i] = rmini 

        
        ## Ending all processes
        if self.debug == True:
            print("Stopping all progresses")
    
        for p in processes:

            if self.debug == True:
                print("Ending %s" %(p.name), end = "\r")
        
            p.join()

            if self.debug == True:
                 print("%s is done!" %(p.name), end = "\r")
        
        return x1array, rminarray

    ##--------------------------------------------------------------------------------------------------------------------------------------------------------

    def CallComputeX1AndDistanceMulti(self, xyzqueue, resqueue):
        """

    input parameters:
    -----------------

        - xyzqueue:                     queue that holds the input positions

        - resqueue:                     queue to buffer x1 and rmin results --> output queue

    
    used parameters:
    -----------------


    output parameters:
    ------------------


        """
        while xyzqueue.empty() != True:
        
            ## Get current pixel
            xi, yi, zi, i = xyzqueue.get()

            ## pixel array
            pos = [xi, yi, zi]
    
            ## call ComputeX1AndDistanceMulti function
            x1, rmin = self.ComputeX1AndDistanceMulti(pos)

            resqueue.put([i, x1, rmin])

        return 



    ##----------------------------------------------------------------------------------------------------------------------------------------------------
    ##
    ## define function to compute the x1 and outflow distance, only importnt for Raga model
    ##
    def ComputeX1AndDistanceMulti(self, pos):
        """

    input parameters:
    -----------------

        - pos:                          position array [xi, yi, zi] of the analysed voxel

    
    used parameters:
    -----------------

        - self.M1:                      Mixing-layer sound speed velocity ratio

        - self.rc:                      function to calculate the outflow radius at a given distance


    output parameters:
    ------------------

        - x1:                           origin of the molecules in absolute distance

        - rmin:                         distance to the outflow surface
        """

        ## get the positions of the to be analysed voxed 
        xi, yi, zi = pos

        z = zi
        r = numpy.sqrt(xi**2 + yi**2)

        # Debug:
        if self.debug == True and zi == abs(self.Zarray[self.pcrit]) and r == self.r[self.pcrit] and self.model == "Raga":
            print ("\nz = ", z)
            print ("r = ", r)
            print ("rc(z) = ", self.rc(z))
            print ("z0 = ", self.z0)
            print ("M1 = ", self.M1)

        
        ## Define some local functions
        ##----------------------------------------------------------------------------------------------------------------------------------------------------
        ##
        ## define function to compute the z origin of the material
        ##
        def funx1(z, zi, ri, rc, M1):
            return (z - zi + (rc(z) - ri) * M1)**2
    
        ##----------------------------------------------------------------------------------------------------------------------------------------------------

    
        ##----------------------------------------------------------------------------------------------------------------------------------------------------
        ##
        ## define function to compute the distance between the point and surface
        ##

        def dps(z, zi, ri, rc):
            return ((rc(z) - ri)**2 + (z - zi)**2)**2
    
    
        ##----------------------------------------------------------------------------------------------------------------------------------------------------



        ##================================================================================================================================================
        ## compute the molecule origin and distance to the working surface
        ##

        ## Check if the pixel is inside the outflow region and then solve the equation
        if r <= self.rc(abs(z)):
        
            solx1 = minimize(funx1, abs(z), args=(abs(z), r, self.rc, self.M1), bounds=numpy.array([[0,1.5*self.z0]]), method="Powell")
            solxdm = minimize(dps, 0.1, args=(abs(z), r, self.rc), bounds=numpy.array([[0,1.5*self.z0]]), method="Powell")

            x1 = solx1.x[0]
            xdmi = solxdm.x[0]
            rmin = numpy.sqrt((self.rc(xdmi) - r)**2 + (xdmi - abs(z))**2)
        
        ## if not, set the origin to zero and the distance to inf
        else:
            x1 = 0
            rmin = numpy.inf
        ##
        ##================================================================================================================================================

        # Debug:
        if self.debug == True and zi == abs(self.Zarray[self.pcrit]) and r == self.r[self.pcrit] and self.model == "Raga":
            print ("x1 = ", x1)
            print ("rmin = ", rmin)
            print("funx1(x1) = ", funx1(x1, abs(z), r, self.rc, self.M1))
            print("funx1(x1-1e-3) = ", funx1(x1-1e-3, abs(z), r, self.rc, self.M1))
            print("funx1(x1+1e-3) = ", funx1(x1+1e-3, abs(z), r, self.rc, self.M1))


        ## we're done
        return x1, rmin
    ##----------------------------------------------------------------------------------------------------------------------------------------------------

    
    ##----------------------------------------------------------------------------------------------------------------------------------------------------
    ##
    ## define function to compute outflow mask
    ##
    def ComputeMask(self, Xarray=None, Yarray=None, Zarray=None, r=None, d=None, theta=None, full=True):
        """

    input parameters:
    -----------------

        - None
        
    
    used parameters:
    -----------------

        - self.z (Raga):                distance to the source along the outflow / z-axis

        - self.r (Raga):                distance to the outflow / z-axis

        - self.rc (Raga):               interpolated function to calculate the outflow radius at a given distance

        - self.d (Cabrit):              distance to the source

        - self.dmin (Cabrit):           minimal outflow radius

        - self.dmax (Cabrit):           maximal outflow radius

        - self.thetamax (Cabrit):       outflow opening angle


    output parameters:
    ------------------

        - self.mask:                    outflow mask
        """

        # Debug:
        if self.debug == True and full == True:
            print("\nModel = ", self.model)
            print ("pcrit = ", self.pcrit)
            print ("x[pcrit] = ", self.Xarray[self.pcrit])
            print ("y[pcrit] = ", self.Yarray[self.pcrit])
            print ("z[pcrit] = ", self.Zarray[self.pcrit])

            if self.model == "Raga":
                print("r[pcrite] = ", self.r[self.pcrit])



            elif self.model == "Cabrit":
                print ("dmin = ", self.dmin)
                print ("dmax = ", self.dmax)
                print("d[pcrit] = ", self.d[self.pcrit])
                print("dmin <= d[pcrit] <= dmax: ", self.dmin <= self.d[self.pcrit] <= self.dmax)
                print ("thetamax = ", self.thetamax)
                print("theta[pcrite] = ", self.theta[self.pcrit])


        if self.model == "Raga":

            ## Generate the outflow mask depending on r and rc(z)
            maskl = numpy.less_equal(r*0, r)
            maskg = numpy.greater_equal(r, self.rc(abs(Zarray)))

            mask = numpy.logical_and(maskl, maskg) 

            MaskArraySwapped = numpy.swapaxes(mask*1., 2, 0)

            ## Save the cube as fits cube
            if full == True:
                MaskName = "Mask.fits"

                hdu = fits.PrimaryHDU(numpy.array(MaskArraySwapped))
                hdu.writeto(self.CalcParameters['CubePath']+MaskName, overwrite=True)

            ## prepare X1 and rmin calculation
            ## Get array shape and number of points
            shapexyz = numpy.shape(Xarray)
            ni = numpy.array(shapexyz).prod()

            ## Reshape arrays
            xrs = Xarray.reshape(ni,)
            yrs = Yarray.reshape(ni,)
            zrs = Zarray.reshape(ni,)
            mrs = mask.reshape(ni,)

            ## Get outflow voxel indices and their x/y/z values
            index = numpy.where(mrs == 0)
            
            xrsa = xrs[index]
            yrsa = yrs[index]
            zrsa = zrs[index]

            ## compute X1 and rmin to return them
            x1sa, rminsa = self.ComputeX1AndDistance(xrsa, yrsa, zrsa, prepinter=True)

            x1s = numpy.zeros_like(xrs) + numpy.inf
            rmins = numpy.zeros_like(xrs) + numpy.inf

            x1s[index] = x1sa
            rmins[index] = rminsa

            x1 = x1s.reshape(shapexyz)
            rmin = rmins.reshape(shapexyz)

            
            if self.debug == True:
                print ("\npcrit = ", self.pcrit)
                print("x1[pcrit] = ", self.x1[self.pcrit])
                print ("rmin[pcrit] = ", self.rmin[self.pcrit])
                print ("shape(x1) = ", numpy.shape(self.x1))


        elif self.model == "Cabrit":
            ## mask based on distance
            #maskd = ma.masked_outside(d, self.dmin, self.dmax).mask
            maskdo = ma.masked_outside(d, 0, self.dmax).mask    # Mask outside of outflow
            maskd0 = ma.masked_equal(d, 0).mask                 # Mask pole at d = 0
            maskd = numpy.logical_or(maskdo, maskd0)

            ## mask based on theta angle
            maskt = ma.masked_inside(theta, self.thetamax, numpy.pi-self.thetamax).mask

            ## Combine both masks
            mask = numpy.logical_or(maskd, maskt)

            ## Empty x1 and rmin arrays
            x1, rmin = numpy.zeros_like(mask), numpy.zeros_like(mask)

        else:
            mask = numpy.zeros_like(d)

        maskindex = numpy.where(mask == 0)
        

        if self.debug:
            print("\nshape(mask) = ", numpy.shape(self.mask))
            print("mask[pcrit] = ", self.mask[self.pcrit])
            print("maskindex[0] = ", (self.maskindex[0][0], self.maskindex[1][0], self.maskindex[2][0]))
            print("mask[maskindex[0]] = ", self.mask[self.maskindex[0][0], self.maskindex[1][0], self.maskindex[2][0]])
            print("pcrit is in mask = ", 0 in (~(numpy.array(self.maskindex).T == self.pcrit)).sum(axis=1))

        return mask, maskindex, x1, rmin

    ##----------------------------------------------------------------------------------------------------------------------------------------------------

    ##----------------------------------------------------------------------------------------------------------------------------------------------------
    ##
    ## define function to compute outflow mask
    ##
    def ComputeMaskLight(self):
        """

    input parameters:
    -----------------

        - None
        
    
    used parameters:
    -----------------

        - self.z (Raga):                distance to the source along the outflow / z-axis

        - self.r (Raga):                distance to the outflow / z-axis

        - self.rc (Raga):               interpolated function to calculate the outflow radius at a given distance

        - self.d (Cabrit):              distance to the source

        - self.dmin (Cabrit):           minimal outflow radius

        - self.dmax (Cabrit):           maximal outflow radius

        - self.thetamax (Cabrit):       outflow opening angle


    output parameters:
    ------------------

        - self.mask:                    outflow mask
        """


        if self.model == "Raga":


            ## compute the outflow shape, mass origin and shortest distance to the outflow border, just necessary for Raga model
            self.ComputeOutflowShape()


            ## mask where the distance to the outflow axis is bigger than predicted
            maskl = numpy.less_equal(self.r*0, self.r)
            maskg = numpy.greater_equal(self.r, self.rc(abs(self.Zarray)))

            self.mask = numpy.logical_and(maskl, maskg)

            ## prepare X1 and rmin calculation
            shapexyz = numpy.shape(self.Xarray)
            ni = numpy.array(shapexyz).prod()

            xrs = self.Xarray.reshape(ni,)
            yrs = self.Yarray.reshape(ni,)
            zrs = self.Zarray.reshape(ni,)
            mrs = self.mask.reshape(ni,)

            index = numpy.where(mrs == 0)
            
            xrsa = xrs[index]
            yrsa = yrs[index]
            zrsa = zrs[index]

            ## compute X1 and rmin
            x1sa, rminsa = self.ComputeX1AndDistance(xrsa, yrsa, zrsa, prepinter=False)

            x1s = numpy.zeros_like(xrs) + numpy.inf
            rmins = numpy.zeros_like(xrs) + numpy.inf

            x1s[index] = x1sa
            rmins[index] = rminsa

            self.x1 = x1s.reshape(shapexyz)
            self.rmin = rmins.reshape(shapexyz)

            
            if self.debug == True:
                print ("\npcrit = ", self.pcrit)
                print("x1[pcrit] = ", self.x1[self.pcrit])
                print ("rmin[pcrit] = ", self.rmin[self.pcrit])
                print ("shape(x1) = ", numpy.shape(self.x1))


        elif self.model == "Cabrit":
            ## mask based on distance
            maskd = ma.masked_outside(self.d, self.dmin, self.dmax).mask
            self.maskd = maskd

            ## mask based on theta angle
            maskt1 = ma.masked_inside(self.theta, self.thetamax*.5, numpy.pi-self.thetamax*.5).mask
            maskt2 = ma.masked_inside(self.theta, numpy.pi+self.thetamax*.5, 2*numpy.pi-self.thetamax*.5).mask
            maskt2 = ma.masked_inside(self.theta, numpy.pi+self.thetamax*.5, 2*numpy.pi-self.thetamax*.5).mask
            maskt  = numpy.logical_or(maskt1, maskt2)
            self.maskt = maskt

            ## Combine both masks
            self.mask = numpy.logical_or(maskd, maskt)

        else:
            self.mask = numpy.zeros_like(self.Zarray)

        self.maskindex = numpy.where(self.mask == 0)


        if self.debug:
            print("\nshape(mask) = ", numpy.shape(self.mask))
            print("mask[pcrit] = ", self.mask[self.pcrit])
            print("maskindex[0] = ", (self.maskindex[0][0], self.maskindex[1][0], self.maskindex[2][0]))
            print("mask[maskindex[0]] = ", self.mask[self.maskindex[0][0], self.maskindex[1][0], self.maskindex[2][0]])
            print("pcrit is in mask = ", 0 in (~(numpy.array(self.maskindex).T == self.pcrit)).sum(axis=1))

        return
    ##----------------------------------------------------------------------------------------------------------------------------------------------------

    
    ##----------------------------------------------------------------------------------------------------------------------------------------------------
    ##
    ## define function to compute velocity for molecules
    ##
    def ComputeVelocityMolecule(self):
        """

    input parameters:
    -----------------

        - None
        
    
    used parameters:
    -----------------
        
        - self.ThetaX:                  rotation angle of the x axis
        
        - self.ThetaY:                  rotation angle of the y axis
        
        - self.ThetaZ:                  rotation angle of the z axis

        - self.z (Raga):                distance to the source along the (jet-) axis

        - self.r (Raga):                distance to central (jet-) axis

        - self.vj (Raga):               initial value of velocity

        - self.z0 (Raga):               working surface
        
        - self.M1 (Raga):               Mixing-layer sound speed velocity ratio

        - self.d (Cabrit):              distance to the source

        - self.dmin (Cabrit):           minimal outflow radius

        - self.dmax (Cabrit):           maximal outflow radius

        - self.vmin (Cabrit):           inner velocity

        - self.alpha (Cabrit):          velocity exponent


    output parameters:
    ------------------

        - VelocityArray:                velocity for given cell
        """

        # Debug:
        if self.debug == True:
            print("\nModel = ", self.model)
            print("pcrit = ", self.pcrit)

            if self.model == "Raga":
                print ("z [pcrit] = ", abs(self.Zarray[self.pcrit]))
                print ("r [pcrit] = ", self.r[self.pcrit])  
                print ("vj = ", self.vj)
                print ("z0 = ", self.z0)
                print ("M1 = ", self.M1)

            elif self.model == "Cabrit":
                print ("d [pcrit] = ", self.d[self.pcrit])
                print ("dmin = ", self.dmin)
                print ("vmin = ", self.vmin)
                print ("alpha = ", self.alpha)


        ##================================================================================================================================================
        ## compute velocity
        ## define function for velocity here (NOTE, if you use an expression for a vector, use only the component along the line of sight!!!)
        ## (Make sure that result is in km / s)
        ##

        VelocityArrayTotal = numpy.zeros_like(self.Xarray)

        ## Raga model
        if self.model == "Raga":
            ## Calculate the velocity
            VelocityArrayTotal[self.maskindex] = self.vj * (1 - (self.z0-abs(self.Zarray[self.maskindex]))/(self.M1*(self.rc(self.x1[self.maskindex]) - self.r[self.maskindex]) + self.z0 - abs(self.Zarray[self.maskindex])))

            ## Correction to get the velocity component in z direction
            zcor = numpy.sign(self.ZarrayUnRot) * numpy.cos(self.ThetaJet)

            VelocityArrayZ = VelocityArrayTotal * zcor


        ## Cabrit model
        elif self.model == "Cabrit":

            ## Calculate the velocity
            VelocityArrayTotal[self.maskindex] = self.vmin * (self.dmin/self.d[self.maskindex])**self.alpha

            ## Correction to get the velocity component in z direction
            zcor = numpy.dot(numpy.array([self.XarrayUnRot, self.YarrayUnRot, self.ZarrayUnRot]).T, numpy.array([0,0,1])).T\
                    / numpy.clip(numpy.linalg.norm(numpy.array([self.XarrayUnRot, self.YarrayUnRot, self.ZarrayUnRot]).T, axis=3).T, 1e-5, None)

            VelocityArrayZ = VelocityArrayTotal * zcor

        ## Every other pixel
        else:

            VelocityArrayTotal = 0

            VelocityArrayZ = 0


        ##
        ##================================================================================================================================================

        # Debug:
        if self.debug == True:
            print ("\nVelocityArrayTotal[pcrit] = ", VelocityArrayTotal[self.pcrit])
            print ("VelocityArrayZ[pcrit] = ", VelocityArrayZ[self.pcrit])


        ## we're done
        return VelocityArrayZ
    ##----------------------------------------------------------------------------------------------------------------------------------------------------


    ##----------------------------------------------------------------------------------------------------------------------------------------------------
    ##
    ## calculate the molecular density for a given vector r = (z, r)
    ## This function is necessary to use the integration function qTrapz3D
    ##
    def ComputeDensityMolecule(self, x, y, z):  

        """

    input parameters:
    -----------------

        - x:                            distance to the source along the (jet-) axis

        - y:                            distance to central (jet-) axis
 
        - z:                            origin of the molecules in absolute distance


    used parameters:
    -----------------

        - self.z0 (Raga):               working surface

        - self.beta (Raga):             shape of the outflow

        - self.epsilon (Raga):          efficiancy

        - self.n0 (Raga):             initial value of density
        
        - self.M1 (Raga):               Mixing-layer sound speed velocity ratio

        - self.A0 (Raga):               amplitude of synthetic gaussian-like data

        - self.sigma (Raga):            standart deviation of synthetic gaussian-like data

        - x1 (Raga):                    origin of the molecules in absolute distance

        - rmin (Raga):                  distance to the outflow surface

        - self.d (Cabrit):              distance to the source

        - self.dmin (Cabrit):           minimal outflow distance

        - self.dmax (Cabrit):           maximal outflow distance

        - self.theta (Cabrit):          angle from outflow axis

        - self.thetamax (Cabrit):       maximal opening angle

        - self.nmin (Cabrit):         minimal density

        - self.delta (Cabrit):          density exponent


    output parameters:
    ------------------

        - LocalDensity:                 density for given cell
        """

        # debug
        if self.debug == True:

            print("\nModel = ", self.model)
            print("shape(x) = ", numpy.shape(x))
            print("shape(y) = ", numpy.shape(y))
            print("shape(z) = ", numpy.shape(z))

        ## Check if x/y/z are equal to the complete arrays
        if numpy.array_equal(x, self.Xarray) and numpy.array_equal(y, self.Yarray) and numpy.array_equal(z, self.Zarray):
            
            ## Set temporal mask array and other parameters to the previously derived values
            mask = self.mask
            maskindex = self.maskindex

            if self.debug == True:
                print("pcrit = ", self.pcrit)
                print("x[pcrit] = ", x[self.pcrit])
                print("y[pcrit] = ", y[self.pcrit])
                print("z[pcrit] = ", z[self.pcrit])
                print("mask[pcrit] = ", mask[self.pcrit])


            if self.model == "Raga":
                r = self.r
                x1 = self.x1
                rmin = self.rmin

                if self.debug == True:
                    print("r[pcrit] = ", r[self.pcrit])
                    print("X1[pcrit] = ", x1[self.pcrit])
                    print("rmin[pcrit] = ", rmin[self.pcrit])

            elif self.model == "Cabrit":
                d = self.d
                theta = self.theta

                if self.debug == True:
                    print("d[pcrit] = ", d[self.pcrit])
                    print("theta[pcrit] = ", theta[self.pcrit])

        ## Else calculate the mask and so on for this case
        else:   
                    
            if self.debug == True and numpy.shape(x) != ():
                pcenter = numpy.array(numpy.floor(numpy.array(numpy.shape(x))/2),dtype=int)
                pcenter[0] = 0 if len(pcenter) == 4 else pcenter[0]
                pcenter = tuple(pcenter)
                print("pcenter = ", pcenter)
                print("pcenter[:-1] = ", pcenter[:-1])
                print("x[pcenter] = ", x[pcenter])
                print("y[pcenter] = ", y[pcenter])
                print("z[pcenter] = ", z[pcenter])

            elif self.debug == True and numpy.shape(x) == ():
                print("x = ", x)
                print("y = ", y)
                print("z = ", z)
    
            ## Calculate the parameters (mask, rmin, X1...) for the Raga model
            if self.model == "Raga":
                r = numpy.sqrt(x**2+y**2)
                mask, maskindex_temp, x1, rmin = self.ComputeMask(Xarray=x, Yarray=y, Zarray=z, r=r, full=False)

                shapexyz = numpy.shape(x)
                ni = numpy.array(shapexyz).prod()

                xrs = x.reshape(ni,)
                yrs = y.reshape(ni,)
                zrs = z.reshape(ni,)
                mrs = mask.reshape(ni,)

                index = numpy.where(mrs == 0) 
            
                xrsa = xrs[index]
                yrsa = yrs[index]
                zrsa = zrs[index]

                x1sa, rminsa = self.ComputeX1AndDistance(xrsa, yrsa, zrsa, prepinter=False)

                x1s = numpy.zeros_like(xrs) + numpy.inf
                rmins = numpy.zeros_like(xrs) + numpy.inf

                x1s[index] = x1sa
                rmins[index] = rminsa

                x1 = x1s.reshape(shapexyz)
                rmin = rmins.reshape(shapexyz)

                maskl = numpy.less_equal(r*0, r)
                maskg = numpy.greater_equal(r, self.rc(abs(z)))

                mask = numpy.logical_and(maskl, maskg)
                
                maskindex = numpy.where(mask == 0)
                
                if self.debug == True and numpy.shape(x) == ():
                    print("mask = ", mask)
                    print("r = ", r)
                    print("x1 = ", x1)
                    print("rmin = ", rmin)

                if self.debug == True and numpy.shape(x) != ():
                    print(mask)
                    print("mask[pcenter] = ", mask[pcenter[:-1]])
                    print("r[pcenter] = ", r[pcenter[:-1]])
                    print("x1[pcenter] = ", x1[pcenter[:-1]])
                    print("rmin[pcenter] = ", rmin[pcenter[:-1]])
                    
                if self.debug == True:
                    print ("z0 = ", self.z0)
                    print ("beta = ", self.beta)
                    print ("epsilon = ", self.epsilon)
                    print ("n0 = ", self.n0)
                    print ("ne0 = ", self.ne0)
                    print ("M1 = ", self.M1)
                    print ("rc(x1[pcenter]) = ", self.rc(x1[pcenter[:-1]]))
                    print ("A0 = ", self.A0)
                    print ("sigma = ", self.sigma)

            ## Calculate the parameters (mask, d, theta, ...) for the Cabrit model
            elif self.model == "Cabrit":
                d = numpy.sqrt(x**2+y**2+z**2)
                theta = numpy.arccos(z/d)
                mask, maskindex_temp, x1, rmin = self.ComputeMask(d=d, theta=theta, full=False)

                
                if numpy.shape(mask) == ():
                    mask = numpy.zeros_like(d, dtype=bool) + mask
                
                maskindex = numpy.where(mask == 0)

                if self.debug == True and numpy.shape(x) != ():
                    print(mask)
                    print("mask[pcenter] = ", mask[pcenter[:-1]])
                    print("d[pcenter] = ", d[pcenter[:-1]])
                    print("theta[pcenter] = ", theta[pcenter[:-1]])
                    print("nmin = ", self.nmin)
                    print("delta = ", self.delta)

                if self.debug == True and numpy.shape(x) == ():
                    print("mask[pcenter] = ", mask)
                    print("d[pcenter] = ", d)
                    print("theta[pcenter] = ", theta)
                    print("nmin = ", self.nmin)
                    print("delta = ", self.delta)

                if self.debug == True:
                    print ("dmin = ", self.dmin)
                    print ("dmax = ", self.dmax)
                    print("nmin = ", self.nmin)
                    print("delta = ", self.delta)



        ##================================================================================================================================================
        ## compute denstiy
        ## define function for density here
        ## (Make sure that result is in m-3)
        ##
        ## calculate local density

        ## Define local density and calculate it for the corresponding model
        LocalDensity = numpy.zeros_like(x) + self.nbackm3

        if self.model == "Raga":

            if numpy.shape(x) != ():
                LocalDensity[maskindex] = self.epsilon*self.ne(x1)[maskindex]*self.rc(x1)[maskindex]**2 \
                    / (r[maskindex]*(1/self.M1 - self.beta*self.rc(x1)[maskindex]/(2*x1[maskindex]))) \
                    * (self.z0-abs(z)[maskindex]) / (self.M1*(self.rc(x1)[maskindex]-r[maskindex])+self.z0-abs(z)[maskindex])**2

            if numpy.shape(x) == () and mask == 0:
                LocalDensity = self.epsilon*self.ne(x1)*self.rc(x1)**2 \
                    / (r*(1/self.M1 - self.beta*self.rc(x1)/(2*x1))) \
                    * (self.z0-abs(z)) / (self.M1*(self.rc(x1)-r[maskindex])+self.z0-abs(z))**2


        elif self.model == "Cabrit":

            if numpy.shape(x) != ():
                LocalDensity[maskindex] = self.nmin*(self.dmin/d[maskindex])**self.delta

            elif numpy.shape(x) == () and mask == 0:
                LocalDensity = self.nmin*(self.dmin/d)**self.delta

        ## For the Raga nodel variate the outflow head
        if self.model == "Raga":
            ## add additional density density in a gaussian shape
            if numpy.shape(x) != ():
                GausEdge = numpy.zeros_like(LocalDensity)
                GausEdge[maskindex] = self.Gaus(z[maskindex], rmin[maskindex])

            if numpy.shape(x) == () and mask == 0:
                GausEdge = self.Gaus(z, rmin)

            else:
                GausEdge = 0

   
            LocalDensity += GausEdge

        ##
        ##================================================================================================================================================
        
        # Debug:
        if self.debug == True:
            if numpy.array_equal(x, self.Xarray) and numpy.array_equal(y, self.Yarray) and numpy.array_equal(z, self.Zarray):
                print ("\nLocalDensity[pcrit] = ", LocalDensity[self.pcrit])
            
                if self.model == "Raga":
                    print ("GausEdge[pcrit] = ", GausEdge[self.pcrit])

            elif numpy.shape(x) != ():
                print ("\nLocalDensity[pcenter] = ", LocalDensity[pcenter])
            
                if self.model == "Raga":
                    print ("GausEdge[pcenter] = ", GausEdge[pcenter])

            elif numpy.shape(x) == ():
                print ("\nLocalDensity = ", LocalDensity)
            
                if self.model == "Raga":
                    print ("GausEdge = ", GausEdge)


        ## we're done
        return LocalDensity
    ##----------------------------------------------------------------------------------------------------------------------------------------------------

    
    ##----------------------------------------------------------------------------------------------------------------------------------------------------
    ##
    ## define function to compute temperature for molecules
    ##
    def ComputeTemperatureMolecule(self):
        """

    input parameters:
    -----------------

        - None


    used parameters:
    -----------------

        - self.z (Raga):                distance to the source along the (jet-) axis

        - self.r (Raga):                distance to central (jet-) axis

        - self.z0 (Raga):               working surface

        - self.beta (Raga):             shape of the outflow

        - self.epsilon (Raga):          outflow efficiancy

        - self.n0 (Raga):             initial value of density
        
        - self.M1 (Raga):               Mixing-layer sound speed velocity ratio

        - self.x1 (Raga):               origin of the molecules in absolute distance

        - self.rmin (Raga):             distance to the outflow surface

        - self.A0 (Raga):               amplitude of synthetic gaussian-like data

        - self.sigma (Raga):            standart deviation of synthetic gaussian-like data

        - self.dmin (Cabrit):           minimal outflow distance

        - self.dmax (Cabrit):           maximal outflow distance

        - self.theta (Cabrit):          angle from outflow axis

        - self.thetamax (Cabrit):      maximal opening angle

        - self.nmin (Cabrit):         minimal density


    output parameters:
    ------------------

        - TemperatureArray:             temperature for given cell
        """

        # Debug:
        if self.debug == True:
            print("\nModel = ", self.model)
            print("pcrit = ", self.pcrit)
            print ("z [pcrit] = ", abs(self.Zarray[self.pcrit]))
            print ("r [pcrit] = ", self.r[self.pcrit])
            print ("d [pcrit] = ", self.d[self.pcrit])
            print ("dmin = ", self.dmin)
            print ("dmax = ", self.dmax)
            print("Theta max = ", self.thetamax)
            print("Theta X = ", self.ThetaX)
            print("Theta Y = ", self.ThetaY)
            print("Theta Z = ", self.ThetaZ)

            if self.model == "Raga":
                print ("T1 = ", self.T1)
                print ("epsilon = ", self.epsilon)
                print ("gamma = ", self.gamma)
                print ("x1 = ", self.x1)
                print ("rmin = ", self.rmin)

            elif self.model == "Cabrit":

                print ("Tmin = ", self.Tmin)
    


        ##================================================================================================================================================
        ## compute temperature
        ## define function for temperature here (make sure that result is Kelvin)
        ##

        TemperatureArray = numpy.zeros_like(self.Xarray) + self.Tback

        if self.model == "Raga":
            TemperatureArray[self.maskindex] = (self.NtotArray[self.maskindex] / (self.epsilon * self.ne(self.x1[self.maskindex]) * 100))**(self.gamma-1) * self.T1

        elif self.model == "Cabrit":
            TemperatureArray[self.maskindex] = (self.Ntotm3[self.maskindex] / (self.nmin))**(self.gamma-1) * self.Tmin

    
        ##================================================================================================================================================

        # Debug:
        if self.debug == True:
            print("\npcrit = ", self.pcrit)
            print ("TemperatureArray [pcrit] = ", TemperatureArray[self.pcrit])


        ## we're done
        return TemperatureArray
    ##----------------------------------------------------------------------------------------------------------------------------------------------------


    ##----------------------------------------------------------------------------------------------------------------------------------------------------
    ##
    ## define function to compute source size for molecules
    ##
    def ComputeSourceSizeMolecule(self):
        """

    input parameters:
    -----------------

        - sSize:                        initial value of source size


    output parameters:
    ------------------

        - SourceSizeArray:              source size for given cell
        """

        # Debug:
        if self.debug == True:
            print ("\nsSize = ", self.ssComp)


        ##================================================================================================================================================
        ## compute source size
        ## define function for source size here (make sure that result is in arcsec)
        ##
        SourceSizeArray = numpy.ones_like(self.Xarray) * self.ssComp
        ##
        ##================================================================================================================================================

        # Debug:
        if self.debug == True:
            print ("\nnpcrit = ", self.pcrit)
            print ("SourceSizeArray [pcrit] = ", SourceSizeArray[self.pcrit])


        ## we're done
        return SourceSizeArray
    ##----------------------------------------------------------------------------------------------------------------------------------------------------


    ##----------------------------------------------------------------------------------------------------------------------------------------------------
    ##
    ## define function to compute line width for molecules
    ##
    def ComputeLineWidthMolecule(self):
        """

    input parameters:
    -----------------

        - vWidth:                       initial value of line width


    output parameters:
    ------------------

        - LocalLineWidth:               line width for given cell
        """

        # Debug:
        if self.debug == True:
            print ("\nvWidth = ", self.vWidthComp)


        ##================================================================================================================================================
        ## compute line width
        ## define function for line width here
        ## (Make sure that result is in km / s)
        ##
        LocalLineWidth = numpy.ones_like(self.Xarray) * self.vWidthComp
        ##
        ##================================================================================================================================================

        # Debug:
        if self.debug == True:
            print ("\nnpcrit = ", self.pcrit)
            print ("LocalLineWidth [pcrit] = ", LocalLineWidth[self.pcrit])


        ## we're done
        return LocalLineWidth
    ##----------------------------------------------------------------------------------------------------------------------------------------------------


    ##----------------------------------------------------------------------------------------------------------------------------------------------------
    ##
    ## integrates function Func(x,y,z) in the cuboid [ax,bx] * [ay,by] * [az,bz] using the trapezoidal rule with (nx * ny * nz) integration points
    ## boosted version of from https://books.google.de/books?id=oCVZBAAAQBAJ&pg=PA390&lpg=PA390&dq=trapezoidal+rule+in+3d+python&source=bl&ots
    ##                          =qDxRaL-fmt&sig=KbSEJ_tTzFgrvv_1UpYSZQV9h3E&hl=en&sa=X&ved=0ahUKEwj8ktLp9MbUAhVQalAKHa7_AUAQ6AEIYDAJ#v=onepage&q
    ##                          =trapezoidal%20rule%20in%203d%20python&f=false
    ##
    def qTrapz3D(self, index, Func):
        """

    input parameters:
    -----------------

        - index:                        [xi, yi, zi] list of integrated voxel indices

        - Func:                         Function to integrate


    used parameters:
    ----------------

        - nx:                           number of grid points along x-axis

        - ny:                           number of grid points along y-axis
        
        - nz:                           number of grid points along z-axis
        
        - xStepSize                     Step size in x direction
        
        - yStepSize                     Step size in y direction
        
        - zStepSize                     Step size in z direction
        

    output parameters:
    ------------------

        - s:                            computed integral
        """

        # Debug:
        if self.debug == True:
            print ("\nFunc = ", Func)
            print("Number of voxel (via index) = ", len(index[0]))
            print ("nx = ", self.nx)
            print ("ny = ", self.ny)
            print ("nz = ", self.nz)
            print("Index voxe [0] = ", (index[0][0], index[1][0], index[2][0]))

        ## Get the analyzed voxel coordinates
        xv = self.Xarray[index]
        yv = self.Yarray[index]
        zv = self.Zarray[index]

        ## Create unit cell around voxel
        xca = [numpy.linspace(-0.5, 0.5, self.nx) if self.nx > 1 else numpy.array([0])]
        yca = [numpy.linspace(-0.5, 0.5, self.ny) if self.ny > 1 else numpy.array([0])]
        zca = [numpy.linspace(-0.5, 0.5, self.nz) if self.nz > 1 else numpy.array([0])]

        #xc, yc, zc = numpy.meshgrid(yca, xca, zca)
        xc, yc, zc = numpy.meshgrid(xca, yca, zca)


        ## Create empty grid, where one has n (matching number of voxels) times the unit cells
        x = numpy.zeros([len(index[0]), self.nx, self.ny, self.nz], dtype = numpy.float32)
        y = numpy.copy(x)
        z = numpy.copy(x)


        ## Filling the empty cells with the unit cells
        x[:] = xc
        y[:] = yc
        z[:] = zc

        ## Shift each cell to voxel position, also considering the actual step size
        x = (x.T*self.xStepSize[index] + xv).T
        y = (y.T*self.yStepSize[index] + yv).T
        z = (z.T*self.zStepSize + zv).T

        ## Apply the function
        funca = Func(x, y, z)
        
        ## Create weights for the voxel, the edges are down weighted 
        if self.nx > 1:
            hx = (self.xStepSize[index]) / (self.nx - 1)
            wx = (numpy.ones([len(index[0]), self.nx, 1, 1]).T * hx).T
            wx[:,[0,-1],0,0] *= 0.5

        else:
            wx = (numpy.ones([len(index[0]), self.nx, 1, 1]).T * self.xStepSize[index]).T


        if self.ny > 1:
            hy = (self.yStepSize[index]) / (self.ny - 1)
            wy = (numpy.ones([len(index[0]), 1, self.ny, 1]).T * hy).T
            wy[:,0,[0,-1],0] *= 0.5

        else:
            wy = (numpy.ones([len(index[0]), 1, self.ny, 1]).T * self.yStepSize[index]).T


        if self.nz > 1:
            hz = (self.zStepSize) / (self.nz - 1)
            wz = (numpy.ones([len(index[0]), 1, 1, self.nz]).T * hz).T
            wz[:,0,0,[0,-1]] *= 0.5

        else:
            wz = (numpy.ones([len(index[0]), 1, 1, self.nz]).T * self.zStepSize).T

        ## Create the full weigth
        w = wx*wy*wz

        ## Applying the weight to the function and sum each cell
        s = (w*funca).sum(axis=(1,2,3))

        # Debug:
        if self.debug == True:
            print("\nVoxel [0] = ", [self.Xarray[(index[0][0], index[1][0], index[2][0])], 
                                     self.Yarray[(index[0][0], index[1][0], index[2][0])], 
                                     self.Zarray[(index[0][0], index[1][0], index[2][0])]])
            print("StepSize [x, y, z] = ", [self.xStepSize[index][0], self.yStepSize[index][0], self.zStepSize])
            print("w [0] = ", w[0][0])
            print("funca [0] = ", funca[0][0])

            print("s [0] = ", s[0])
            print("Func [Voxel [0]] = ", Func(self.Xarray[(index[0][0], index[1][0], index[2][0])], 
                                              self.Yarray[(index[0][0], index[1][0], index[2][0])], 
                                              self.Zarray[(index[0][0], index[1][0], index[2][0])]))
        return s

    ##----------------------------------------------------------------------------------------------------------------------------------------------------


    ##----------------------------------------------------------------------------------------------------------------------------------------------------
    ##
    ## calculate molfit parameters for current distance
    ##
    def CalcMolfitParameters(self):
        """

    input parameters:
    -----------------

        - None


    output parameters:
    ------------------

        - LocalMolfitContent:           current line of molfit file for current cell
        """


        ##------------------------------------------------------------------------------------------------------------------------------------------------
        ## Estimate cube size and compute the outflow shape (Raga model)
        self.ComputeOutflowShape()
    
        ##------------------------------------------------------------------------------------------------------------------------------------------------
        ## Create coordinate grid  
        self.CreateCoordinateGrid()


        # Debug:
        if self.debug == True:
            print("\npcrit = ", self.pcrit)
            print("Xarray[pcrit] = ", self.Xarray[self.pcrit])
            print("Yarray[pcrit] = ", self.Yarray[self.pcrit])
            print("Zarray[pcrit] = ", self.Zarray[self.pcrit])
            print("d[pcrit] = ", self.d[self.pcrit])
            print("z[pcrit] = ", abs(self.Zarray[self.pcrit]))
            print("r[pcrit] = ", self.r[self.pcrit])
            print("phi[pcrit] = ", self.phi[self.pcrit])
            print("theta[pcrit] = ", self.theta[self.pcrit])
       
        ##------------------------------------------------------------------------------------------------------------------------------------------------
        ## Create outflow mask 
        self.mask, self.maskindex, self.x1, self.rmin = self.ComputeMask(r=self.r, Xarray=self.Xarray, Yarray=self.Yarray, Zarray=self.Zarray, d=self.d, theta=self.theta, full=True)

        ##------------------------------------------------------------------------------------------------------------------------------------------------
        ## compute velocity offset for current cell
        self.VelocityArray = self.ComputeVelocityMolecule()

        # Debug:
        if self.debug == True:
            print ("\npcrit = ", self.pcrit)
            print("VelocityArray [pcrit] = ", self.VelocityArray[self.pcrit])
            print("Max velocity = ", self.VelocityArray.max())
            print("Min velocity = ", self.VelocityArray.min())


        ##------------------------------------------------------------------------------------------------------------------------------------------------
        ## calculate column density
        self.nx = 10
        self.ny = 10
        self.nz = 10

        ## Compute the density per cell and recalculate it to m3
        PPCArray = self.qTrapz3D(self.maskindex, self.ComputeDensityMolecule)
        self.Ntotm3 = numpy.zeros_like(self.Xarray)
        self.Ntotm3[self.maskindex] = PPCArray / (self.xStepSize[self.maskindex]*self.yStepSize[self.maskindex]*self.zStepSize)

        ## Recalculate it to cm2
        PPCM2Array = self.Ntotm3 / self.m3TOcm2 * self.zStepSize
        
        self.nbackcm2 = self.nbackcm3 * 100 * self.zStepSize                 ## Particle background density in cm-2

        ## Complete density array
        self.NtotArray = PPCM2Array + self.nbackcm2

        ## Debug:
        if self.debug == True:
            print ("\npcrit = ", self.pcrit)
            print ("NtotArray [pcrit] = ", self.NtotArray[self.pcrit])

        ##------------------------------------------------------------------------------------------------------------------------------------------------
        ## compute temperature for current cell
        self.TemperatureArray = self.ComputeTemperatureMolecule()

        # Debug:
        if self.debug == True:
            print ("\npcrit = ", self.pcrit)
            print("TemperatureArray [pcrit] = ", self.TemperatureArray[self.pcrit])


        ##------------------------------------------------------------------------------------------------------------------------------------------------
        ## compute source size for current cell
        self.SourceSizeArray = self.ComputeSourceSizeMolecule()

        # Debug:
        if self.debug == True:
            print ("\npcrit = ", self.pcrit)
            print("SourceSizeArray [pcrit] = ", self.SourceSizeArray[self.pcrit])


        ##------------------------------------------------------------------------------------------------------------------------------------------------

        ## compute line width for current cell
        self.vWidthArray = self.ComputeLineWidthMolecule()

        # Debug:
        if self.debug == True:
            print ("\npcrit = ", self.pcrit)
            print("vWidthArray [pcrit] = ", self.vWidthArray[self.pcrit])


        # Debug:
        if self.debug == True:
            print("\nmodel = ", self.model)
            print("pcrit = ", self.pcrit)
            print("Xarray[pcrit] = ", self.Xarray[self.pcrit])
            print("Yarray[pcrit] = ", self.Yarray[self.pcrit])
            print("Zarray[pcrit] = ", self.Zarray[self.pcrit])
            print("d[pcrit] = ", self.d[self.pcrit])
            print("z[pcrit] = ", abs(self.Zarray[self.pcrit]))
            print("r[pcrit] = ", self.r[self.pcrit])
            print("phi[pcrit] = ", self.phi[self.pcrit])
            print("theta[pcrit] = ", self.theta[self.pcrit])
            print("VelocityArray [pcrit] = ", self.VelocityArray[self.pcrit])
            print ("NtotArray [pcrit] = ", self.NtotArray[self.pcrit])
            print("TemperatureArray [pcrit] = ", self.TemperatureArray[self.pcrit])
            print("SourceSizeArray [pcrit] = ", self.SourceSizeArray[self.pcrit])
            print("vWidthArray [pcrit] = ", self.vWidthArray[self.pcrit])
            print("DistanceObserverArray[pcrit] = ", self.DistanceObserverArray[self.pcrit[2]])


        ## we're done
        return
    ##----------------------------------------------------------------------------------------------------------------------------------------------------


    ##----------------------------------------------------------------------------------------------------------------------------------------------------
    ##
    ## function to create MolFit files from arrays
    ##
    def WriteMolfit(self):

        ## initialize return parameter and define molfit parameters for each component (only foreground components -> no source size definition
        Molecule = self.CalcParameters['Molecule']
        MolfitContent = "{:s}   {:d}\n".format(Molecule, self.NumberOfCellsAlongLineOfSight)

        # Debug:
        if self.debug == True:
            print("\nMolfitPath = ", self.CalcParameters['MolfitPath'])
            print("Molecule = ", Molecule)
            print("1st Molfitline = ", MolfitContent)
            print("NumberXCoord = ", self.NumberXCoord)
            print("NumberYCoord = ", self.NumberYCoord)
            print("NumberZCoord = ", self.NumberOfCellsAlongLineOfSight)

            print("\nmodel = ", self.model)
            print("pcrit = ", self.pcrit)
            print("Xarray[pcrit] = ", self.Xarray[self.pcrit])
            print("Yarray[pcrit] = ", self.Yarray[self.pcrit])
            print("Zarray[pcrit] = ", self.Zarray[self.pcrit])
            print("d[pcrit] = ", self.d[self.pcrit])
            print("z[pcrit] = ", abs(self.Zarray[self.pcrit]))
            print("r[pcrit] = ", self.r[self.pcrit])
            print("phi[pcrit] = ", self.phi[self.pcrit])
            print("theta[pcrit] = ", self.theta[self.pcrit])
            print("VelocityArray [pcrit] = ", self.VelocityArray[self.pcrit])
            print ("NtotArray [pcrit] = ", self.NtotArray[self.pcrit])
            print("TemperatureArray [pcrit] = ", self.TemperatureArray[self.pcrit])
            print("SourceSizeArray [pcrit] = ", self.SourceSizeArray[self.pcrit])
            print("vWidthArray [pcrit] = ", self.vWidthArray[self.pcrit])
            print("DistanceObserverArray[pcrit] = ", self.DistanceObserverArray[self.pcrit[2]])

        index = ((x, y) for x in range(self.NumberXCoord) for y in range(self.NumberYCoord))
        
        list(map(self.WriteMolfitFile, index))

        return
    ##----------------------------------------------------------------------------------------------------------------------------------------------------
    


    ##----------------------------------------------------------------------------------------------------------------------------------------------------
    ##
    ## function to create MolFit files from arrays
    ##
    def WriteMolfitFile(self, index):

        ## get the pixel
        i, ii = index
        
        ## initialize local molfit file componetet
        Molecule = self.CalcParameters['Molecule']
        MolfitContent = "{:s}   {:d}\n".format(Molecule, self.NumberOfCellsAlongLineOfSight)

        for iii in range(self.NumberOfCellsAlongLineOfSight):

            MolfitContent += "{:10.3f} {:10.3f} {:15.3e} {:10.3f} {:10.3f} {:5.0f}\n".format(self.SourceSizeArray[i,ii,iii], self.TemperatureArray[i,ii,iii],
                                                                                self.NtotArray[i,ii,iii], self.vWidthArray[i,ii,iii],
                                                                                self.VelocityArray[i,ii,iii], iii+1)
        ##----------------------------------------------------------------------------------------------------------------------------------------------------
        ## write molfit file to disk
        MolfitsFileName = self.CalcParameters['MolfitPath'] + "%i-%i_model.molfit" %(i+1, ii+1)
        MolfitsFile = open(MolfitsFileName, 'w')
        MolfitsFile.write(MolfitContent)
        MolfitsFile.close()

        return
    ##----------------------------------------------------------------------------------------------------------------------------------------------------
    


    ##----------------------------------------------------------------------------------------------------------------------------------------------------
    ##
    ## function to create an empty fits file (only the header and extension is relevant) and modify the observation-xml file
    ##
    def CreateEmptyFitsAndModifyXML(self):

        ## read in the comparison header
        header = fits.getheader(self.CalcParameters['ReadPath']+self.CalcParameters['cube_file'], ext=0)

        ## Open xml file and getting its rood
        modtree = ET.parse(self.CalcParameters['ReadPath']+self.CalcParameters['xml_file'])
        modrood = modtree.getroot()

        ## get the restfrequency [in MHz]
        restfreq = self.CalcParameters["RestFreq"]

        ## Calculate velocity range and enlarge it a bit [in km/s]
        vmin = self.VelocityArray.min()
        vmax = self.VelocityArray.max()

        ## Enlarge v space
        vmin = vmin*1.2 if vmin <= 0 else vmin*0.8
        vmax = vmax*1.2 if vmax >= 0 else vmax*0.8


        ## Read frequency range from Master xml file [in MHz]
        FreqRange1 = self.CalcParameters["MinFExtend"]

        ## Calculate the frequency range from the xml file [in MHz]
        MinExpRangeXML = float(modrood.find("file").find("FrequencyRange").find("MinExpRange").text)
        MaxExpRangeXML = float(modrood.find("file").find("FrequencyRange").find("MaxExpRange").text)

        if MinExpRangeXML == 0 or MaxExpRangeXML == 0 or MinExpRangeXML == MaxExpRangeXML:
            FreqRange2 = 0

        else:
            FreqRange2 = 2 * max(MaxExpRangeXML-restfreq, restfreq-MinExpRangeXML)

        ## Calculate frequency range from velocity [in MHz]
        MinExpRangeCalc = SpectralCoord(vmax * u.km/u.s, doppler_convention='radio', doppler_rest=restfreq * u.MHz).to(u.MHz).value
        MaxExpRangeCalc = SpectralCoord(vmin * u.km/u.s, doppler_convention='radio', doppler_rest=restfreq * u.MHz).to(u.MHz).value
        FreqRange3 = 2 * max(MaxExpRangeCalc-restfreq, restfreq-MinExpRangeCalc)

        
        ## Find max frequency range and outer values [in MHz] 
        FreqRange = max(FreqRange1, FreqRange2, FreqRange3)

        ## Select the outer frequency ranges
        self.MinExpRange = restfreq - FreqRange/2
        self.MaxExpRange = restfreq + FreqRange/2

        ## Calculate the frequency step size
        CalcFreqRange = (self.MaxExpRange - self.MinExpRange) / self.CalcParameters["MinFSteps"]
        self.StepFrequency = min(CalcFreqRange, self.CalcParameters["FrequencyStep"])


        ## Calculate the velocty range [in km/s] (to match the eventually updated frequency range)
        vmin = SpectralCoord(self.MaxExpRange * u.MHz,
                            doppler_convention='radio',
                            doppler_rest=restfreq * u.MHz).to(u.km/u.s).value

        vmax = SpectralCoord(self.MinExpRange * u.MHz,
                            doppler_convention='radio',
                            doppler_rest=restfreq * u.MHz).to(u.km/u.s).value


        ### Create empty fits with a header
        ## Modify the header
        header["BITPIX"]  = 64
        ## Set the number of axes to 3
        header["NAXIS"]   =   3

        print(self.resolution)

        ## Adjust the first axis value
        header["NAXIS1"]  =   self.NumberXCoord
        header["CTYPE1"]  =   "RA---SIN"
        header["CRVAL1"]  =   0.
        header["CDELT1"]  =   self.resolution
        header["CRPIX1"]  =   (self.NumberXCoord-1)/2
        header["CUNIT1"]  =   "deg"

        ## Adjust the second axis value
        header["NAXIS2"]  =   self.NumberYCoord
        header["CTYPE2"]  =   "DEC--SIN"
        header["CRVAL2"]  =   0.
        header["CDELT2"]  =   self.resolution
        header["CRPIX2"]  =   (self.NumberYCoord-1)/2
        header["CUNIT2"]  =   "deg"
        
        ## Adjust the third axis value
        header["NAXIS3"]  =   self.NumberOfCellsAlongLineOfSight
        header["CTYPE3"]  =   "VRAD"
        header["CRVAL3"]  =   0.
        header["CDELT3"]  =   (vmax-vmin) / self.NumberOfCellsAlongLineOfSight
        header["CRPIX3"]  =   (self.NumberOfCellsAlongLineOfSight-1)/2
        header["CUNIT3"]  =   "km s-1"

        ## Set the restfrequency in Hz
        header["RESTFRQ"] =   restfreq * 1e6 
        ## Set the intensity unit
        header["BUNIT"]   =   self.CalcParameters["BUNIT"]
        
        ## Set the beam
        header["BMAJ"]    =   self.CalcParameters["BMAJ"]
        header["BMIN"]    =   self.CalcParameters["BMIN"]
        header["BPA"]     =   self.CalcParameters["BPA"]

        ### Change some other parameter
        header["OBJECT"]  =  "Synthetic Outflow"
        header["TELESCOP"]=  (platform.node(), "Used platform to simulate the cubes")
        header["OBSERVER"]=  "Kahl"
        header["DATE-OBS"]=  (str(datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")), "Time of the calculation")
        header["TIMESYS"] =  (str(datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")), "Time of the calculation")

        ### Save general model parameters in header
        if self.model == "Cabrit":
            header["MODEL"]  =  (self.model, "Applied outflow model")
            header["JETTHE"] =  (self.ThetaJet, "Jet outflow angle theta in deg")
            header["JETPHI"] =  (self.PhiJet, "Jet outflow angle phi in deg")
            header["OFVEL"]  =  (self.vmin, "Outflow velocity in km/s")
            header["OFDEN"]  =  (self.nmin, "Outflow density in cm-3")
            header["OFTEM"]  =  (self.Tmin, "Outflow temperature in K")

        elif self.model == "Raga":
            header["MODEL"]  =  (self.model, "Applied outflow model")
            header["JETTHE"] =  (self.ThetaJet, "Jet outflow angle theta in deg")
            header["JETPHI"] =  (self.PhiJet, "Jet outflow angle phi in deg")
            header["OFVEL"]  =  (self.vj, "Outflow velocity in km/s")
            header["OFDEN"]  =  (self.n0, "Outflow density in cm-3")
            header["OFTEM"]  =  (self.T1, "Outflow temperature in K")

        ## Add model relevant parameter
        if self.model == "Raga":
            header["EPSIL"] = self.epsilon
            header["GAMMA"] = self.gamma
            header["M1"]    = self.M1
            header["BETA"]  = self.beta
            header["R0"]    = (self.r0, "in au")
            header["Z0"]    = (self.z0, "in au")

        elif self.model == "Cabrit":
            header["ALPHA"] = self.alpha
            header["DMIN"]  = (self.dmin, "in au")
            header["SIGMA"] = self.sigmaof
            header["OFOPEN"]= (self.thetamax, "in deg")


        ## Delete some feature
        features = ["NAXIS4", "CTYPE4", "CRVAL4", "CDELT4", "CRPIX4", "CUNIT4", "HISTORY", "COMMENT", 
                    "PC01_04", "PC02_04", "PC03_04", "PC04_01", "PC04_02", "PC04_03", "PC04_04", "ORIGIN"]

        for feature in features:
            if feature in header:
                del header[feature]

        ## Handling PC0X_0Y
        PCX_X = ["PC01_01", "PC02_02", "PC03_03"]
        PCX_Y = ["PC01_02", "PC01_03", "PC02_01", "PC02_03", "PC03_01", "PC03_02"]

        for XX in PCX_X:
            header[XX] = 1.

        for XY in PCX_Y:
            header[XY] = 0.

        ## Create wcs
        wcs = WCS(header) 

        ## Create zero array (has to have a unit, but it'll get changed again later on)
        dummy_data = numpy.zeros([header["NAXIS3"],header["NAXIS2"],header["NAXIS1"]], dtype=int) * u.K

        ## Create empty cube
        dummy_cube = SpectralCube(data=dummy_data, wcs=wcs, header=header)

        ## Save the cube
        dummy_cube.write(self.CalcParameters['CubePath']+self.CalcParameters["cube_file"], overwrite=True)

        ## Correct the intensity unit
        fits.setval(self.CalcParameters['CubePath']+self.CalcParameters["cube_file"],
                                   "BUNIT", value=self.CalcParameters["BUNIT"], comment="Brightness (pixel) unit", ext=0)

        if self.debug == True:
            print("The created header is:\n", header)


        ### Modify xml file
        ## update Frequency range inkl. MinExpRange, MaxExpRange, StepFrequency
        modrood.find("file").find("FrequencyRange").find("MinExpRange").text = "%.5e" %(self.MinExpRange)
        modrood.find("file").find("FrequencyRange").find("MaxExpRange").text = "%.5e" %(self.MaxExpRange)
        modrood.find("file").find("FrequencyRange").find("StepFrequency").text = "%.3f" %(self.StepFrequency)

        ## update FileNamesExpFiles
        modrood.find("file").find("FileNamesExpFiles").text = self.CalcParameters['CubePath']+self.CalcParameters["cube_file"]

        ## Save the updated xml file in the parent dir
        modtree.write(self.CalcParameters['DatePath']+self.CalcParameters['xml_file'])

        return
    ##----------------------------------------------------------------------------------------------------------------------------------------------------

    

    ##----------------------------------------------------------------------------------------------------------------------------------------------------
    ##
    ## calculate molfit parameters for ith component
    ##
    def CreateMolfitFile(self):
        """

        input parameters:
        -----------------

            - None


        output parameters:
        ------------------

            - MolfitParameterList:      list of calculated molfit parameters
        """

        ## calculate molfit parameters for the array
        self.CalcMolfitParameters()

        ## Create an empty cube with a header and modify the xml file
        self.CreateEmptyFitsAndModifyXML()

        ## write the results to a molfit file if needed
        if self.CalcParameters["XCLASS"] == True:
            self.WriteMolfit()

        ## debug:
        if self.debug == True:
            print("\npcrit = ", self.pcrit)
            print("x_rot[pcrit] = ", self.Xarray[self.pcrit])
            print("y_rot[pcrit] = ", self.Yarray[self.pcrit])
            print("z_rot[pcrit] = ", self.Zarray[self.pcrit])
            print("SourceSizeArray[pcrit] = ", self.SourceSizeArray[self.pcrit])
            print("TemperatureArray[pcrit] = ", self.TemperatureArray[self.pcrit])
            print("NtotArray[pcrit] = ", self.NtotArray[self.pcrit])
            print("vWidthArray[pcrit] = ", self.vWidthArray[self.pcrit])
            print("VelocityArray[pcrit] = ", self.VelocityArray[self.pcrit])
            print("DistanceObserverArray[pcrit] = ", self.DistanceObserverArray[self.pcrit[2]])


        ## define return parameter
        return self.SourceSizeArray, self.TemperatureArray, self.NtotArray, self.vWidthArray, self.VelocityArray, self.DistanceObserverArray, self.mask
##--------------------------------------------------------------------------------------------------------------------------------------------------------


##--------------------------------------------------------------------------------------------------------------------------------------------------------
##
## script to compute content of molfit file (called from CubeFit function)
##
def Start(CalcParameters):
    """

input parameters:
-----------------

    - XCoord:                   x-coord.  of current pixel

    - YCoord:                   y-coord.  of current pixel

    - XCoordIndex:              x-index of current pixel

    - YCoordIndex:              y-index of current pixel

    - CalcParameters            list containing fit parameters



output parameters:
------------------

    - MolfitFileContents:       content of molfit file
    """
    
    ## A timer to measure the time of the full algorithm, ends with tef
    tsf = time.time()

    # Debug:
    if CalcParameters['Debug'] == True:
        print ("\nNumberXCoord = ", CalcParameters['NumberXCoord'])
        print ("RAStep = ", CalcParameters['RAStep'])
        print("SourceXCoord = ", CalcParameters['SourceXCoord'])
        print ("\nNumberYCoord = ", CalcParameters['NumberYCoord'])
        print ("resolution = ", CalcParameters['resolution'])
        print("SourceYIndex = ", CalcParameters['SourceYIndex'])
        print("SourceYCoord = ", CalcParameters['SourceYCoord'])
        print("\nZn = ", CalcParameters['NumberOfCellsAlongLineOfSight'])
        print ("\nCalcParameters = \n", CalcParameters)
        
    ##----------------------------------------------------------------------------------------------------------------------------------------------------
    ## initialize class MolfitGenerator
    ClassMolfitGenerator = MolfitGenerator(CalcParameters)
    
    ##----------------------------------------------------------------------------------------------------------------------------------------------------
    ## define content of molfit file
    ## A timer to measure the time of the Molfit Generator algorithm, ends with tem
    tsm = time.time()
    SourceSizeArray, TemperatureArray, NtotArray, vWidthArray, VelocityArray, DistanceObserverArray, MaskArray = ClassMolfitGenerator.CreateMolfitFile()
    tem = time.time()

    ### Read the cube header
    header = fits.getheader(CalcParameters['CubePath']+CalcParameters['cube_file'], ext=0)

    DistanceObserverArray3D = numpy.zeros_like(TemperatureArray)
    DistanceObserverArray3D[:] = DistanceObserverArray
    DistanceObserverArray3DMoved = numpy.swapaxes(DistanceObserverArray3D, 2, 0)

    ## A timer to measure the time of writing the cubes, ends with tec
    tsc = time.time()

    ## Modify the header's third axis so that it's in m 
    header_cube = header.copy()
    header_cube["CTYPE3"] = "Distance"
    header_cube["CRVAL3"] = DistanceObserverArray[0]
    header_cube["CDELT3"] = DistanceObserverArray[1] - DistanceObserverArray[0]
    header_cube["CRPIX3"] = 0
    header_cube["CUNIT3"] = "m"  
    header_cube["PROFTIME"] = (str(datetime.timedelta(seconds=numpy.ceil(tem-tsm))), "Profile calculation duration")

    ## Copy the header and remove the 3rd dimension header info for the fits slices
    headersl = header.copy()

    headersl["NAXIS"] = 2
    if "WCSAXES" in headersl:
        headersl["WCSAXES"] = 2

    header3rdaxis = ["NAXIS3", "CTYPE3", "CRVAL3", "CDELT3", "CRPIX3", "CUNIT3", "PC01_03", "PC02_03", "PC03_01", "PC03_02", "PC03_03"]

    for entry in header3rdaxis:
        if entry in headersl:
            del headersl[entry]

        
    ## Create complete fits files 
    ## Check if the Source Size array is not constant
    if numpy.any(SourceSizeArray != SourceSizeArray[0,0,0]):
        ## Shift axis from [x, y, z] --> [z, x, y]
        SourceSizeArraySwapped = numpy.swapaxes(SourceSizeArray, 2, 0)

        ## Modify the header
        header_cube['BTYPE'] = "Source Size"
        header_cube['BUNIT'] = "arcsec"

        ## Create unique name and save the complete array as a fits cube
        SourceSizeName = "BestResults___parameter__source_size___molecule__%s___complete.fits" %(CalcParameters['Molecule'].replace(";","_"))
        hdu = fits.PrimaryHDU(numpy.array(SourceSizeArraySwapped), header=header_cube)
        hdu.writeto(CalcParameters['CubePath']+SourceSizeName, overwrite=True) 

    ## Check if the Temperature array is not constant
    if numpy.any(TemperatureArray != TemperatureArray[0,0,0]):
        TemperatureArraySwapped = numpy.swapaxes(TemperatureArray, 2, 0)

        header_cube['BTYPE'] = "Temperature"
        header_cube['BUNIT'] = "K"

        TemperatureName = "BestResults___parameter__T_rot___molecule__%s___complete.fits" %(CalcParameters['Molecule'].replace(";","_"))

        hdu = fits.PrimaryHDU(numpy.array(TemperatureArraySwapped), header=header_cube)
        hdu.writeto(CalcParameters['CubePath']+TemperatureName, overwrite=True) 
        
    ## Check if the Density array is not constant
    if numpy.any(NtotArray != NtotArray[0,0,0]):
        NtotArraySwapped = numpy.swapaxes(NtotArray, 2, 0)

        header_cube['BTYPE'] = "Surface Density"
        header_cube['BUNIT'] = "cm-2"

        NtotName = "BestResults___parameter__N_tot___molecule__%s___complete.fits" %(CalcParameters['Molecule'].replace(";","_"))

        hdu = fits.PrimaryHDU(numpy.array(NtotArraySwapped), header=header_cube)
        hdu.writeto(CalcParameters['CubePath']+NtotName, overwrite=True) 
        
        ## Check if the Line Width array is not constant
    if numpy.any(vWidthArray != vWidthArray[0,0,0]):
        vWidthArraySwapped = numpy.swapaxes(vWidthArray, 2, 0)

        header_cube['BTYPE'] = "Line Width"
        header_cube['BUNIT'] = "km s-1"

        vWidthName = "BestResults___parameter__V_width_Gaus___molecule__%s___complete.fits" %(CalcParameters['Molecule'].replace(";","_"))

        hdu = fits.PrimaryHDU(numpy.array(vWidthArraySwapped), header=header_cube)
        hdu.writeto(CalcParameters['CubePath']+vWidthName, overwrite=True) 
        
    ## Check if the velocity array is not constant
    if numpy.any(VelocityArray != VelocityArray[0,0,0]):
        VelocityArraySwapped = numpy.swapaxes(VelocityArray, 2, 0)

        header_cube['BTYPE'] = "Velocity"
        header_cube['BUNIT'] = "km s-1"

        VelocityName = "BestResults___parameter__V_off___molecule__%s___complete.fits" %(CalcParameters['Molecule'].replace(";","_"))

        hdu = fits.PrimaryHDU(numpy.array(VelocityArraySwapped), header=header_cube)
        hdu.writeto(CalcParameters['CubePath']+VelocityName, overwrite=True) 

    ## Check if the velocity array is not constant
    if numpy.any(MaskArray != MaskArray[0,0,0]):
        MaskArraySwapped = numpy.swapaxes(MaskArray*1., 2, 0)

        header_cube['BTYPE'] = "Mask"
        header_cube['BUNIT'] = "None"

        MaskName = "BestResults___parameter__Mask___molecule__%s___complete.fits" %(CalcParameters['Molecule'].replace(";","_"))

        hdu = fits.PrimaryHDU(numpy.array(MaskArraySwapped), header=header_cube)
        hdu.writeto(CalcParameters['CubePath']+MaskName, overwrite=True)
        
    
    ## Create fits files and molfit file for each distance if MapFit is True
    if CalcParameters['MapFit'] == True:
        for i in range(CalcParameters['NumberOfCellsAlongLineOfSight']):
                        
            ## Write arrays slices to fits files, same as for the whole cubes
            if numpy.any(SourceSizeArray != SourceSizeArray[0,0,0]):

                headersl['BTYPE'] = "Source Size"
                headersl['BUNIT'] = "arcsec"

                SourceSizeName = "BestResults___parameter__source_size___molecule__%s___component__%i.fits" %(CalcParameters['Molecule'].replace(";","_"), i+1)

                hdu = fits.PrimaryHDU(SourceSizeArraySwapped[i], header=headersl)
                hdu.writeto(CalcParameters['CubePath']+SourceSizeName, overwrite=True) 

            if numpy.any(TemperatureArray != TemperatureArray[0,0,0]):

                headersl['BTYPE'] = "Temperature"
                headersl['BUNIT'] = "K"

                TemperatureName = "BestResults___parameter__T_rot___molecule__%s___component__%i.fits" %(CalcParameters['Molecule'].replace(";","_"), i+1)

                hdu = fits.PrimaryHDU(TemperatureArraySwapped[i], header=headersl)
                hdu.writeto(CalcParameters['CubePath']+TemperatureName, overwrite=True) 
        
            if numpy.any(NtotArray != NtotArray[0,0,0]):

                headersl['BTYPE'] = "Surface Density"
                headersl['BUNIT'] = "cm-2"

                NtotName = "BestResults___parameter__N_tot___molecule__%s___component__%i.fits" %(CalcParameters['Molecule'].replace(";","_"), i+1)

                hdu = fits.PrimaryHDU(NtotArraySwapped[i], header=headersl)
                hdu.writeto(CalcParameters['CubePath']+NtotName, overwrite=True) 
        
            if numpy.any(vWidthArray != vWidthArray[0,0,0]):

                headersl['BTYPE'] = "Line Width"
                headersl['BUNIT'] = "km s-1"

                vWidthName = "BestResults___parameter__V_width_Gaus___molecule__%s___component__%i.fits" %(CalcParameters['Molecule'].replace(";","_"), i+1)

                hdu = fits.PrimaryHDU(vWidthArraySwapped[i], header=headersl)
                hdu.writeto(CalcParameters['CubePath']+vWidthName, overwrite=True) 
        
            if numpy.any(VelocityArray != VelocityArray[0,0,0]):

                headersl['BTYPE'] = "Velocity"
                headersl['BUNIT'] = "m s-1"

                VelocityName = "BestResults___parameter__V_off___molecule__%s___component__%i.fits" %(CalcParameters['Molecule'].replace(";","_"), i+1)

                hdu = fits.PrimaryHDU(VelocityArraySwapped[i], header=headersl)
                hdu.writeto(CalcParameters['CubePath']+VelocityName, overwrite=True) 
        
            if numpy.any(DistanceObserverArray != DistanceObserverArray[0]):

                headersl['BTYPE'] = "Distance"
                headersl['BUNIT'] = "m"

                DistanceObserverArrayName = "BestResults___parameter__dist___molecule__%s___component__%i.fits" %(CalcParameters['Molecule'].replace(";","_"), i+1)

                hdu = fits.PrimaryHDU(DistanceObserverArray3DMoved[i], header=headersl)
                hdu.writeto(CalcParameters['CubePath']+DistanceObserverArrayName, overwrite=True) 

    tec = time.time()

    
    ##----------------------------------------------------------------------------------------------------------------------------------------------------
    ## write molecule molfit file and save it to disk
    ## A timer to measure the time of writing the molfit files, ends with tew
    tsw = time.time()

    ## prepare molfit file
    MoleculeMolfitFile = "%s\t%i\n" %(CalcParameters['Molecule'], CalcParameters['n_molfit_components'])

    ## write n lines
    for i in range(CalcParameters['n_molfit_components'], 0, -1):
        MoleculeMolfitFile += '\tn\t1.0\t100.0\t20.0\tn\t0.1\t5.e9\t400.0\tn\t1.e+01\t1.e+30\t1.e+15\tn\t1.0\t20.0\t5.0\tn\t-500.0\t500.0\t0.0\t%3.5f\n' %(i)

    ## Overwrite the my_map-molecules.molfit file
    MolfitsFileName = CalcParameters['ReadPath'] + "my_map-molecules.molfit"
    MolfitsFile = open(MolfitsFileName, 'w')
    MolfitsFile.write(MoleculeMolfitFile)
    MolfitsFile.close()

    ## Saving a copy in the MolFit Path
    MolfitsFileName = CalcParameters['MolfitPath'] + "my_map-molecules.molfit"
    MolfitsFile = open(MolfitsFileName, 'w')
    MolfitsFile.write(MoleculeMolfitFile)
    MolfitsFile.close()

    ## Save the used parameters in the date path
    header_Data = ["Model name", "Theta [deg]", "Phi [deg]", "v [km/s]", "n [cm-3]", "T [K]", "alpha", "dmin [au]", "sigmaof", 
                   "ThetaMax [deg]", "epsilon", "gamma", "M1", "beta", "r0 [au]", "z0 [au]", "Label"]
    Outflow_Data = [[CalcParameters['modelname'], CalcParameters['Theta'], CalcParameters['Phi'], CalcParameters['vmax'], 
                    CalcParameters['nmin'], CalcParameters['Tmin'], CalcParameters['alpha'], CalcParameters['dmin'], CalcParameters['sigmaof'],
                    CalcParameters['ThetaMax'], CalcParameters['epsilon'], CalcParameters['gamma'], CalcParameters['M1'], CalcParameters['beta'],
                    CalcParameters['r0'], CalcParameters['z0'], CalcParameters['Label']]]

    dataframe = pandas.DataFrame(data=Outflow_Data, index=["Sample"], columns=header_Data)

    dataframe.to_csv(CalcParameters['DatePath'] + "Outflow_Data.dat")
    
    tew = time.time()
    tef = time.time()

    print("Run times:")
    print("Total run time =\t%s" %(str(datetime.timedelta(seconds=numpy.ceil(tef-tsf)))))
    print("Calculate data =\t%s" %(str(datetime.timedelta(seconds=numpy.ceil(tem-tsm)))))
    print("Write cubes time =\t%s" %(str(datetime.timedelta(seconds=numpy.ceil(tec-tsc)))))
    print("Write molfit files time =\t%s\n\n" %(str(datetime.timedelta(seconds=numpy.ceil(tew-tsw)))))

    ## define return parameter
    return
##--------------------------------------------------------------------------------------------------------------------------------------------------------

## Function to turn off all output via print (I'm sorry Thomas...)
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

## Restore the output
def enablePrint():
    sys.stdout = sys.__stdout__

##--------------------------------------------------------------------------------------------------------------------------------------------------------

## Function to calculate the scale bar properties
def scale_bar(l, d, cdelt1):
    lsbm = abs(d * numpy.tan(abs(cdelt1*l/4)/180*numpy.pi))
    lsb = numpy.floor(lsbm/10**numpy.floor(numpy.log10(lsbm)))*10**numpy.floor(numpy.log10(lsbm))
    asb = (lsb * u.pc / d / u.pc).to(u.deg, equivalencies=u.dimensionless_angles())

    if 3 > numpy.floor(numpy.log10(lsb)) >= 0:
        return asb, "%i pc" %(lsb)
    elif -1 == numpy.floor(numpy.log10(lsb)):
        return asb, "%.1f pc" %(lsb)
    elif -2 == numpy.floor(numpy.log10(lsb)):
        return asb, "%.2f pc" %(lsb)
    else:
        return asb, "%.0e pc" %(lsb)

##--------------------------------------------------------------------------------------------------------------------------------------------------------
##
## start main program 
##
def Main(MasterFile):
    print("\nStarting the Python code '%s'.\n\n" %(os.path.basename(__file__)))
    tst = time.time()

    ##----------------------------------------------------------------------------------------------------------------------------------------------------
    ## get the operating system
    system = platform.system()

    ## Set important commands
    if system == "Linux":
        rmfi = "rm"         # Remove file
        rmdi = "rm -r"      # Remove directory
        mvfd = "mv"         # Move file or directory
    
    elif system == "Windows":
        rmfi = "del"        # Remove file
        rmdi = "rmdir"      # Remove directory
        mvfd = "move"       # Move file or directory

    if not (system == "Linux" or system == "Windows"):
        print("The operating system is neither Linux or Windows.")
        print("Try using Linux commands.")
        rmfi = "rm"         # Remove file
        rmdi = "rm -r"      # Remove directory
        mvfd = "mv"         # Move file or directory

    
    ##----------------------------------------------------------------------------------------------------------------------------------------------------
    # extend sys.path variable


    # get path of XCLASS directory
    XCLASSRootDir = str(os.environ.get('XCLASSRootDir', '')).strip()
    XCLASSRootDir = os.path.normpath(XCLASSRootDir) + "/"


    # extend sys.path variable
    NewPath = XCLASSRootDir + "build_tasks/"
    if (not NewPath in sys.path):
        sys.path.append(NewPath)


    # import XCLASS packages
    try:
        import task_myXCLASSMapFit
        import task_myXCLASS

    except:
        print("Couldn't import task_myXCLASSMapFit and/or task_myXCLASS. So the algorithm just calculates the cubes but does not fit them!")
          

    ##----------------------------------------------------------------------------------------------------------------------------------------------------
    ## Outflow Master File
    MasterFileDir = str(Path(__file__).parent.resolve()) + "/"
        
    ## Read the master tree
    try:
        MasterTree = ET.parse(MasterFileDir+MasterFile)

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


    ## Get name of the local machine
    namelocal = MasterTree.find("namelocal").text

    ## Give the sub roots for a local machine and a server
    if platform.node() == namelocal:
        flagroot = MasterTree.find("flags")
        pathroot = MasterTree.find("pathslocal")
        fileroot = MasterTree.find("fileslocal")
        pararoot = MasterTree.find("parameters")

    else:
        flagroot = MasterTree.find("flags")
        pathroot = MasterTree.find("pathsserver")
        fileroot = MasterTree.find("filesserver")
        pararoot = MasterTree.find("parameters")
    
    ##----------------------------------------------------------------------------------------------------------------------------------------------------
    ## define model parameters similar to G29


    ## initialize dictionary for model parameters
    MyParameters = {}

    ## Get the given flags
    MyParameters['CalculateModel'] = strtobool(flagroot.find("CalculateModel").text)
    MyParameters['Debug'] = strtobool(flagroot.find("Debug").text)
    MyParameters['UpdateDataBase'] = strtobool(flagroot.find("UpdateDataBase").text)
    MyParameters['Plot'] = strtobool(flagroot.find("Plot").text)
    MyParameters['XCLASS'] = strtobool(flagroot.find("XCLASS").text)
    MyParameters['MapFit'] = strtobool(flagroot.find("MapFit").text)
    MyParameters['FixRandom'] = strtobool(flagroot.find("FixRandom").text)
    MyParameters['SmoothGaussian'] = strtobool(flagroot.find("SmoothGaussian").text)
    MyParameters['Cloud'] = strtobool(flagroot.find("Cloud").text)
    MyParameters['WhiteNoise'] = strtobool(flagroot.find("WhiteNoise").text)
    MyParameters['UploadPlot'] = strtobool(flagroot.find("UploadPlot").text)
    MyParameters['printXCLASS'] = strtobool(flagroot.find("printXCLASS").text)
    MyParameters['printMapFit'] = strtobool(flagroot.find("printMapFit").text)
    MyParameters['printThumbsPage'] = strtobool(flagroot.find("printThumbsPage").text)
    MyParameters['printWarnings'] = strtobool(flagroot.find("printWarnings").text)
    MyParameters['ProductionRun'] = strtobool(flagroot.find("ProductionRun").text)

    if MyParameters['printWarnings'] == False:
        warnings.filterwarnings("ignore")


    ## Debug
    if MyParameters["Debug"] == True:
        print("Used flags:")
        print("CalculateModel: %s" %MyParameters['CalculateModel'])
        print("Debug: %s" %MyParameters['Debug'])
        print("Plot: %s" %MyParameters['Plot'])
        print("XCLASS: %s" %MyParameters['XCLASS'])
        print("MapFit: %s" %MyParameters['MapFit'])
        print("MapFit: %s" %MyParameters['FixRandom'])
        print("SmoothGaussian: %s" %MyParameters['SmoothGaussian'])
        print("Cloud: %s" %MyParameters['Cloud'])
        print("WhiteNoise: %s" %MyParameters['WhiteNoise'])
        print("UploadPlot: %s" %MyParameters['UploadPlot'])
        print("printXCLASS: %s" %MyParameters['printXCLASS'])
        print("printMapFit: %s" %MyParameters['printMapFit'])
        print("printThumbsPage: %s" %MyParameters['printThumbsPage'])
        print("printWarnings: %s" %MyParameters['printWarnings'])
        print("ProductionRun: %s" %MyParameters['ProductionRun'])

    ## read in relevant stuff from the MasterFile
    ## read the paths from xml file 
    MyParameters['SavePath'] = pathroot.find("SavePath").text                                   ## Path to save in the results
    MyParameters['ReadPath'] = pathroot.find("ReadPath").text                                   ## Path to read some files
    MyParameters['DatabasePath'] = pathroot.find("DatabasePath").text                           ## Path to the database
    MyParameters['CloudPath'] = pathroot.find("CloudPath").text                                 ## Path to the simulated cloud
    MyParameters['ExistingCalculationDir'] = pathroot.find("ExistingCalculation").text          ## Path to existing an existing calculation
    MyParameters['LocalThumbsPagePath'] = pathroot.find("LocalThumbsPagePath").text             ## Path to the root of the local thumbs pages to display relevant images
    MyParameters['ThumbsPageScriptPath'] = pathroot.find("ThumbsPageScriptPath").text           ## Path to the thumbs page script
    MyParameters['UploadPath'] = pathroot.find("UploadPath").text                               ## Path to root of the generated upload directory


    ## read the file names from xml file
    MyParameters['xml_file'] = fileroot.find("xml_file").text                                   ## Name of the outflow xml file (NOT THE MASTER FILE!)
    MyParameters['cube_file'] = fileroot.find("cube_file").text                                 ## Name of the cube file
    MyParameters['name_cloud'] = fileroot.find("cloud_file").text                               ## Name of the simulated cloud
    MyParameters['data_file'] = fileroot.find('data_file').text                                 ## Name of the data file
    MyParameters['database_file'] = fileroot.find('database_file').text                         ## Name of the database file
    MyParameters['thumbspage_script'] = fileroot.find('ThumbsPageScript').text                  ## Name of the thumbspage script


    ## read in model parameters
    MyParameters['NumberOfCellsAlongLineOfSight'] = int(pararoot.find("ZSteps").text)           ## define number of cells along line of sight
    MyParameters['DistanceToSource'] = float(pararoot.find("Distance").text)                    ## distance to source
    MyParameters['MaxResolution'] = float(pararoot.find("MaxResolution").text)                  ## maximal resolution of telescope
    MyParameters['MaxXYSteps'] = int(pararoot.find("MaxXYSteps").text)                          ## maximal number of cells in x and y direction


    ## read in parameters used for the cube header 
    MyParameters["BUNIT"] = pararoot.find("Bunit").text                                         ## Intensity unit of fit
    MyParameters["RestFreq"] = float(pararoot.find("RestFreq").text)                            ## Rest frequency in MHz
    MyParameters["MinFSteps"] = int(pararoot.find("MinFSteps").text)                            ## Minimal number of steps along frequency
    MyParameters["FrequencyStep"] = float(pararoot.find("FrequencyStep").text)                  ## Frequency step size in MHz, might be smaller
    MyParameters["MinFExtend"] = float(pararoot.find("FreqRange").text)                         ## Minimal frequency extend in MHz
    MyParameters['BMAJ'] = float(pararoot.find("BMAJ").text)                                    ## define major beam axis in deg
    MyParameters['BMIN'] = float(pararoot.find("BMIN").text)                                    ## define minor beam axis in deg
    MyParameters['BPA'] = float(pararoot.find("BPA").text)                                      ## define beam orientation in deg


    ## read in other parameters
    MyParameters['Molecule'] = pararoot.find("Molecule").text                                   ## define molecule
    MyParameters['n_molfit_components'] = int(MyParameters['NumberOfCellsAlongLineOfSight'])    ## number of components
    MyParameters['ssComp'] = float(pararoot.find("ssComp").text)                                ## define source size
    MyParameters['vWidthComp'] = float(pararoot.find("vWidthComp").text)                        ## define line width
    MyParameters['nbackground'] = float(pararoot.find("nbackground").text)                  ## define background density in cm-3
    MyParameters['Tbackground'] = float(pararoot.find("Tbackground").text)                      ## define background temperature in K
    MyParameters['MinCloudV'] = float(pararoot.find("MinCloudV").text)                          ## define minimal cloud velocity in km/s
    MyParameters['MaxCloudV'] = float(pararoot.find("MaxCloudV").text)                          ## define maximal cloud velocity in km/s
    MyParameters['CloudLoc'] = float(pararoot.find("CloudLoc").text)                            ## define mean cloud max intensity in K
    MyParameters['CloudScale'] = float(pararoot.find("CloudScale").text)                        ## define standard deviation of cloud max intensity in K
    MyParameters['WNLoc'] = float(pararoot.find("WNLoc").text)                                  ## define mean white noise intensity in K 
    MyParameters['WNScale'] = float(pararoot.find("WNScale").text)                              ## define standard deviation of white noise in K
    

    ## Get the number of processes
    MyParameters['npro'] = float(pararoot.find("npro").text)                                    ## define number of used processes
    MyParameters['npro'] = min(MyParameters['npro'], int(numpy.floor(mp.cpu_count()*.8)))       ## ensure that the number is smaller than 80% of the computer cores

    ## Set the random seed if requested
    if MyParameters["FixRandom"] == True:
        numpy.random.seed(42)        


    ## Get the name of the database and completing the local thumbspage Path and upload path
    database_name = ".".join(MyParameters['database_file'].split(".")[:-1])
    MyParameters["LocalThumbsPagePath"] = "%s%s/" %(MyParameters["LocalThumbsPagePath"], database_name)
    MyParameters["UploadPath"] = "%s%s/" %(MyParameters["UploadPath"], database_name)


    ##--------------------------------------------------------------------------------------------------------------------------------------------------------
    ## read in the relevant parameters

    ## try to read the data file
    try:
        data = pandas.read_csv(MyParameters['ReadPath']+MyParameters['data_file'])
        if MyParameters["CalculateModel"] == True:
            print("The data will be read from the data file '%s'.\n\n\n" %(MyParameters['data_file']))

    except Exception as e:
        exception_type, exception_object, exception_traceback = sys.exc_info()
        filename = exception_traceback.tb_frame.f_code.co_filename
        line_number = exception_traceback.tb_lineno
        
        print("\nReading the data from the %s file failed." %(MyParameters['data_file']))
        print("Exception type:\t%s" %(exception_type))
        print("File name:\t%s" %(filename))
        print("Line number:\t%s" %(line_number))
        print("The error itself:\n%s\n\n" %(e))        
        print("\n\nPlease indicate a valid data file.")



    for i in range(len(data.index)):

        if MyParameters["CalculateModel"] == False and i == 1:

            print("Trying to run a loop but the code uses an existing calculation. Cancel this calculation!")

            break
            
        
        elif MyParameters["CalculateModel"] == True:
            print("Run data set %i from %i." %(i+1, len(data.index)))
            print(data.iloc[i])

        MyParameters['modelname'] = data.iloc[i]["Model name"]                                                      ## Name of the model; Cabrit or Raga

        ## define rotation angles in °
        MyParameters['Theta'] = data.iloc[i]["Theta [deg]"]
        MyParameters['Phi'] = data.iloc[i]["Phi [deg]"]

        ## label of dataset
        MyParameters['Label'] = data.iloc[i]["Label"]


        ##--------------------------------------------------------------------------------------------------------------------------------------------------------
        ## define model parameters for analyitc expression of Cabrit et al
        
    
        ## general parameters
        MyParameters['alpha'] = data.iloc[i]["alpha"]                                                               ## velocity exponent

        ## outflow shape parameters 
        MyParameters['dmin'] = data.iloc[i]["dmin [au]"]                                                            ## Minimal outflow range in au
        MyParameters['sigmaof'] = data.iloc[i]["sigmaof"]                                                           ## Factor that describes the length of the outflow in units of the inner distance
        MyParameters['ThetaMax'] = data.iloc[i]["ThetaMax [deg]"]                                                   ## Maximal opening angle in °

        ## initial distribution parameters
        MyParameters['vmax'] = data.iloc[i]["v [km/s]"]                                                             ## initial jet velocity in km/s
        MyParameters['nmin'] = data.iloc[i]["n [cm-3]"]                                                         ## initial density in cm-3
        MyParameters['Tmin'] = data.iloc[i]["T [K]"]                                                                ## inítial temperature of the mixing-layer in K


        ##--------------------------------------------------------------------------------------------------------------------------------------------------------
        ## define model parameters for analytic expressions of Raga et al


        ## general parameters
        MyParameters['epsilon'] = data.iloc[i]["epsilon"]                                                           ## Jet efficency
        MyParameters['gamma'] = data.iloc[i]["gamma"]                                                               ## Molecular gas
        MyParameters['M1'] = data.iloc[i]["M1"]                                                                     ## Mixing-layer sound speed velocity ratio

        ## outflow shape parameters
        MyParameters['beta'] = data.iloc[i]["beta"]                                                                 ## Shape of the outflow 
        MyParameters['r0'] = data.iloc[i]["r0 [au]"]                                                                ## Cavity radius at the working surface in au
        MyParameters['z0'] = data.iloc[i]["z0 [au]"]                                                                ## Present position of the working surface in au

        ## initial distribution parameters
        MyParameters['vj'] = data.iloc[i]["v [km/s]"]                                                               ## initial jet velocity in km/s
        MyParameters['n0'] = data.iloc[i]["n [cm-3]"]                                                           ## initial density in cm-3
        MyParameters['T1'] = data.iloc[i]["T [K]"]                                                                  ## inítial temperature of the mixing-layer in K


        #-----------------------------------------------------------------------------------------------------------------------------------------------------
        ## create current and all used paths
        ## Time-stamp for parent dir
        currenttime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") # Get the date to create unique dirs

        ## Specify the name of the parent directory from current or former used parameters
        if MyParameters['CalculateModel'] == True:
            MyParameters['NameDir'] = currenttime + "_model-%s_Theta-%.3fdeg_Phi-%.3fdeg_v-%.3fkms-1_n-%.3fcm-3f_T-%.3fK/" \
                %(MyParameters['modelname'], MyParameters['Theta'], MyParameters['Phi'], MyParameters['vmax'], MyParameters['nmin'], MyParameters['Tmin'])

        else:
            MyParameters['NameDir'] = currenttime + MyParameters['ExistingCalculationDir'].split("/")[-2][19:] + "/"


        ## Name all other files
        MyParameters['OutputPath'] = MyParameters['SavePath'] + "Output/"
        MyParameters['DatePath'] = MyParameters['OutputPath'] + MyParameters['NameDir']
        MyParameters['CubePath'] = MyParameters['DatePath'] + "Cubes/"
        MyParameters['MolfitPath'] = MyParameters['DatePath'] + "Molfit/"
        MyParameters['MapFitPath'] = MyParameters['DatePath'] + "MapFit/"
        MyParameters['PlotPath'] = MyParameters['DatePath'] + "Plot/"
        MyParameters['myXCLASSPath'] = MyParameters['DatePath'] + "myXCLASS/"

        if MyParameters['CalculateModel'] == True:
            MyParameters['ReadMapFit'] = MyParameters['MapFitPath']

        else:
            MyParameters['ReadMapFit'] = MyParameters['ExistingCalculationDir'] + "MapFit/"


        ## Create all necessary paths
        if not os.path.isdir(MyParameters['OutputPath']):
            Path(MyParameters['OutputPath']).mkdir(parents=True)
            #os.mkdir(MyParameters['OutputPath'])

        if not os.path.isdir(MyParameters['DatePath']):
            Path(MyParameters['DatePath']).mkdir(parents=True)
            #os.mkdir(MyParameters['DatePath'])

        if not os.path.isdir(MyParameters['MapFitPath']):
            Path(MyParameters['MapFitPath']).mkdir(parents=True)
            #os.mkdir(MyParameters['MapFitPath'])

        if not os.path.isdir(MyParameters['CubePath']) and MyParameters['CalculateModel'] == True:
            Path(MyParameters['CubePath']).mkdir(parents=True)
            #os.mkdir(MyParameters['CubePath'])

        if not os.path.isdir(MyParameters['MolfitPath']) and MyParameters['CalculateModel'] == True:
            Path(MyParameters['MolfitPath']).mkdir(parents=True)
            #os.mkdir(MyParameters['MolfitPath'])
        
        if not os.path.isdir(MyParameters['PlotPath']) and MyParameters['Plot'] == True:
            Path(MyParameters['PlotPath']).mkdir(parents=True)
            #os.mkdir(MyParameters['PlotPath'])
        
        if not os.path.isdir(MyParameters['myXCLASSPath']) and MyParameters['CalculateModel'] == True and MyParameters['XCLASS'] == True:
            Path(MyParameters['myXCLASSPath']).mkdir(parents=True)
            #os.mkdir(MyParameters['myXCLASSPath'])

        print("All files will be saved at: %s" %(MyParameters['DatePath']))

        #-----------------------------------------------------------------------------------------------------------------------------------------------------

        ## Calculate Model
        if MyParameters['CalculateModel'] == True:

            ## Start the parameter calculation timer
            tss = time.time()
            
            ## start script to create molfit file
            Start(MyParameters)
            
            ## End and print the parameter calculation timer
            tes = time.time()
            print("Creating the outflow profile cubes took %s." %(str(datetime.timedelta(seconds=numpy.ceil(tes-tss)))))
     
            # Fitting spectra with myXCLASS task
            if MyParameters['XCLASS'] == True:

                print("Starting the XCLASS function.")
                tsx = time.time()

                ## Read the xml file
                xmlTree = ET.parse(MyParameters['DatePath']+MyParameters['xml_file'])
                FreqRangeroot = xmlTree.getroot().find("file").find("FrequencyRange")

                header_cube = fits.getheader(MyParameters['CubePath']+MyParameters["cube_file"], ext=0)

                xr = header_cube["NAXIS1"]
                yr = header_cube["NAXIS2"]

                R = min(xr, yr) / 2

                ###########################################################################
                # TO MODIFY BY THE USER

                # define min. freq. (in MHz)
                FreqMin = float(FreqRangeroot.find("MinExpRange").text)

                # define max. freq. (in MHz)
                FreqMax = float(FreqRangeroot.find("MaxExpRange").text)

                # define freq. step (in MHz)
                FreqStep = float(FreqRangeroot.find("StepFrequency").text)

                # depending on parameter "Inter_Flag" define beam size (in arcsec)
                # (Inter_Flag = True) or size of telescope (in m) (Inter_Flag = False)
                TelescopeSize = 1.7769575715066

                # define beam minor axis length (in arsec)
                BMIN = None

                # define beam major axis length (in arsec)
                BMAJ = None

                # define beam position angle (in degree)
                BPA = None

                # interferrometric data?
                Inter_Flag = True

                # define red shift
                Redshift = None

                # BACKGROUND: describe continuum with tBack and tslope only
                t_back_flag = True

                # BACKGROUND: define background temperature (in K)
                tBack = 0.0

                # BACKGROUND: define temperature slope (dimensionless)
                tslope = 0.0

                # BACKGROUND: define path and name of ASCII file describing continuum as function
                #             of frequency
                BackgroundFileName = ""

                # DUST: define hydrogen column density (in cm^(-2))
                N_H = 1.e24

                # DUST: define spectral index for dust (dimensionless)
                beta_dust = 0.0

                # DUST: define kappa at 1.3 mm (cm^(2) g^(-1))
                kappa_1300 = 0.0

                # DUST: define path and name of ASCII file describing dust opacity as
                #       function of frequency
                DustFileName = ""

                # FREE-FREE: define electronic temperature (in K)
                Te_ff = None

                # FREE-FREE: define emission measure (in pc cm^(-6))
                EM_ff = None

                # SYNCHROTRON: define kappa of energy spectrum of electrons (electrons m^(−3) GeV^(-1))
                kappa_sync = None

                # SYNCHROTRON: define magnetic field (in Gauss)
                B_sync = None

                # SYNCHROTRON: energy spectral index (dimensionless)
                p_sync = None

                # SYNCHROTRON: thickness of slab (in au)
                l_sync = None

                # PHEN-CONT: define phenomenological function which is used to describe
                #            the continuum
                ContPhenFuncID = None

                # PHEN-CONT: define first parameter for phenomenological function
                ContPhenFuncParam1 = None

                # PHEN-CONT: define second parameter for phenomenological function
                ContPhenFuncParam2 = None

                # PHEN-CONT: define third parameter for phenomenological function
                ContPhenFuncParam3 = None

                # PHEN-CONT: define fourth parameter for phenomenological function
                ContPhenFuncParam4 = None

                # PHEN-CONT: define fifth parameter for phenomenological function
                ContPhenFuncParam5 = None

                # use iso ratio file?
                iso_flag = True

                # define path and name of iso ratio file
                IsoTableFileName = MyParameters['ReadPath'] + "my_isonames.txt"

                # define path and name of file describing Non-LTE parameters
                CollisionFileName = ""

                # define number of pixels in x-direction (used for sub-beam description)
                NumModelPixelXX = 100

                # define number of pixels in y-direction (used for sub-beam description)
                NumModelPixelYY = 100

                # take local-overlap into account or not
                LocalOverlapFlag = False

                # disable sub-beam description
                NoSubBeamFlag = True

                # define path and name of database file
                dbFilename = ""

                # define rest freq. (in MHz)
                RestFreq = MyParameters["RestFreq"]

                # define v_lsr (in km/s)
                vLSR = 0.0
                ###########################################################################

                individual_path = MyParameters['myXCLASSPath']
                if not os.path.isdir(individual_path):
                    Path(individual_path).mkdir(parents=True)

                ## create figure
                fig = pylab.figure(figsize = (15, 10))                                                  ## create figure with width 15 inch and height 10 inch
                fig.clear()
                pylab.subplots_adjust(hspace = 0.2, wspace = 0.2)                                       ## adjust plot
                layer = pylab.subplot(1, 1, 1)                                                          ## create one plot
                layer.grid(True)                                                                        ## show grid
                layer.set_title("Test spectrum")                                                        ## title of plot
                layer.set_ylabel("Brightness temperature (K)")                                          ## label of y axis
                layer.set_xlabel("Velocity (m/s)")                                                      ## label of x axis


                RFs = [-0.8, -0.6, -0.4, -0.2, 0, 0.1, 0.25, 0.5, 0.75, 1]

                for RF in RFs:
                    
                    xi = int(R*RF*numpy.cos(MyParameters["Phi"]*numpy.pi/180) + xr/2)
                    yi = int(R*RF*numpy.sin(MyParameters["Phi"]*numpy.pi/180) + yr/2)

                    # define path and name of molfit file
                    MolfitsFileName = MyParameters['MolfitPath'] + "%i-%i_model.molfit" %(xi, yi)

                    if MyParameters["printXCLASS"] == False:
                        blockPrint() 

                    ## call myXCLASS function
                    modeldata, log, TransEnergies, IntOpt, JobDir = task_myXCLASS.myXCLASS(
                                                    FreqMin, FreqMax, FreqStep,
                                                    TelescopeSize, BMIN, BMAJ,
                                                    BPA, Inter_Flag, Redshift,
                                                    t_back_flag, tBack, tslope,
                                                    BackgroundFileName,
                                                    N_H, beta_dust, kappa_1300,
                                                    DustFileName, Te_ff, EM_ff,
                                                    kappa_sync, B_sync, p_sync,
                                                    l_sync, ContPhenFuncID,
                                                    ContPhenFuncParam1,
                                                    ContPhenFuncParam2,
                                                    ContPhenFuncParam3,
                                                    ContPhenFuncParam4,
                                                    ContPhenFuncParam5,
                                                    MolfitsFileName, iso_flag,
                                                    IsoTableFileName,
                                                    CollisionFileName,
                                                    NumModelPixelXX,
                                                    NumModelPixelYY,
                                                    LocalOverlapFlag,
                                                    NoSubBeamFlag,
                                                    dbFilename,
                                                    RestFreq, vLSR)


                    if MyParameters["printXCLASS"] == False:
                        enablePrint()


                    ## add spectrum, stored in 'modeldata'
                    layer.plot(modeldata[:, 1], modeldata[:, 2], '-', linewidth = 2.0, label="Pixel (%i|%i)" %(xi, yi))

                    individual_path = MyParameters['myXCLASSPath'] + "Pixel_%i-%i/" %(xi, yi)
                    if not os.path.isdir(individual_path):
                        Path(individual_path).mkdir(parents=True)

                    ## Move jobdir to Output directory and remove the empty folder afterwards
                    cmdMVJString = "%s %s* %s" %(mvfd, JobDir, individual_path)
                    cmdRMJString = "%s %s" %(rmdi, JobDir)

                    ## Remove unnecessary optical depth and intensity files
                    cmdRMODString = "%s %soptical_depth*" %(rmfi, individual_path)
                    cmdRMIString = "%s %sintensity*" %(rmfi, individual_path)


                    if MyParameters['Debug'] == True:
                        print(cmdMVJString)
                        print(cmdRMJString)
                        print(cmdRMODString)
                        print(cmdRMIString)

                    os.system(cmdMVJString)
                    os.system(cmdRMJString)
                    os.system(cmdRMODString)
                    os.system(cmdRMIString)

                vmin, vmax = SpectralCoord((FreqMax, FreqMin) * u.MHz,
                                doppler_convention='radio',
                                doppler_rest=RestFreq * u.MHz).to(u.km/u.s)
                layer.set_xlim(vmin.value, vmax.value)                                                  ## set limits of x axis
                layer.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.5e'))           ## reformat format of number of x-axis
                layer.legend()                                                                          ## add legend
                pylab.savefig(MyParameters['PlotPath'] + "/spectrum.png", dpi = 300)                    ## save figure
                pylab.draw()                                                                            ## draw figure
                pylab.close(fig)

                ## Remove Molfit directory
                cmdStringRMMolfit = "%s %sMolfit/" %(rmdi, MyParameters['DatePath'])
                os.system(cmdStringRMMolfit)


                print("Finished XCLASS-Part")
                tex = time.time()

                print("The fit of %i XCLASS-spectra took %s" %(len(RFs), str(datetime.timedelta(seconds=numpy.ceil(tex-tsx)))))


            else:
                print("Skipped XCLASS.")


            # Fitting spectra with myXCLASSMapFit task
            if MyParameters['MapFit'] == True:
                tsm = time.time()
                ###########################################################################
                # TO MODIFY BY THE USER
                print("Starting the MapFit function.")
                # define path and name of molfit file
                MolfitsFileName = MyParameters['ReadPath'] + "my_map-molecules.molfit"

                # define path and name of obs. data file
                ObsXMLFileName = MyParameters['DatePath'] + MyParameters['xml_file']

                # define path and name of algorithm xml file
                AlgorithmXMLFileName = MyParameters['ReadPath'] + "my_algorithm-settings.xml"

                # define the number of used processors
                NumberProcessors = MyParameters['npro']

                ## use fast-fitting method?
                FastFitFlag = True

                ## use full-fitting method?
                FullFitFlag = False

                # define path and name of region file
                regionFileName = MyParameters['ReadPath'] + "my_map-region.reg"

                # define lower limit for intensity (pixel with max. intensity below given
                # limit are ignored)
                Threshold = 0.0

                # define path and name of so-called cluster file
                clusterdef = MyParameters['ReadPath'] + "clusterdef.txt"

                # define path and name of region file
                #reionfilename = TBD

                # define number of iterations to smooth parameter maps
                ParamMapIterations = 1

                # define parameter used for smoothing
                ParamSmoothMethodParam = 1.0

                # define scipy method ("gaussian", "uniform", "median") used for parameter
                # map smoothing
                ParamSmoothMethod = "uniform"

                # define path of a directory containing FITS images describing parameter
                # maps to update the parameter defined in the molfit file for each pixel
                ParameterMapDir = MyParameters['CubePath']
                ###########################################################################

                # call myXCLASSMapFit function
                if MyParameters["printMapFit"] == False:
                    blockPrint() 
                
                JobDir = task_myXCLASSMapFit.myXCLASSMapFitCore( \
                                            MolfitsFileName = MolfitsFileName,
                                            ObsXMLFileName = ObsXMLFileName,
                                            FastFitFlag = FastFitFlag,
                                            FullFitFlag = FullFitFlag,
                                            AlgorithmXMLFileName = AlgorithmXMLFileName,
                                            clusterdef = clusterdef,
                                            # regionFileName = regionFileName,
                                            Threshold = Threshold,
                                            ParameterMapDir = ParameterMapDir,
                                            ParamMapIterations = ParamMapIterations,
                                            ParamSmoothMethodParam = ParamSmoothMethodParam,
                                            ParamSmoothMethod = ParamSmoothMethod,
                                            NumberProcessors = NumberProcessors)

                if MyParameters["printMapFit"] == False:
                    enablePrint()


                ## create a new directory for the fits
                if not os.path.isdir(MyParameters['MapFitPath']):
                    Path(MyParameters['MapFitPath']).mkdir(parents=True)

                # move the fits to this directory
                cmdStringMove = "%s %s* %s" %(mvfd, JobDir, MyParameters['MapFitPath'])
       
                # remove empty JobDir
                cmdStringDelete = "%s %s" %(rmdi, JobDir)
                
                # remove unnecessary files to save storage
                cmdStringRenameCubes = "%s %s %sCubes_Temp/" %(mvfd, MyParameters['CubePath'], MyParameters['DatePath'])
                cmdStringMVcomplete = "%s %sCubes_Temp/*complete* %s" %(mvfd, MyParameters['DatePath'], MyParameters['CubePath'])
                cmdStringRMCubesTemp = "%s %sCubes_Temp/" %(rmdi, MyParameters['DatePath'])

                # remove Molfit Dir to save storage
                cmdStringRMMolFit = "%s %s" %(rmdi, MyParameters['MolfitPath'])

                if MyParameters['Debug'] == True:
                    print("Move directory:\n%s\n" %(cmdStringMove))
                    print("Delete empty job directory:\n%s\n" %(cmdStringDelete))
                    print("Remove unused fits files I:\n%s\n" %(cmdStringRenameCubes))
                    print("Remove unused fits files II:\n%s\n" %(cmdStringMVcomplete))
                    print("Remove unused fits files III:\n%s\n" %(cmdStringRMCubesTemp))
                    print("Remove MolFit files:\n%s\n" %(cmdStringRMMolFit))

                # execute strings
                try:
                    os.system(cmdStringMove)                             # Move the files from the JobDir to the main directory
                except:
                    print("Failed to execute:\n%s\n\n" %(cmdStringMove))

                try:
                    os.system(cmdStringDelete)                           # Delete the JobDir
                except:
                    print("Failed to execute:\n%s\n\n" %(cmdStringDelete))
                    
                try:
                    os.system(cmdStringRenameCubes)                      # Rename the Cubes dir to a temporary directory
                except:
                    print("Failed to execute:\n%s\n\n" %(cmdStringRenameCubes))
                    
                if not os.path.isdir(MyParameters['CubePath']):          # Recreate the CubePath
                    Path(MyParameters['CubePath']).mkdir(parents=True)
                    #os.mkdir(MyParameters['CubePath'])

                try:
                    os.system(cmdStringMVcomplete)                       # Move the cubes containing "complete" back to the Cube dir
                except:
                    print("Failed to execute:\n%s\n\n" %(cmdStringMVcomplete))

                try:
                    os.system(cmdStringRMCubesTemp)                      # Remove the Cubes_Temp dir --> all cube slices
                except:
                    print("Failed to execute:\n%s\n\n" %(cmdStringRMCubesTemp))
                    
                try:
                    os.system(cmdStringRMMolFit)                         # Remove the MolFit Files
                except:
                    print("Failed to execute:\n%s\n\n" %(cmdStringRMMolFit))



                tem = time.time()

                ## Add the profile creation and MapFit duration to the header
                fits.setval(MyParameters['MapFitPath']+'%s__model.out.fits' %(MyParameters["cube_file"][:-5]),
                           "PROFTIME", value=str(datetime.timedelta(seconds=numpy.ceil(tes-tss))), comment="Profile calculation duration", ext=0)
                fits.setval(MyParameters['MapFitPath']+'%s__model.out.fits' %(MyParameters["cube_file"][:-5]),
                           "MAPFTIME", value=str(datetime.timedelta(seconds=numpy.ceil(tem-tsm))), comment="MapFit duration", ext=0)

                print("The MapFit-fit took %s" %(str(datetime.timedelta(seconds=numpy.ceil(tem-tsm)))))

            else:
                print("Skipped cube calculation and XCLASS fits.")
                print("Instead using the fits from %s" %(MyParameters["MapFitPath"]))
                

        ## Try reading in the model data with Spectral Cube
        try:
            full_cube = SpectralCube.read('%s%s__model.out.fits' %(MyParameters['ReadMapFit'], MyParameters["cube_file"][:-5]))
            header_model = full_cube.header
            wcs = WCS(header_model)
            read_success = True

        ## If uncussessfull, skip the remaining calculations of this run
        except:
            print("Warning!\nIt was not possible to read in the MapFit data cube.\nSkipping the remaining calculations.")
            read_success = False

        if read_success == True:
            ## Measure the modification time
            tsv = time.time()

            ## Apply a gaussian filter to smooth the data
            if MyParameters['SmoothGaussian'] == True:

                ## Print out a warning if the resolution is worse than beamsize
                if header_model["CDELT2"] > header_model["BMAJ"] or header_model["CDELT1"] > header_model["BMAJ"]:
                    print("WARNING: The resolution is worse than the beam size.\nSo the convolution will not do anything usefull!")
                    print("CDELT1 = %f" %header_model["CDELT1"])
                    print("CDELT2 = %f" %header_model["CDELT2"])
                    print("BMAJ = %f" %header_model["BMAJ"])

                ## Change the beamsize to a really small one
                header_model_copy = header_model.copy()
                header_model_copy["BMAJ"] = 0
                header_model_copy["BMIN"] = 0

                ## Create a fits with smaller beam, save it to a file, read it out and delete the file afterwards cause f**k SpectralCube
                full_cube_copy = SpectralCube(data=full_cube._data*full_cube.unit, wcs=wcs, header=header_model_copy)
                full_cube_copy.write(MyParameters['MapFitPath']+"temp.fits", overwrite=True)
                full_cube_copy = SpectralCube.read(MyParameters['MapFitPath']+"temp.fits")
                cmdStringRMTempFits = "%s %s" %(rmdi, MyParameters['MapFitPath']+"temp.fits")
                os.system(cmdStringRMTempFits)

                ## Read the beam from header
                SC_beam = radio_beam.Beam.from_fits_header(full_cube.header)
            
                ## Apply convolve function
                SC_convolve = full_cube_copy.convolve_to(SC_beam)

                ## Reset beam
                SC_convolve.with_beam(SC_beam)

                ## Save the cube
                convolve_name = '%s__model__convolve.out.fits' %(MyParameters["cube_file"][:-5])
                SC_convolve.write(MyParameters['MapFitPath']+convolve_name, overwrite=True)

                ## Update the result
                full_cube = SC_convolve

                ## Debug:
                if MyParameters["Debug"] == True:
                    print("Use a real beam == %s" %("BMAJ" in header_model and "BMIN" in header_model and "BPA" in header_model))
                    print("SC beam:\n", SC_beam)

            ## Creade cloud background
            if MyParameters['Cloud'] == True:
            
                ## Read cloud and model data
                data_cloud = fits.getdata(MyParameters['CloudPath']+MyParameters['name_cloud'], ext=0)
                header_cloud = fits.getheader(MyParameters['CloudPath']+MyParameters['name_cloud'], ext=0)

                ## Flip cloud cube along each axis by chance
                if numpy.random.randint(2) == 1:
                    data_cloud = numpy.flip(data_cloud, 0)

                if numpy.random.randint(2) == 1:
                    data_cloud = numpy.flip(data_cloud, 1)

                if numpy.random.randint(2) == 1:
                    data_cloud = numpy.flip(data_cloud, 2)

                ## Get the shapes of the cloud and full cube
                csv, csx, csy = numpy.shape(data_cloud)
                msv, msx, msy = numpy.shape(full_cube._data)

                ## Calculate the pixel with v=0
                cpv0 = header_cloud["CRPIX3"] - header_cloud["CRVAL3"]/header_cloud["CDELT3"]
                mpv0 = header_model["CRPIX3"] - header_model["CRVAL3"]/header_model["CDELT3"]

                ## Matching the outer cloud layer with the maximal cloud velocity which is random distributed
                v_cloud_max = numpy.random.uniform(low=MyParameters['MinCloudV'], high=MyParameters['MaxCloudV'])

                ## Empty grid for the cloud
                x_cloud = numpy.linspace(0,csx-1,csx,dtype=int)
                y_cloud = numpy.linspace(0,csy-1,csy,dtype=int)
                v_cloud = numpy.linspace(-v_cloud_max,v_cloud_max,csv)

                ## Interpolating the cloud
                interp = RegularGridInterpolator((v_cloud, x_cloud, y_cloud), data_cloud, method="linear", bounds_error=False, fill_value=0)

                ## Randomly variate the cloud stretch in x direction
                min_x_scale = min(msx/4, csx)
                max_x_scale = min(msx*4, csx)
                x_length = numpy.random.uniform(low=min_x_scale, high=max_x_scale)
                x_shift = numpy.random.uniform(low=0, high=csx-x_length)

                model_x_pixel = numpy.linspace(0,x_length,msx) + x_shift

                ## Randomly variate the cloud stretch in y direction
                min_y_scale = min(msy/4, csy)
                max_y_scale = min(msy*4, csy)
                y_length = numpy.random.uniform(low=min_y_scale, high=max_y_scale)
                y_shift = numpy.random.uniform(low=0, high=csy-y_length)

                model_y_pixel = numpy.linspace(0,y_length,msy) + y_shift

                ## Span the v space
                v_model_min = - header_model["CRPIX3"]*header_model["CDELT3"] + header_model["CRVAL3"]
                v_model_max = - header_model["CRPIX3"]*header_model["CDELT3"] + header_model["CRVAL3"] + header_model["CDELT3"] * msv

                model_v_pixel = numpy.linspace(v_model_min,v_model_max,msv)
                
                ## Span the cloud space
                list_of_voxels = numpy.array(list(itertools.product(model_v_pixel, model_x_pixel, model_y_pixel)))

                ## Interpolate the cloud
                cloud_background = interp(list_of_voxels).reshape(numpy.shape(full_cube)) #  
    
                ## Randomize the intensity gaussianwise and ensure it's larger than 1 K
                cloud_intensity = 0

                while cloud_intensity <= 1:
                    cloud_intensity = numpy.random.normal(loc=MyParameters['CloudLoc'],
                                                          scale=MyParameters['CloudScale'], size=None)

                cloud_background = SpectralCube(data=cloud_background*cloud_intensity / cloud_background.max() * full_cube.unit,
                                                wcs=wcs, header=header_model)

                ## Save the cloud background
                cloud_name = '%s__model__cloud.out.fits' %(MyParameters["cube_file"][:-5])
                cloud_background.write(MyParameters['MapFitPath']+cloud_name, overwrite=True)
            
                ## Add the cloud to the full cube
                full_cube += cloud_background
    
                ## Debug
                if MyParameters["Debug"] == True:
                    print("Cloud/model v0 pixel = %i/%i" %(cpv0, mpv0))
                    print("x variation:", min_x_scale, max_x_scale, x_length, x_shift)
                    print("y variation:", min_y_scale, max_y_scale, y_length, y_shift)
                    print("v variation:", 3, 10, v_cloud_max, v_model_min, v_model_max)
                    print("Compare intensities:", full_cube._data.max(), cloud_intensity, data_cloud.max())


            ## Create white noise
            if MyParameters['WhiteNoise'] == True:

                ## Get the location and std of the noise
                noise_loc = MyParameters["WNLoc"]
                noise_scale = MyParameters["WNScale"]

                ## Generate random, gaussian distributed white noise
                white_noise = numpy.random.normal(loc=noise_loc, scale=noise_scale,
                                                                    size=numpy.shape(full_cube._data))

                ## Create a SpectralCube out of it
                white_noise = SpectralCube(data= white_noise * full_cube.unit, 
                                           wcs=wcs, header=header_model)

                ## Save the noise cube
                noise_name = '%s__model__noise.out.fits' %(MyParameters["cube_file"][:-5])
                white_noise.write(MyParameters['MapFitPath']+noise_name, overwrite=True)

                ## Add noise to the full cube
                full_cube += white_noise

                ## Debug
                if MyParameters["Debug"] == True:
                    print("Noise location = %.3e" %(noise_loc))
                    print("Noise scale = %.3e" %(noise_scale))

            else:
                ## If there is no background noise, set the noise_loc and noise_scale to small values
                noise_loc = 1e-5
                noise_scale = 1e-5



            ## Add flags to the header on how it was modified
            if MyParameters['SmoothGaussian'] == True:
                header_model["GAFLAG"] = (True, "Applied a gaussian filter")

            if MyParameters['Cloud'] == True:
                header_model["CLFLAG"] = (True, "Added a simulated cloud")

            if MyParameters['WhiteNoise'] == True:
                header_model["WNFLAG"] = (True, "Added white noise")

            ## Save the modified cube
            try:
                full_name = 'MapFit_Fit_%s_%s__model.out.fits' %(MyParameters['Molecule'].replace(";", ","), MyParameters['NameDir'][:-1])
                full_cube.write(MyParameters['MapFitPath']+full_name, overwrite=True)
            except:
                print("Using the save name %s failed... Using a shorter one." %(full_name))
                full_name = 'test.fits'
                full_cube.write(MyParameters['MapFitPath']+full_name, overwrite=True)

            ### Generate the mask
            ## Read in the gaussian convoluted cube, if it wasn't created, use the MapFit result instead
            if MyParameters["SmoothGaussian"]:
                original_name = '%s__model__convolve.out.fits' %(MyParameters["cube_file"][:-5])
                original_cube = SpectralCube.read(MyParameters['MapFitPath']+original_name)

            else:
                original_cube = SpectralCube.read('%s%s__model.out.fits' %(MyParameters['ReadMapFit'], MyParameters["cube_file"][:-5]))
    

            ## Creating a mask above 3 sigma threshold, 1 --> Outflow
            try:
                full_cube_mask = SpectralCube(data=numpy.ma.masked_greater(original_cube._data, noise_loc+3*noise_scale).mask * 1.,
                                                wcs=wcs, header=header_model)
            except:
                print("Warning! All outflow values are below the noise level. There will be no outflow mask.")
                full_cube_mask = SpectralCube(data=numpy.ones_like(original_cube._data),
                                                wcs=wcs, header=header_model)

            ## Adjust the mask header
            full_cube_mask.meta['BTYPE'] = "Mask"

            ## Save the mask
            try:
                mask_name = 'MapFit_Fit_%s_%s__mask.out.fits' %(MyParameters['Molecule'].replace(";", ","), MyParameters['NameDir'][:-1])
                full_cube_mask.write(MyParameters['MapFitPath']+mask_name, overwrite=True)
            except:
                print("Using the name %s fail. Using a shorter one." %(mask_name))
                mask_name = 'mask.fits'
                full_cube_mask.write(MyParameters['MapFitPath']+mask_name, overwrite=True)
        
            ## Update database
            if MyParameters["UpdateDataBase"] == True and MyParameters["CalculateModel"] == True and MyParameters["MapFit"] == True:

                ## Read the xml file
                xmlTree = ET.parse(MyParameters['DatePath']+MyParameters['xml_file'])
                FreqRangeroot = xmlTree.getroot().find("file").find("FrequencyRange")

                ## Calculate the frequency range
                FreqMin = float(FreqRangeroot.find("MinExpRange").text)
                FreqMax = float(FreqRangeroot.find("MaxExpRange").text)
                freqrange = FreqMax - FreqMin

                ## Create header
                header_Data = ["Model name", "Theta [deg]", "Phi [deg]", "v [km/s]", "n [cm-3]", "T [K]", "alpha", "dmin [au]", "sigmaof", "ThetaMax [deg]",
                               "epsilon", "gamma", "M1", "beta", "r0 [au]", "z0 [au]", "RestFreq [MHz]", "Frequency Range [MHz]", "Platform", "Date created",
                               "Parent dir", "Cube name", "Mask name", "label"]
            
                ## Create data line
                Outflow_Data = [MyParameters['modelname'], MyParameters['Theta'], MyParameters['Phi'], MyParameters['vmax'], 
                                MyParameters['nmin'], MyParameters['Tmin'], MyParameters['alpha'], MyParameters['dmin'], MyParameters['sigmaof'],
                                MyParameters['ThetaMax'], MyParameters['epsilon'], MyParameters['gamma'], MyParameters['M1'], MyParameters['beta'],
                                MyParameters['r0'], MyParameters['z0'], MyParameters["RestFreq"], freqrange, platform.node(), currenttime, MyParameters['MapFitPath'], 
                                full_name, mask_name, MyParameters["Label"]]


                ## If the database does not exist, create a new one to write to and ad the header
                if not os.path.exists(MyParameters['DatabasePath'] + MyParameters['database_file']):
                    database_pd = pandas.DataFrame(data=[Outflow_Data], index=[currenttime], columns=header_Data)
                    database_pd.to_csv(MyParameters['DatabasePath'] + MyParameters['database_file'], mode="w", index=True, header=True)


                ## If it exists just open it to append the data
                else:
                    database_pd = pandas.DataFrame(data=[Outflow_Data], index=[currenttime], columns=header_Data)
                    database_pd.to_csv(MyParameters['DatabasePath'] + MyParameters['database_file'], mode="a", index=True, header=False)

                print("The data base is updated!\n") 

            ## Create some plots
            if MyParameters["Plot"] == True:

                ## Generate Moment maps
                ## Generate moment 0 map of the MapFit result
                cube_data = SpectralCube.read('%s%s__model.out.fits' %(MyParameters['ReadMapFit'], MyParameters["cube_file"][:-5]))
                moment_map = cube_data.moment(order=0)
                mm_header = moment_map.header
                mm_title = "MapFit Moment 0 Map"

                ## Empty arrays/list to buffer the moment maps, their units, and title for later usage
                mom_map_buffer = numpy.empty([6, cube_data.header["NAXIS2"], cube_data.header["NAXIS1"]])
                mom_map_units = list()
                mom_map_title = list()

                ## Buffer the arrays and unit
                mom_map_buffer[0] = moment_map._data
                mom_map_units.append(moment_map.unit)
                mom_map_title.append(mm_title)

                ## Generate moment 0 map of the convolved cube if existing
                if MyParameters["SmoothGaussian"] == True:
                    cube_name = '%s__model__convolve.out.fits' %(MyParameters["cube_file"][:-5])
                    cube_data = SpectralCube.read(MyParameters['MapFitPath']+cube_name)
                    moment_map = cube_data.moment(order=0)
                    mm_header = moment_map.header
                    mm_title = "Convolution Moment 0 Map"

                    ## Buffer the arrays and unit
                    mom_map_buffer[1] = moment_map._data
                    mom_map_units.append(moment_map.unit)
                    mom_map_title.append(mm_title)

                ## Else generate an empty image
                else:
                    ## Buffer the arrays and unit
                    mom_map_buffer[1] = moment_map._data * numpy.nan
                    mom_map_units.append("")
                    mom_map_title.append("Convolution Moment 0 Map")                


                ## Generate moment 0 map of the full result
                cube_name = 'MapFit_Fit_%s_%s__mask.out.fits' %(MyParameters['Molecule'].replace(";", ","), MyParameters['NameDir'][:-1])
                cube_data = SpectralCube.read(MyParameters['MapFitPath']+cube_name)
                mm_title = "2D Mask Projection"

                ## Buffer the arrays and unit
                mom_map_buffer[2] = numpy.ma.masked_greater(cube_data._data.sum(0), 0).mask*1.
                mom_map_units.append("")
                mom_map_title.append(mm_title)  


                ## Read in the full result
                cube_name = 'MapFit_Fit_%s_%s__model.out.fits' %(MyParameters['Molecule'].replace(";", ","), MyParameters['NameDir'][:-1])
                cube_data = SpectralCube.read(MyParameters['MapFitPath']+cube_name)

                ## Generate moment 0 map of the full result
                moment_map = cube_data.moment(order=0)
                mm_header = moment_map.header
                mm_title = "Full Cube Moment 0 Map"

                ## Buffer the arrays and unit
                mom_map_buffer[3] = moment_map._data
                mom_map_units.append(moment_map.unit)
                mom_map_title.append(mm_title)    


                ## Generate moment 1 map of the full result
                moment_map = cube_data.moment(order=1)
                mm_header = moment_map.header
                mm_title = "Full Cube Moment 1 Map"

                ## Buffer the arrays and unit
                mom_map_buffer[4] = moment_map._data
                mom_map_units.append(moment_map.unit)
                mom_map_title.append(mm_title)        


                ## Generate moment 2 map of the full result
                moment_map = cube_data.moment(order=2)
                mm_header = moment_map.header
                mm_title = "Full Cube Moment 2 Map"

                ## Buffer the arrays and unit
                mom_map_buffer[5] = moment_map._data
                mom_map_units.append(moment_map.unit)
                mom_map_title.append(mm_title)

                ## Plot single moment maps
                ## Use header to generate WCS Axis
                wcs = WCS(mm_header)

                for i in range(6):

                    ## Include wcs the coordinate grid
                    fig = matplotlib.pyplot.figure()
                    gs = fig.add_gridspec(1, 1, wspace=0.5)
                    axs = gs.subplots(subplot_kw={'projection': wcs, 'slices': ("x", "y")})

                    ## Plot the moment map
                    if i in [0,1,3]:
                        img = axs.imshow(mom_map_buffer[i], origin = "lower", norm="log")
                    else:
                        img = axs.imshow(mom_map_buffer[i], origin = "lower", norm="linear")

                    ## Add the grid
                    axs.grid(color='gray', ls='solid', lw=0.1)
                
                    ## Add labels/title
                    axs.set_xlabel(mm_header["CTYPE1"])
                    axs.set_ylabel(mm_header["CTYPE2"])
                    axs.set_title(mom_map_title[i])
                
                    ## Add colorbar
                    im_ratio = mom_map_buffer[i].shape[0]/mom_map_buffer[i].shape[1]
                    matplotlib.pyplot.colorbar(img, label=mom_map_units[i], fraction=0.047*im_ratio)

                    ## Calculate and plot the scale bar
                    scva, scte = scale_bar(mm_header["NAXIS1"], MyParameters['DistanceToSource'], mm_header["CDELT1"])
                    add_scalebar(axs, scva, label=scte, color="white")

                    ## Save the figure
                    matplotlib.pyplot.savefig(MyParameters['PlotPath'] + '%s.pdf' %(mom_map_title[i].replace(" ", "_")),
                                                dpi='figure', format="pdf", metadata=None,
                                                bbox_inches="tight", pad_inches=0.1,
                                                facecolor='auto', edgecolor='auto',
                                                backend=None
                                                )
                    matplotlib.pyplot.close()

                ## Create one picture containing all moment maps
                ## Init the figure
                fig = matplotlib.pyplot.figure(figsize=(24, 18))
                gs = fig.add_gridspec(2, 3, wspace=0.5)
                axs = gs.subplots(subplot_kw={'projection': wcs, 'slices': ("x", "y")})

                for i in range(6):

                    ## Set the current axis
                    ax = axs[int(i/3), i%3]

                    ## Plot the data
                    if i in [0,1,3]:
                        img = ax.imshow(mom_map_buffer[i], origin = "lower", norm="log")
                    else:
                        img = ax.imshow(mom_map_buffer[i], origin = "lower", norm="linear")
                    ## Add the grid
                    ax.grid(color='gray', ls='solid', lw=0.1)

                    ## Add labels/title
                    ax.set_xlabel(mm_header["CTYPE1"])
                    ax.set_ylabel(mm_header["CTYPE2"])
                    ax.set_title(mom_map_title[i])
                
                    ## Add colorbar
                    im_ratio = mom_map_buffer[i].shape[0]/mom_map_buffer[i].shape[1]
                    fig.colorbar(img, ax=ax, label=mom_map_units[i], fraction=0.047*im_ratio)

                    ## Calculate and plot the scale bar
                    scva, scte = scale_bar(mm_header["NAXIS1"], MyParameters['DistanceToSource'], mm_header["CDELT1"])
                    add_scalebar(ax, scva, label=scte, color="white")


                ## Save the file in the plot path
                local_mom_map_plot_name = 'Moment_Maps_Overview.pdf'
                matplotlib.pyplot.savefig(MyParameters['PlotPath'] + local_mom_map_plot_name,
                                            dpi='figure', format="pdf", metadata=None,
                                            bbox_inches="tight", pad_inches=0.1,
                                            facecolor='auto', edgecolor='auto',
                                            backend=None
                                            )

                ## Save the file in the thumbspage if requested
                if MyParameters["UploadPlot"] == True:

                    print("Update Thumpspage")
                    ## create a thubspage directory if it's not existing
                    if not os.path.isdir(MyParameters['LocalThumbsPagePath']):
                        Path(MyParameters['LocalThumbsPagePath']).mkdir(parents=True)

                    ## Save the picture there as a png
                    thumbspage_mom_map_plot_name = 'Moment_Maps_%s_%s.png' %(MyParameters['Molecule'].replace(";", ","), MyParameters['NameDir'][:-1])
                    matplotlib.pyplot.savefig(MyParameters['LocalThumbsPagePath'] + thumbspage_mom_map_plot_name,
                                                dpi='figure', format="png", metadata=None,
                                                bbox_inches="tight", pad_inches=0.1,
                                                facecolor='auto', edgecolor='auto',
                                                backend=None
                                                )

                ## Close the plot again
                matplotlib.pyplot.close()

            tev = time.time()

            ## In case of a production run, remove all files but model and mask
            if MyParameters["ProductionRun"] == True:

                ## Go thruough all directories 
                for root, dirnames, filenames in os.walk(MyParameters["DatePath"]):

                    ## Remove all but the MapFit directory
                    for name in dirnames:
                        if os.path.join(root, name) != MyParameters["MapFitPath"][:-1]:
                            cmdStringPRRMF = "%s %s" %(rmdi, os.path.join(root, name))
                            os.system(cmdStringPRRMF)

                            ## Debug
                            if MyParameters["Debug"] == True:
                                print(cmdStringPRRMF)


                ## Go thruough all remaining files
                for root, dirnames, filenames in os.walk(MyParameters["DatePath"]):
                
                    ## Remove all files but the mask and model cubes in the MapFit directory
                    for name in filenames:
                        if not "MapFit_Fit" in name:
                            cmdStringPRRMF = "%s %s" %(rmdi, os.path.join(root, name))
                            os.system(cmdStringPRRMF)
         
                            ## Debug
                            if MyParameters["Debug"] == True:
                                print(cmdStringPRRMF)

            print("Time to variate and save the data was %s" %(str(datetime.timedelta(seconds=numpy.ceil(tev-tsv)))))

            print("All data are saved at %s\n" %(MyParameters['DatePath']))


        ## Update thumbs page and upload the latest version to hera/another server
        if MyParameters["Plot"] == True and MyParameters["UploadPlot"] == True:

            if system != "Linux":
                print("Warning: The server synchronization is (so far) only possible for a Linux system.\nThe synchronization is NOT executed!")

            else:
                ## Update the thumbs page

                ## Create the input file
                thumpspage_input = "%s\ny\n2\n(1000,1000)\ny"  %(MyParameters['LocalThumbsPagePath'])     # Compare the standart input
                thumpspageFileName = "%sinputs.txt" %(MyParameters['LocalThumbsPagePath'])
                thumpspageFile = open(thumpspageFileName, 'w')
                thumpspageFile.write(thumpspage_input)
                thumpspageFile.close()

                ## Block thumspage output if requested
                if MyParameters["printThumbsPage"] == False:
                    blockPrint()             

                ## Create/update the thumbs page
                cmdStringCreateTP = "python3 %s%s < %s" %(MyParameters['ThumbsPageScriptPath'], MyParameters['thumbspage_script'], thumpspageFileName)
                try:
                    os.system(cmdStringCreateTP)
                except:
                    print("Failed to execute:\n%s\n\n" %(cmdStringCreateTP))

                ## Enable print again
                if MyParameters["printThumbsPage"] == False:
                    enablePrint()       

                ## rsync string
                cmdStringRSYNC = "rsync -rut %s %s" %(MyParameters['LocalThumbsPagePath'], MyParameters['UploadPath'])
                
                ## Check if the upload path exists
                if not os.path.isdir(MyParameters['UploadPath']):
                    ## Try to sync the thumbspage
                    try:
                        if MyParameters['UploadPath'][0] == "/" or MyParameters['UploadPath'][:2] == "./" or MyParameters['UploadPath'][1:3] == ":/":
                            Path(MyParameters['UploadPath']).mkdir(parents=True)

                        else:
                            server_name = MyParameters['UploadPath'].split("/")[0][:-1]
                            path_name = "/".join(MyParameters['UploadPath'].split("/")[1:])
                            cmdStringMKDS = "ssh %s mkdir -p %s" %(server_name, path_name)

                            print(cmdStringMKDS)

                            ## Sync hera by executing rsync string
                            try:
                                os.system(cmdStringRSYNC)
                            except:
                                print("Failed to execute:\n%s\n\n" %(cmdStringRSYNC))

                    except Exception as e:
                        exception_type, exception_object, exception_traceback = sys.exc_info()
                        filename = exception_traceback.tb_frame.f_code.co_filename
                        line_number = exception_traceback.tb_lineno

                        print("\nThere was an issue while syncing the directories. Failed to execute\n\n%s\n" %(cmdStringRSYNC))
                        print("Exception type:\t%s" %(exception_type))
                        print("File name:\t%s" %(filename))
                        print("Line number:\t%s" %(line_number))
                        print("The error itself:\n%s\n\n" %(e))


                ## Sync hera by executing rsync string
                try:
                    os.system(cmdStringRSYNC)
                except:
                    print("Failed to execute:\n%s\n\n" %(cmdStringRSYNC))

    print("All done.")
    tet = time.time()
    print("The total run time was %s" %(str(datetime.timedelta(seconds=numpy.ceil(tet-tst)))))

 
## The end

if __name__ == '__main__':

    MasterFile = "OutflowMaster.xml"
    Main(MasterFile)