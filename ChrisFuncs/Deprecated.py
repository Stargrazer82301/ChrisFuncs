# Import smorgasbord
import pdb
import pdb
"""from IPython import get_ipython
get_ipython().run_line_magic('pdb','on')"""
import sys
import os
#sys.path.append( os.path.split( os.path.realpath(__file__) )[:-1][0] )
#sys.path.append( os.path.split( os.path.split( os.path.realpath(__file__) )[:-1][0] )[:-1][0] )
#sys.path.insert(0, '../')
import numpy as np
import scipy.stats
import scipy.ndimage
import scipy.ndimage.measurements
import scipy.spatial
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.patches
import astropy
astropy.log.setLevel('ERROR')
import astropy.io.fits
import astropy.wcs
import astropy.convolution
import astropy.coordinates
import astropy.units
import astroquery.irsa_dust
import shutil
import wget
import glob
import time
import re
import copy

# A python 2/3 compatability hack for stirng type handling
try:
  basestring
except NameError:
  basestring = str





# Function to convert GALEX data units into AB pogson magnitudes
# Args: Value to be converted (data units), GALEX band (1 or FUV for FUV, 2 or NUV for NUV)
# Returns: AB pogson magnitudes (mags; duh)
def GALEXCountsToMags(GALEX, w):
    if w==0 or w=='FUV':
        mag = 18.82 - ( 2.5*np.log10(GALEX) )
    if w==1 or w=='NUV':
        mag = 20.08 - ( 2.5*np.log10(GALEX) )
    return mag



# Function to convert AB pogson magnitudes into GALEX data counts
# Args: Value to be converted (mags), GALEX band (1 for FUV, 2 for NUV)
# Returns: Galex  counts (data units)
def GALEXMagsToCounts(mag,w):
    if w==0:
        GALEX = 10.0**( (18.82-mag) / 2.5 )
    if w==1:
        GALEX = 10.0**( (20.08-mag) / 2.5 )
    return GALEX



# Function to perform gaussian fit to data, and perform plot of data and fit
# Args: Array of data to be fit
# Returns: Mean of fit, standard deviation of fit (,plot of data and fit)
def GaussFitPlot(data, n_bins=50, show=True):
    def QuickGauss(x, *p):
        A, mu, sigma = p
        return A * np.exp( (-((x-mu)**2.0)) / (2.0*sigma**2.0) )
    hist_tots, hist_bins = np.histogram(data, bins=n_bins)
    hist_bins = HistBinMerge(hist_bins)
    hist_guesses = [1.0, np.mean(data), np.std(data)]
    hist_fit = scipy.optimize.curve_fit(QuickGauss, hist_bins, hist_tots, p0=hist_guesses)
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.scatter(hist_bins, hist_tots, marker='+', c='black')
    gauss_x = np.linspace(hist_bins.min(), hist_bins.max(), num=10000)
    gauss_y = np.zeros([10000])
    for g in range(0,10000):
        gauss_y[g] = QuickGauss(gauss_x[g], hist_fit[0][0], hist_fit[0][1], hist_fit[0][2])
    ax.plot(gauss_x, gauss_y, c='red')
    if show==True:
        fig.canvas.draw()
    return abs(hist_fit[0][1]), abs(hist_fit[0][2]), fig, ax



# Function to plot histogram of input data
# Args: Array of data to be fit
# Returns: (plot of data)
def HistPlot(data, n_bins=25, show=True):
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.hist(data, bins=n_bins, facecolor='#0174DF')
    if show==True:
        fig.canvas.draw()
    return fig, ax