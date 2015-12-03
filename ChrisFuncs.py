# Import smorgasbord
import sys
import os
import pdb
current_module = sys.modules[__name__]
from numpy import *
import numpy as np
from numpy.linalg import eig, inv
import scipy.stats
import scipy.ndimage
import scipy.ndimage.measurements
import scipy.spatial
from cStringIO import StringIO
from subprocess import call
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.patches
import astropy.io.fits
import astropy.wcs
import astropy.convolution
#import cutout as ag_cutout
import FITS_tools
import random
import pickle
import pip
import aplpy
import ChrisFuncs



# Function to sum all elements in an ellipse centred on the middle of a given array
# Input: Array, semi-major axis (pix), axial ratio, position angle (deg), i & j coords of centre of ellipse
# Returns: Numpy array containing the sum of the pixel values in the ellipse, total number of pixels counted, and an array containing the pixel values
def EllipseSum(array, rad, axial_ratio, angle, i_centre, j_centre):

    # Define semi-major & semi-minor axes, then convert input angle to radians
    semi_maj = float(rad)
    semi_min = float(rad) / float(axial_ratio)
    angle = np.radians(float(angle))

    # Create meshgrids with which to access i & j coordinates for ellipse calculations
    i_linespace = np.linspace(0, array.shape[0]-1, array.shape[0])
    j_linespace = np.linspace(0, array.shape[1]-1, array.shape[1])
    i_grid, j_grid = np.meshgrid(i_linespace, j_linespace, indexing='ij')

    # Use meshgrids to create array identifying which coordinates lie within ellipse
    i_trans = -(j_grid-float(j_centre))*np.sin(angle) + (i_grid-float(i_centre))*np.cos(angle)
    j_trans = (j_grid-float(j_centre))*np.cos(angle) + (i_grid-float(i_centre))*np.sin(angle)
    ellipse_check = (j_trans**2 / semi_maj**2) + (i_trans**2 / semi_min**2 )

    # Calculate flux & pixels in aperture, and store pixel values
    ellipse_tot = sum( array[ np.where( (ellipse_check<=1) & (np.isnan(array)==False) ) ] )
    ellipse_count = np.where( (ellipse_check<=1) & (np.isnan(array)==False) )[0].shape[0]
    ellipse_pix = array[ np.where( (ellipse_check<=1) & (np.isnan(array)==False) ) ]

    # Return results
    return [ellipse_tot, ellipse_count, ellipse_pix]



# Function to sum all elements in an annulus centred upon the middle of the given array
# Input: Array, semi-major axis of inside edge of annulus (pix), width of annulus (pix), axial ratio, position angle (deg), i & j coords of centre of ellipse
# Returns: Numpy array containing the sum of the pixel values in the annulus, the total number of pixels counted, and an array containing the pixel values
def AnnulusSum(array, rad_inner, width, axial_ratio, angle, i_centre, j_centre):

    # Define semi-major & semi-minor axes, then convert input angle to radians
    semi_maj_inner = float(rad_inner)
    semi_min_inner = float(semi_maj_inner) / float(axial_ratio)
    semi_maj_outer = float(rad_inner) + float(width)
    semi_min_outer  = float(semi_maj_outer) / float(axial_ratio)
    angle = np.radians(float(angle))

    # Create meshgrids with which to access i & j coordinates for ellipse calculations
    i_linespace = np.linspace(0, array.shape[0]-1, array.shape[0])
    j_linespace = np.linspace(0, array.shape[1]-1, array.shape[1])
    i_grid, j_grid = np.meshgrid(i_linespace, j_linespace, indexing='ij')

    # Use meshgrids to create array identifying which coordinates lie within inner ellipse
    i_trans = -(j_grid-float(j_centre))*np.sin(angle) + (i_grid-float(i_centre))*np.cos(angle)
    j_trans = (j_grid-float(j_centre))*np.cos(angle) + (i_grid-float(i_centre))*np.sin(angle)
    ellipse_check_inner = (j_trans**2 / semi_maj_inner**2) + (i_trans**2 / semi_min_inner**2 )

    # Use meshgrids to create array identifying which coordinates lie within outer ellipse
    i_trans = -(j_grid-float(j_centre))*np.sin(angle) + (i_grid-float(i_centre))*np.cos(angle)
    j_trans = (j_grid-float(j_centre))*np.cos(angle) + (i_grid-float(i_centre))*np.sin(angle)
    ellipse_check_outer = (j_trans**2 / semi_maj_outer**2) + (i_trans**2 / semi_min_outer**2 )

    # Calculate flux & pixels in aperture, and store pixel values
    annulus_tot = sum( array[ np.where( (ellipse_check_outer<=1) & (ellipse_check_inner>1) & (np.isnan(array)==False) ) ] )
    annulus_count = np.where( (ellipse_check_outer<=1) & (ellipse_check_inner>1) & (np.isnan(array)==False) )[0].shape[0]
    annulus_pix = array[ np.where( (ellipse_check_outer<=1) & (ellipse_check_inner>1) & (np.isnan(array)==False) ) ]

    # Return results
    return [annulus_tot, annulus_count, annulus_pix]



# Function to make annular photometry faster by pre-preparing arrays of transposed coords that are to be repeatedly used
# Input: Array, semi-major axis of inside edge of annulus (pix), width of annulus (pix), axial ratio, position angle (deg), i & j coords of centre of ellipse
# Returns: List containing i & j transposed coords
def AnnulusQuickPrepare(array, angle, i_centre, j_centre):

    # Convert input angle to radians
    angle = np.radians(float(angle))

    # Create meshgrids with which to access i & j coordinates for ellipse calculations
    i_linespace = np.linspace(0, array.shape[0]-1, array.shape[0])
    j_linespace = np.linspace(0, array.shape[1]-1, array.shape[1])
    i_grid, j_grid = np.meshgrid(i_linespace, j_linespace, indexing='ij')

    # Use meshgrids to create array identifying which coordinates lie within inner ellipse
    i_trans = -(j_grid-float(j_centre))*np.sin(angle) + (i_grid-float(i_centre))*np.cos(angle)
    j_trans = (j_grid-float(j_centre))*np.cos(angle) + (i_grid-float(i_centre))*np.sin(angle)

    # Return results
    return [i_trans, j_trans]



# Function to sum all elements in an annulus centred upon the middle of the given array, usingpre-prepared transposed coord arrays
# Input: Array, semi-major axis of inside edge of annulus (pix), width of annulus (pix), axial ratio, position angle (deg), i & j coords of centre of ellipse, i & j transposed coord arrays
# Returns: Numpy array containing the sum of the pixel values in the annulus, the total number of pixels counted, and an array containing the pixel values
def AnnulusQuickSum(array, rad_inner, width, axial_ratio, angle, i_centre, j_centre, i_trans, j_trans):

    # Define semi-major & semi-minor axes, then convert input angle to radians
    semi_maj_inner = float(rad_inner)
    semi_min_inner = float(semi_maj_inner) / float(axial_ratio)
    semi_maj_outer = float(rad_inner) + float(width)
    semi_min_outer  = float(semi_maj_outer) / float(axial_ratio)
    angle = np.radians(float(angle))

    # Use meshgrids to create array identifying which coordinates lie within inner & outer ellipses
    ellipse_check_inner = (j_trans**2 / semi_maj_inner**2) + (i_trans**2 / semi_min_inner**2 )

    # Use meshgrids to create array identifying which coordinates lie within outer ellipse
    ellipse_check_outer = (j_trans**2 / semi_maj_outer**2) + (i_trans**2 / semi_min_outer**2 )

    # Calculate flux & pixels in aperture, and store pixel values
    annulus_tot = sum( array[ np.where( (ellipse_check_outer<=1) & (ellipse_check_inner>1) & (np.isnan(array)==False) ) ] )
    annulus_count = np.where( (ellipse_check_outer<=1) & (ellipse_check_inner>1) & (np.isnan(array)==False) )[0].shape[0]
    annulus_pix = array[ np.where( (ellipse_check_outer<=1) & (ellipse_check_inner>1) & (np.isnan(array)==False) ) ]

    # Return results
    return [annulus_tot, annulus_count, annulus_pix]



# Function to sum all elements in an annulus centred upon the middle of the given array, using pre-prepared transposed coord arrays
# Input: Array, semi-major axis of ellipse (pix), position angle (deg), i & j coords of centre of ellipse, i & j transposed coord arrays
# Returns: Numpy array containing the sum of the pixel values in the ellipse, the total number of pixels counted, and an array containing the pixel values
def EllipseQuickSum(array, rad, axial_ratio, angle, i_centre, j_centre, i_trans, j_trans):

    # Define semi-major & semi-minor axes, then convert input angle to radians
    semi_maj = float(rad)
    semi_min = float(semi_maj) / float(axial_ratio)
    angle = np.radians(float(angle))

    # Use meshgrids to create array identifying which coordinates lie within  ellipses
    ellipse_check = (j_trans**2 / semi_maj**2) + (i_trans**2 / semi_min**2 )

    # Calculate flux & pixels in aperture, and store pixel values
    ellipse_tot = sum( array[ np.where( (ellipse_check<=1) & (np.isnan(array)==False) ) ] )
    ellipse_count = np.where( (ellipse_check<=1) & (np.isnan(array)==False) )[0].shape[0]
    ellipse_pix = array[ np.where( (ellipse_check<=1) & (np.isnan(array)==False) ) ]

    # Return results
    return [ellipse_tot, ellipse_count, ellipse_pix]



# Function to return a mask identifying all pixels within an ellipse of given parameters
# Input: Array, semi-major axis (pix), axial ratio, position angle (deg), i & j coords of centre of ellipse
# Output: Mask array of same dimensions as input array where pixels that lie within ellipse have value 1
def EllipseMask(array, rad, axial_ratio, angle, i_centre, j_centre):

    # Define semi-major & semi-minor axes, then convert input angle to radians
    semi_maj = float(rad)
    semi_min = float(rad) / float(axial_ratio)
    angle = np.radians(float(angle))

    # Create meshgrids with which to access i & j coordinates for ellipse calculations
    i_linespace = np.linspace(0, array.shape[0]-1, array.shape[0])
    j_linespace = np.linspace(0, array.shape[1]-1, array.shape[1])
    i_grid, j_grid = np.meshgrid(i_linespace, j_linespace, indexing='ij')

    # Use meshgrids to create array identifying which coordinates lie within ellipse
    i_trans = -(j_grid-float(j_centre))*np.sin(angle) + (i_grid-float(i_centre))*np.cos(angle)
    j_trans = (j_grid-float(j_centre))*np.cos(angle) + (i_grid-float(i_centre))*np.sin(angle)
    ellipse_check = (j_trans**2 / semi_maj**2) + (i_trans**2 / semi_min**2 )

    # Create ellipse mask
    ellipse_mask = np.zeros([array.shape[0], array.shape[1]])
    ellipse_mask[ where( ellipse_check<=1 ) ] = 1.0

    # Return array
    return ellipse_mask



# Function to sum all pixel elements inside a given circle... the old-fashioned way
# Input: Array to be used, i & j coordinates of centre of circle, radius of circle
# Output: Sum of elements within circle, number of pixels within circle
def CircleSum(fits, i_centre, j_centre, r):
    i_centre, j_centre, r = int(i_centre), int(j_centre), int(r)
    ap_sum = 0.0
    ap_pix = 0.0
    ap_values = []
    for i in range(-r, r+1):
        for j in range(-r, r+1):
            if i**2.0 + j**2.0 <= r**2.0:
                try:
                    ap_sum += fits[i_centre+i, j_centre+j]
                    ap_pix += 1.0
                    ap_values.append(fits[i_centre+i, j_centre+j])
                except:
                    continue
    return [ap_sum, ap_pix, ap_values]



# Function to sum all pixel elements inside a given circle... the old-fashioned way
# Input: Array to be used, i & j coordinates of centre of circle, radius of circle
# Output: Sum of elements within circle, number of pixels within circle
def CircleAnnulusSum(fits, i_centre, j_centre, r, width):
    i_centre, j_centre, r, width = int(i_centre), int(j_centre), int(r), int(width)
    ann_sum = 0.0
    ann_pix = 0.0
    ann_values = []
    for i in range(-r, r+1):
        for j in range(-r, r+1):
            if (i**2.0 + j**2.0 > r**2.0) and (i**2.0 + j**2.0 <= (r+width)**2.0):
                try:
                    ann_sum += fits[i_centre+i, j_centre+j]
                    ann_pix += 1.0
                    ann_values.append(fits[i_centre+i, j_centre+j])
                except:
                    continue
    return [ann_sum, ann_pix, ann_values]



# Function to sum all elements in an ellipse centred on the middle of an array that has been resized to allow better pixel sampling
# Input: Array, semi-major axis (pix), axial ratio, position angle (deg), i & j coords of centre of ellipse, upscaling factor
# Returns: Numpy array containing the sum of the pixel values in the ellipse, the total number of pixels counted, and an array containing the pixel values
def EllipseSumUpscale(cutout, rad, axial_ratio, angle, i_centre, j_centre, upscale=1):

    # Resize array to increase pixel sampling, cupdate centre coords, and downscale pixel values accordinly to preserve flux
    cutout_inviolate = np.copy(cutout)
    cutout = np.zeros([cutout_inviolate.shape[0]*upscale, cutout_inviolate.shape[1]*upscale])
    scipy.ndimage.zoom(cutout_inviolate, upscale, output=cutout, order=0)
    cutout *= float(upscale)**-2.0
    i_centre = float(i_centre) * float(upscale)
    j_centre = float(j_centre) * float(upscale)

    # Define semi-major & semi-minor axes, then convert input angle to radians
    semi_maj = float(rad) * float(upscale)
    semi_min = semi_maj / float(axial_ratio)
    angle = np.radians(float(angle))

    # Create meshgrids with which to access i & j coordinates for ellipse calculations
    i_linespace = np.linspace(0, cutout.shape[0]-1, cutout.shape[0])
    j_linespace = np.linspace(0, cutout.shape[1]-1, cutout.shape[1])
    i_grid, j_grid = np.meshgrid(i_linespace, j_linespace, indexing='ij')

    # Use meshgrids to create array identifying which coordinates lie within ellipse
    i_trans = -(j_grid-float(j_centre))*np.sin(angle) + (i_grid-float(i_centre))*np.cos(angle)
    j_trans = (j_grid-float(j_centre))*np.cos(angle) + (i_grid-float(i_centre))*np.sin(angle)
    ellipse_check = (j_trans**2 / semi_maj**2) + (i_trans**2 / semi_min**2 )

    # Calculate flux & pixels in aperture, and store pixel values
    ellipse_tot = sum( cutout[ np.where( (ellipse_check<=1) & (np.isnan(cutout)==False) ) ] )
    ellipse_count = np.where( (ellipse_check<=1) & (np.isnan(cutout)==False) )[0].shape[0]
    ellipse_pix = cutout[ np.where( (ellipse_check<=1) & (np.isnan(cutout)==False) ) ]

    # Scale output values down to what they would've been for original array
    ellipse_count *= float(upscale)**-2.0

    # Return results
    return [ellipse_tot, ellipse_count, ellipse_pix]



# Function to sum all elements in an annulus centred upon the middle of an array that has been resized to allow better pixel sampling
# Input: Array, semi-major axis of inside edge of annulus (pix), width of annulus (pix), axial ratio, position angle (deg), i & j coords of centre of ellipse, upscaling factor
# Returns: Numpy array containing the sum of the pixel values in the annulus, the total number of pixels counted, and an array containing the pixel values
def AnnulusSumUpscale(cutout, rad_inner, width, axial_ratio, angle, i_centre, j_centre, upscale=1):

    # Resize array to increase pixel sampling, update centre coords, and downscale pixel values accordinly to preserve flux
    cutout_inviolate = np.copy(cutout)
    cutout = np.zeros([cutout_inviolate.shape[0]*upscale, cutout_inviolate.shape[1]*upscale])
    scipy.ndimage.zoom(cutout_inviolate, upscale, output=cutout, order=0)
    cutout *= float(upscale)**-2.0
    i_centre = float(i_centre) * float(upscale)
    j_centre = float(j_centre) * float(upscale)

    # Define semi-major & semi-minor axes, then convert input angle to radians
    semi_maj_inner = float(rad_inner) * float(upscale)
    semi_min_inner = semi_maj_inner / float(axial_ratio)
    semi_maj_outer = ( float(rad_inner) * float(upscale) ) + ( float(width) * float(upscale) )
    semi_min_outer  = semi_maj_outer / float(axial_ratio)
    angle = np.radians(float(angle))

    # Create meshgrids with which to access i & j coordinates for ellipse calculations
    i_linespace = np.linspace(0, cutout.shape[0]-1, cutout.shape[0])
    j_linespace = np.linspace(0, cutout.shape[1]-1, cutout.shape[1])
    i_grid, j_grid = np.meshgrid(i_linespace, j_linespace, indexing='ij')

    # Use meshgrids to create cutout identifying which coordinates lie within inner ellipse
    i_trans = -(j_grid-float(j_centre))*np.sin(angle) + (i_grid-float(i_centre))*np.cos(angle)
    j_trans = (j_grid-float(j_centre))*np.cos(angle) + (i_grid-float(i_centre))*np.sin(angle)
    ellipse_check_inner = (j_trans**2 / semi_maj_inner**2) + (i_trans**2 / semi_min_inner**2 )

    # Use meshgrids to create cutout identifying which coordinates lie within outer ellipse
    i_trans = -(j_grid-float(j_centre))*np.sin(angle) + (i_grid-float(i_centre))*np.cos(angle)
    j_trans = (j_grid-float(j_centre))*np.cos(angle) + (i_grid-float(i_centre))*np.sin(angle)
    ellipse_check_outer = (j_trans**2 / semi_maj_outer**2) + (i_trans**2 / semi_min_outer**2 )

    # Calculate flux & pixels in aperture, and store pixel values
    annulus_tot = sum( cutout[ np.where( (ellipse_check_outer<=1) & (ellipse_check_inner>1) & (np.isnan(cutout)==False) ) ] )
    annulus_count = np.where( (ellipse_check_outer<=1) & (ellipse_check_inner>1) & (np.isnan(cutout)==False) )[0].shape[0]
    annulus_pix = cutout[ np.where( (ellipse_check_outer<=1) & (ellipse_check_inner>1) & (np.isnan(cutout)==False) ) ]

    # Scale output values down to what they would've been for original array
    annulus_count *= float(upscale)**-2.0

    # Return results
    return [annulus_tot, annulus_count, annulus_pix]



# Function to perform a sigma clip upon a set of values
# Input: Array of values, convergence tolerance, state if median instead of mean should be used for clip centrepoint, clipping threshold, boolean for whether sigma of zero can be accepted
# Returns: List containing the clipped standard deviation, the average, and the values themselves
def SigmaClip(values, tolerance=0.001, median=False, sigma_thresh=3.0, no_zeros=False):

    # Remove NaNs from input values
    values = np.array(values)
    values = values[ np.where(np.isnan(values)==False) ]
    values_original = np.copy(values)

    # Continue loop until result converges
    diff = 10E10
    while diff>tolerance:

        # Assess current input iteration
        if median == False:
            average = np.mean(values)
        elif median == True:
            average = np.median(values)
        sigma_old = np.std(values)

        # Mask those pixels that lie more than 3 stdev away from mean
        check = np.zeros([len(values)])
        check[ np.where( values>(average+(sigma_thresh*sigma_old)) ) ] = 1
        check[ np.where( values<(average-(sigma_thresh*sigma_old)) ) ] = 1
        values = values[ np.where(check<1) ]

        # Re-measure sigma and test for convergence
        sigma_new = np.std(values)
        diff = abs(sigma_old-sigma_new) / sigma_old

    # Perform final mask
    check = np.zeros([len(values)])
    check[ np.where( values>(average+(sigma_thresh*sigma_old)) ) ] = 1
    check[ np.where( values<(average-(sigma_thresh*sigma_old)) ) ] = 1
    values = values[ np.where(check<1) ]

    # If required, check if calculated sigma is zero
    if no_zeros==True:
        if sigma_new==0.0:
            sigma_new = np.std(values_original)
            if median==False:
                average = np.mean(values)
            elif median==True:
                average = np.median(values)

    # Return results
    return [sigma_new, average, values]



# Function to iteratively calculate SPIRE aperture noise of photometry cutout using randomly-placed (annular-background-subtracted) circular aperture
# Input: Map, radius of aperture (pix), area of aperture (pix), boolean of whether or not to sky-subtract the noise apertures, relative radius of inner edge of annulus, relative width of annulus, angle of source ellipse, axial ratio of source ellipse
# Returns: Aperture standard deviation, and list of mean background values, list of aperture sum values
def CircularApertureStandardDeviationFinder(fits, area, ann=True, ann_inner=1.5, ann_width=1.0, angle=0.0, axial_ratio=1.0, apertures=100):

    # Calculate aperture's radius from given circular area
    rad = np.sqrt(area/np.pi)

    # Define width of border
    if ann==True:
        factor = ann_inner + ann_width
    elif ann==False:
        factor = 1.0

    # Define exclusion zone
    semi_maj = ( area / (axial_ratio*np.pi) )**0.5
    source_i_centre, source_j_centre = int(round(float(fits.shape[0])/2.0)), int(round(float(fits.shape[1])/2.0))
    exclusion_mask = ChrisFuncs.EllipseMask(fits, semi_maj, axial_ratio, angle, source_i_centre, source_j_centre)

    # Definine initial values and initiate loop
    if rad < 2:
        rad = 2

    # Define limits for fixed sigma-clip
    fits_clip = ChrisFuncs.SigmaClip(fits[ np.where(exclusion_mask==0) ], sigma_thresh=5.0, median=True, no_zeros=True)
    clip_upper = fits_clip[1] + (3.0 * fits_clip[0])
    clip_lower = fits_clip[1] - (3.0 * fits_clip[0])

    # Loop over apertures
    ap_sum_array = []
    bg_mean_array = []
    rejects_array = []
    clip_frac_array = []
    old_school_ap_sum_array = []
    a = 0
    b = 0
    sum_ratio_array = []
    while a<apertures*2.5:

        # Generate box coordinates, excluding cutout border, and target source
        ap_border = int((factor*rad)+1)
        neverending = 1
        while neverending > 0:
            b += 1
            neverending = 1
            i_centre = random.randint(ap_border, fits.shape[0]-ap_border)
            j_centre = random.randint(ap_border, fits.shape[1]-ap_border)

            # Do crude check that generated coords do not intersect source
            if ( abs(0.5*fits.shape[0] - i_centre) < rad ) and ( abs(0.5*fits.shape[1] - j_centre) < rad ):
                continue

            # Do sophisticated check that generated coords do not intersect source
            exclusion_cutout = exclusion_mask[ i_centre-(rad+1):i_centre+(rad+1) , j_centre-(rad+1):j_centre+(rad+1) ]
            exclusion_sum = np.sum(exclusion_cutout)
            if exclusion_sum>0:
                continue
            else:
                break

        # Extract slice around aperture; if slice has either dimension 0, then continue
        cutout = fits[ i_centre-(rad*factor+1):i_centre+(rad*factor+1) , j_centre-(rad*factor+1):j_centre+(rad*factor+1) ]
        box_centre = int(round(cutout.shape[0]/2.0))
        if cutout.shape[0]==0 or cutout.shape[1]==0:
            continue

        # If background-subtraction required, measure sky annulus
        if ann==True:
            bg_phot = ChrisFuncs.AnnulusSum(cutout, ann_inner*rad, ann_width*rad, 1.0, 0.0, box_centre, box_centre)
            bg_mean = ChrisFuncs.SigmaClip(np.array(bg_phot[2]))[1]
        elif ann==False:
            bg_mean = 0.0

        # Measure sky aperture, and reject if it contains nothing but NaN pixels
        ap_phot = ChrisFuncs.EllipseSum(cutout, rad, 1.0, 0.0, box_centre, box_centre)
        old_school_ap_sum_array.append(ap_phot[0])
        if ap_phot[2].shape[0] == 0:
            continue
        if np.where( ap_phot[2]==0 )[0].shape[0] == ap_phot[2].shape[0]:
            continue

        # Perform fixed sigma-clip of sky aperture; reject if >20% of pixels are clipped, otherwise scale flux for removed pixels
        rejects_array.append(ap_phot[0])
        ap_clipped = ap_phot[2][ np.where( (ap_phot[2]<clip_upper) & (ap_phot[2]>clip_lower) ) ]
        ap_clipped = ap_clipped[ np.where( ap_clipped<clip_upper ) ]
        ap_clipped = ap_clipped[ np.where( ap_clipped>clip_lower ) ]
        clip_frac = float(ap_clipped.shape[0]) / float(ap_phot[2].shape[0])
        if clip_frac<0.8:
            continue
        ap_sum = np.sum(ap_clipped) / clip_frac
        sum_ratio_array.append(ap_sum/ap_phot[0])
        clip_frac_array.append(clip_frac)

        # Store calculated sum and tick up counter
        a += 1
        ap_sum_array.append(ap_sum)
        if ann==True:
            bg_mean_array.append(bg_mean)
    #pdb.set_trace()

    # Clip out the random apertures with the most extremal clip fractions, to get rid of stray light and so forth
    clip_frac_threshold = np.median(clip_frac_array) + ( 2.0 * np.std(clip_frac_array) )
    ap_sum_array = np.array(ap_sum_array)
    bg_mean_array = np.array(bg_mean_array)
    old_school_ap_sum_array = np.array(old_school_ap_sum_array)
    ap_sum_array = ap_sum_array[ np.where( (clip_frac_array<clip_frac_threshold) & (np.isnan(ap_sum_array)==False) ) ]
    bg_mean_array = bg_mean_array[ np.where( (clip_frac_array<clip_frac_threshold) & (np.isnan(bg_mean_array)==False) ) ]
    max_index = np.min([100, int(ap_sum_array.shape[0])])
    ap_sum_array = ap_sum_array[:max_index]

    # Now take standard deviation of output array to get a noise value as pure as the driven snow
    ap_sigma = np.std(ap_sum_array)
    rejects_array = np.array(rejects_array)
    rejects_array = rejects_array[:max_index]

    # Report calculated standard deviation
    return ap_sigma, bg_mean_array, ap_sum_array, old_school_ap_sum_array



# Function to identify large regions of zeros due to no map coverage, and set them to be NaNs
# Input: Array
# Output: Array with large contiguous zero regions removed
def NoCoverageZerosToNans(cutout):

    # Perform tophat convolution of array, using a kernel with radius 5% the diameter of the array
    kernel_diam = 20
    kernel = astropy.convolution.kernels.Tophat2DKernel(kernel_diam)
    cutout_conv = astropy.convolution.convolve(cutout, kernel)
    cutout[ np.where( cutout_conv==0 ) ] = np.NaN
    return cutout



# Function to find all contiguous pixels that lie above a given flux limit
# Input: Array, radius of guess region (pix), i & j coords of centre of guess region, cutoff value for pixel selection
# Returns: Array of ones and zeros indicating contiguous region
def ContiguousPixels(cutout, rad_initial, i_centre, j_centre, cutoff):

    # Create version of cutout where significant pixels have value 1, insignificant pixels have value 0
    cont_array_binary = np.zeros([(cutout.shape)[0], (cutout.shape)[1]])
    cont_array_binary[where(cutout>=cutoff)[0], where(cutout>=cutoff)[1]] = 1

    # Use SciPy's label function to identify contiguous features in binary map
    cont_structure = array([[1,1,1], [1,1,1], [1,1,1]])
    cont_array = np.zeros([(cutout.shape)[0], (cutout.shape)[1]])
    scipy.ndimage.measurements.label(cont_array_binary, structure=cont_structure, output=cont_array)

    # Identify primary contiguous feature within specified radius of given coordinates
    cont_array_mask = ChrisFuncs.EllipseMask(cont_array, rad_initial, 1.0, 0.0, i_centre, j_centre)
    cont_search_values = cont_array[ where( cont_array_mask==1 ) ]

    # If no features found, only return central "cross" of pixels; otherwise, identify primary feature
    if int(sum(cont_search_values)) == 0:
        cont_array = np.zeros([(cutout.shape)[0], (cutout.shape)[1]])
        cont_array[i_centre, j_centre] = 1
        cont_array[i_centre+1, j_centre] = 1
        cont_array[i_centre-1, j_centre] = 1
        cont_array[i_centre, j_centre+1] = 1
        cont_array[i_centre, j_centre-1] = 1
    else:

        # Take mode of values
        cont_search_values = array(cont_search_values)
        cont_target = scipy.stats.mode(cont_search_values[where(cont_search_values>0)])[0][0]

        # Remove all features other than primary, set value of primary feature to 1
        cont_array[where(cont_array!=cont_target)] = 0
        cont_array[where(cont_array!=0)] = 1

        # If feautre contains fewer than 5 pixels, once again default to central "cross"
        if int(sum(cont_search_values)) < 5:
            cont_array = np.zeros([(cutout.shape)[0], (cutout.shape)[1]])
            cont_array[i_centre, j_centre] = 1
            cont_array[i_centre+1, j_centre] = 1
            cont_array[i_centre-1, j_centre] = 1
            cont_array[i_centre, j_centre+1] = 1
            cont_array[i_centre, j_centre-1] = 1

    # Report array and count
    return cont_array



# Function that combines all of the ellipse-fitting steps (finds convex hull, fits ellipse to this, then finds properties of ellipse)
# Input: x & y coordinates to which the ellipse is to be fitted
# Output: Array of x & y coordinates of ellipse centre, array of ellipse's major & minor axes, ellipse's position angle
def EllipseFit(x,y):

    # Find convex hull of points
    p = np.zeros([x.shape[0],2])
    p[:,0], p[:,1] = x, y
    h = []
    for s in scipy.spatial.ConvexHull(p).simplices:
        h.append(p[s[0]])
        h.append(p[s[1]])
    h = np.array(h)
    x, y = h[:,0], h[:,1]

    # Carry out ellipse-fitting witchcraft
    x = x[:,np.newaxis]
    y = y[:,np.newaxis]
    D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
    S = np.dot(D.T,D)
    C = np.zeros([6,6])
    C[0,2] = C[2,0] = 2; C[1,1] = -1
    E, V =  eig(np.dot(inv(S), C))
    n = np.argmax(np.abs(E))
    a = V[:,n]

    # Calculate and return properties of ellipse
    centre = np.real(ChrisFuncs.EllipseCentre(a))
    axes = np.real(ChrisFuncs.EllipseAxes(a))
    angle = (180/3.14159) * np.real(ChrisFuncs.EllipseAngle(a))
    if axes[0]<axes[1]:
        angle += 90.0
    return np.array([centre, axes, angle, [x,y]])



# Function to calculate the coordinates of the centre of an ellipse produced by EllipseFit
# Input: Ellipse produced by EllipseFit
# Output: Array of x & y coordinates of ellipse centre
def EllipseCentre(a):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    b,c,d,f,g,a = b,c,d,f,g,a
    num = b*b-a*c
    x0=(c*d-b*f)/num
    y0=(a*f-b*d)/num
    return np.array([x0,y0])



# Function to calculate the lengths of the axes of an ellipse produced by EllipseFit
# Input: Ellipse produced by EllipseFit
# Output: Array of ellipse's major & minor axes
def EllipseAxes(a):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    res1=np.sqrt(up/down1)
    res2=np.sqrt(up/down2)
    return np.array([res1, res2])



# Function to calculat the position angle of the centre of an ellipse produced by EllipseFit
# Input: Ellipse produced by EllipseFit
# Output: Ellipse's position angle
def EllipseAngle(a):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    b,c,d,f,g,a = b,c,d,f,g,a
    return 0.5*np.arctan(2*b/(a-c))



# Function to create on-the-fly PSF by stacking point sources
# Input: Map to be used, size of output map desired, array of i-coordinates of point soucres, array of j-coordinates of point soucres
# Output: Constructed PSF for current image
def PSF(fits, dim, stars_i, stars_j):
    stack = np.zeros([stars_i.shape[0], dim, dim])
    for s in range(0, stars_i.shape[0]):
        tile = fits[ stars_i[s]-100:stars_i[s]+(dim/2), stars_j[s]-100:stars_j[s]+(dim/2) ]
        if tile.shape[0] * tile.shape[1] == dim**2:
            tile *= tile.max()**-1
            tile = np.rot90(tile, int(np.round(random.uniform(-0.5,3.5))))
            stack[s, :, :] = tile
    psf = np.zeros([dim, dim])
    for i in range(0, dim):
        for j in range(0, dim):
            psf[i,j] = np.mean(stack[:, i, j])
    psf -= psf.min()
    psf *= (sum(psf))**-1
    return psf



# Function to remove a star from a given position in a map
# Input: Map to be used, i & j coordinates of star to be removed, band to be used (w index: FUV, NUV, u, g, r, i, Z, Y, J, H, K, w1, w2, w3, w4, 100, 160, 250, 350, 500)
# Outout: Map purged of unctuous stars
def PurgeStarPieter(fits, star_i, star_j, w):
	found=False
	r=1
	tot=0	# total signal in aperture
	num=0	# total number of pixels in aperture
	for ii in range(-r,r):			# loop over all pixels in first circular aperture
		for jj in range(-r,r):
			if ii**2+jj**2<=(r)**2:
				tot+=fits[star_i+ii,star_j+jj]
				num+=1
	profileOld=tot/num  # set up the first value of the profile as the average brightness/pixel in aperture
	if CircleSum(fits, star_i, star_j, 5)[0]!=0:
	    while found==False:
		r+=1
		tot=0
		num=0
		for ii in range(-r,r): 		# loop over all pixels in ring-aperture
			for jj in range(-r,r):
				if ii**2+jj**2<=(r)**2 and ii**2+jj**2>=(r-1)**2:
					tot+=fits[star_i+ii,star_j+jj]
					num+=1
		profile=tot/num
		if w<2:
			r_max=10
			Smax=2
			corr=2
			coef=0.02
		elif w<6:
			r_max=60
			Smax=5000
			corr=20
			coef=0.025
		elif w<11:
			r_max=60
			Smax=5000
			corr=20
			coef=0.04
		else:
			r_max=10
			Smax=500000
			corr=5
			coef=0.015
		if profile<Smax and abs(profile)>0.8*abs(profileOld) or r>r_max:  #If the new profile has flattened off enough, start the stellar 	subtraction and end loop
			#r=temp
			if r>2:
				bg_calc=[]
				rmax=int(r+coef*r**2.7)-corr # empirical correction to r, necessary for big sources
				found=True
				for ii in range(-rmax-int((2**0.5-1)*rmax),rmax+int((2**0.5-1)*rmax)):   # loop over 	all pixels in ring-aperture around (but not including) the star (same size aperture for stellar aperture and surrounding ring aperture)
					for jj in range(-rmax-int((2**0.5-1)*rmax),rmax+int((2**0.5-1)*rmax)):
						if ii**2+jj**2<=(rmax+int((2**0.5-1)*rmax))**2 and ii**2+jj**2>=rmax**2:

		        				bg_calc.append(fits[star_i+ii,star_j+jj])   # store 	the values of the pixels in a background
				for ii in range(-rmax,rmax):	# loop over all pixels in stellar aperture
					for jj in range(-rmax,rmax):
						if ii**2+jj**2<=rmax**2:
							fits[star_i+ii,star_j+jj]=random.choice(bg_calc)  # 	exchange the stellar pixels for background pixels from the surrounding area
		profileOld=profile
	return fits



# Function to remove a star from a given position in a map
# Input: Map to be used, i & j coordinates of star to be removed
# Outout: Map purged of unctuous stars
def PurgeStar(fits, star_i, star_j, beam_pix):

    # Make preparations for radial profiling
    beam_pix = int(round(beam_pix))
    r = 1
    fraction = 0.0
    ap = ChrisFuncs.CircleSum(fits, star_i, star_j, r)
    centre = ap[0] / ap[1]
    profile_old = centre
    profile_old_old = centre

    # Assess if star appears to be saturated; if so immediately skip out to a radius of 2 beam-widths
    ann = ChrisFuncs.CircleSum(fits, star_i, star_j, (1*beam_pix)+1)
    ann[0] -= ap[0]
    ann[1] -= ap[1]
    profile = ann[0] / ann[1]
    fraction = profile / np.mean([ profile_old, profile_old_old ])
    if fraction > 0.90:
        fraction = 0.0
        r = 2.0 * beam_pix

    # Loop outward with radial annuli, evaluating steepness of radial profile, in order to identify edge of star
    while fraction < 0.75:
        ann = ChrisFuncs.CircleAnnulusSum(fits, star_i, star_j, r, 1)
        profile = ann[0] / ann[1]
        fraction = profile / np.mean([ profile_old, profile_old_old ])
        profile_old_old = np.copy(profile_old)
        profile_old = np.copy(profile)
        r += 1

    # Once edge of star found, perform empirical correction, and sample average of surrounding background values with radial annulus
    r += int(1e-10 * (0.25*beam_pix)**7.0)
    bg = ChrisFuncs.CircleAnnulusSum(fits, star_i, star_j, r, int(r*1.25))
    average = np.median(bg[2])

    # Replace pixels in region of stellar contamination with background value, and return purified map
    for i in range(-r, r+1):
            for j in range(-r, r+1):
                if i**2.0 + j**2.0 <= r**2.0:
                    try:
                        fits[star_i+i, star_j+j] = average
                    except:
                        continue
    return fits



# Function to create a cutout of a fits file
# Input: Input fits, cutout central ra (deg), cutout central dec (deg), cutout radius (arcsec), fits image extension, boolean stating if an output variable is desired, output fits pathname if required
# Output: HDU of new file
def FitsCutout(pathname, ra, dec, rad_arcsec, exten=0, variable=False, outfile=False):
    #ChrisFuncs.FitsCutout('E:\\Work\\H-ATLAS\\HAPLESS_Cutouts\\Photometry_Cutouts\\2000_Arcsec\\GALEX_AIS\\10_f.fits', 221.10186, 1.6797723, 1000.0, exten=0, outfile='E:\\Work\\H-ATLAS\\HAPLESS_Cutouts\\Photometry_Cutouts\\2000_Arcsec\\GALEX_AIS\\HAPLESS_10_FUV_Temp.fits')

    # Open input fits and extract data
    if isinstance(pathname,str):
        fitsdata_in = astropy.io.fits.open(pathname)
    elif isinstance(pathname,astropy.io.fits.HDUList):
        fitsdata_in = pathname
    fits_in = fitsdata_in[exten].data
    header_in = fitsdata_in[exten].header
    wcs_in = astropy.wcs.WCS(header_in)

    # Determine pixel width by really fucking crudely exctracting cdelt or cd
    try:
        pixsize_in = np.max(np.abs(wcs_in.wcs.cd)) * 3600.0
    except:
        pixsize_in = np.max(np.abs(wcs_in.wcs.cdelt)) * 3600.0
    rad_pix = int( round( float(rad_arcsec) / float(pixsize_in) ) )

    # Embed map in larger array to deal with edge effects
    fits_bigger = np.copy(fits_in)
    margin = int(round(float(rad_pix)))
    fits_bigger = np.zeros([fits_in.shape[0]+(2.0*margin), fits_in.shape[1]+(2.0*margin)])
    fits_bigger[:] = np.NaN
    fits_bigger[margin:fits_bigger.shape[0]-margin, margin:fits_bigger.shape[1]-margin] = fits_in

    # Identify coordinates of central pixel
    wcs_centre = wcs_in.wcs_world2pix(np.array([[float(ra), float(dec)]]), 0)
    i_centre = wcs_centre[0][1] + float(margin)
    j_centre = wcs_centre[0][0] + float(margin)

    # Fail if coords not in map
    if i_centre<0 or i_centre>(fits_bigger.shape)[0] or j_centre<0 or j_centre>(fits_bigger.shape)[1]:
        raise ValueError('Coordinates not located within bounds of map')#pdb.set_trace()

    # Cut out a small section of the map centred upon the source in question
    i_cutout_min = max([0, i_centre-rad_pix])
    i_cutout_max = min([(fits_bigger.shape)[0], i_centre+rad_pix])
    j_cutout_min = max([0, j_centre-rad_pix])
    j_cutout_max = min([(fits_bigger.shape)[1], j_centre+rad_pix])
    cutout_inviolate = fits_bigger[ int(round(i_cutout_min)):int(round(i_cutout_max))+1, int(round(j_cutout_min)):int(round(j_cutout_max))+1 ]

    # Re-calibrate centre coords to account for discrete pixel size
    i_centre = (int(round(float(cutout_inviolate.shape[0])/2.0))) + ( (i_centre-rad_pix) - (int(round(i_cutout_min))) )
    j_centre = (int(round(float(cutout_inviolate.shape[1])/2.0))) + ( (j_centre-rad_pix) - (int(round(j_cutout_min))) )
    i_centre_inviolate, j_centre_inviolate = i_centre, j_centre

    # Construct FITS HDU
    cutout_hdu = astropy.io.fits.PrimaryHDU(cutout_inviolate)
    cutout_hdulist = astropy.io.fits.HDUList([cutout_hdu])
    cutout_header = cutout_hdulist[0].header

    # Populate header
    cutout_wcs = astropy.wcs.WCS(naxis=2)
    cutout_wcs.wcs.crpix = [float(j_centre_inviolate), float(i_centre_inviolate)]
    cutout_wcs.wcs.cdelt = [float(pixsize_in/-3600.0) , float(pixsize_in/-3600.0) ]
    cutout_wcs.wcs.crval = [float(ra), float(dec)]
    cutout_wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    cutout_header = cutout_wcs.to_header()
    cutout_header.set('ORIGIN', 'This FITS cutout created using Chris Clark\'s \"ChrisFuncs\" library')

    # Save, tidy, and return; all to taste
    if outfile!=False:
        cutout_hdulist.writeto(outfile, clobber=True)
    if isinstance(pathname,str):
        fitsdata_in.close()
    if variable==True:
        return cutout_hdulist



# Wrapper of keflavich function to rebin a fits file the coordinate grid of another fits file
# Input: Input fits, comparison fits, imput fits image extension, comparison fits image extension, boolean for if surface brightness instead of flux should be preserved, boolean stating if an output variable is desired, output fits pathname
# Output: HDU of new file
def FitsRebin(pathname_in, pathname_comp, exten_in=0, exten_comp=0, preserve_sb=False, variable=False, outfile=False):
    #ChrisFuncs.FitsRebin('E:\\Work\\H-ATLAS\\HAPLESS_Cutouts\\Photometry_Cutouts\\2000_Arcsec\\GALEX_AIS\\HAPLESS_10_FUV_Temp.fits', 'E:\\Work\H-ATLAS\\HAPLESS_Cutouts\\Photometry_Cutouts\\2000_Arcsec\\HAPLESS_10_w4.fits', outfile='E:\\Work\\H-ATLAS\\HAPLESS_Cutouts\\Photometry_Cutouts\\2000_Arcsec\\GALEX_AIS\\HAPLESS_10_FUV.fits')

    # Open input fits and extract data
    if isinstance(pathname_in,str):
        fitsdata_in = astropy.io.fits.open(pathname_in)
    elif isinstance(pathname_in,astropy.io.fits.HDUList):
        fitsdata_in = pathname_in
    fits_in = fitsdata_in[exten_in].data
    header_in = fitsdata_in[exten_in].header
    wcs_in = astropy.wcs.WCS(header_in)

    # Open comparison fits and extract data
    if isinstance(pathname_comp,str):
        fitsdata_comp = astropy.io.fits.open(pathname_comp)
    elif isinstance(pathname_comp,astropy.io.fits.HDUList):
        fitsdata_comp = pathname_comp
    header_comp = fitsdata_comp[exten_comp].header
    wcs_comp = astropy.wcs.WCS(header_comp)

    # Use keflavich hcongrid to regrid input fits to header of comparison fits
    fits_new = FITS_tools.hcongrid.hcongrid(fits_in, header_in, header_comp)
    hdu_new = astropy.io.fits.PrimaryHDU(fits_new, header=header_comp)
    hdulist_new = astropy.io.fits.HDUList([hdu_new])

    # Determine pixels width by really fucking crudely exctracting cdelt or cd
    try:
        pixsize_in = np.max(np.abs(wcs_in.wcs.cd))
    except:
        pixsize_in = np.max(np.abs(wcs_in.wcs.cdelt))
    try:
        pixsize_comp = np.max(np.abs(wcs_comp.wcs.cd))
    except:
        pixsize_comp = np.max(np.abs(wcs_comp.wcs.cdelt))

    # Unless not needed, rescale pixel values to preserve flux (as opposed to surface brightness)
    if preserve_sb==False:
        factor = float(pixsize_in) / float(pixsize_comp)
        fits_new *= factor**-2.0

    # Save new fits if required
    if outfile!=False:
        hdulist_new.writeto(outfile, clobber=True)

    # Return new HDU if required
    if variable==True:
        return hdulist_new

    # Close files
    if isinstance(pathname_in,str):
        fitsdata_in.close()
    if isinstance(pathname_comp,str):
        fitsdata_comp.close()



# Function to embed a fits file in a larger array of NaNs (for APLpy or the like)
# Input: Input fits pathname, margin to place around array, fits extension of interest, boolean stating if margin is in arcseconds, no pixelsboolean stating if an output variable is desired, output fits pathname
# Output: HDU of new file
def FitsEmbed(pathname, margin, exten=0, variable=False, outfile=False):
    # ChrisFuncs.FitsEmbed('E:\\Work\\Supernovae\\Kepler\\Postgrad\\Kepler_JCMT_CO(2-1)_1.3mm.fits', 0, 200, outfile='E:\\Work\\Supernovae\\Kepler\\Postgrad\\New.fits')

    # Open fits file and extract data
    if isinstance(pathname,str):
        fitsdata = astropy.io.fits.open(pathname)
    elif isinstance(pathname,astropy.io.fits.HDUList):
        fitsdata = pathname
    fits_old = fitsdata[exten].data
    wcs_old = astropy.wcs.WCS(fitsdata[exten].header)

    # Identify wcs coords of central pixel
    ref_coords = wcs_old.wcs_pix2world(np.array([[fits_old.shape[0]/2, fits_old.shape[1]/2]]), 0)

    # Determine pixel width by really fucking crudely exctracting cdelt or cd
    try:
        cdelt = np.max(np.abs(wcs_old.wcs.cd))
    except:
        cdelt = np.max(np.abs(wcs_old.wcs.cdelt))

    # Create larger array
    fits_new = np.zeros([ fits_old.shape[0]+(2*int(margin)), fits_old.shape[1]+(2*int(margin)) ])
    fits_new[:] = np.NaN

    # Plonck old array into new
    if margin>=0:
        fits_new[margin:margin+fits_old.shape[0], margin:margin+fits_old.shape[1]] = fits_old
    elif margin<0:
        fits_new = fits_old[-margin:margin+fits_old.shape[0], -margin:margin+fits_old.shape[1]]

    # Construct fits HDU
    new_hdu = astropy.io.fits.PrimaryHDU(fits_new)
    new_hdulist = astropy.io.fits.HDUList([new_hdu])
    new_header = new_hdulist[0].header

    # Populate header
    new_header.set('CTYPE1', 'RA---TAN')
    new_header.set('CRPIX1', margin+(fits_old.shape[0]/2)+1)
    new_header.set('CRVAL1', ref_coords[0][0])
    new_header.set('CDELT1', -cdelt)
    new_header.set('CTYPE2', 'DEC--TAN')
    new_header.set('CRPIX2', margin+(fits_old.shape[1]/2)+1)
    new_header.set('CRVAL2', ref_coords[0][1])
    new_header.set('CDELT2', cdelt)

    # Save new fits
    if outfile!=False:
        try:
            os.remove(outfile)
            new_hdulist.writeto(outfile)
        except:
            new_hdulist.writeto(outfile)

    # Return new HDU
    if variable==True:
        return new_hdulist

    # Close original file
    fitsdata.close()



# Function to find uncertainty in an array, in terms of distance from given value, out to a certain percentile limit
# Input: Array of numbers to find uncertainty of, percentile range to find uncertainty out to, boolean of whether to return up-and-down bound values
# Output: Percentile uncertainty
def PercentileError(data, value, percentile=66.6, bounds=False):
    data = ChrisFuncs.Nanless(data)
    percentile = np.float(percentile)
    if bounds==False:
        error = ( np.sort( np.abs( data - value ) ) )[ np.int( (percentile/100.0) * data.shape[0] ) ]
        return error
    elif bounds==True:
        data_up = data[ np.where( data>=value ) ]
        data_down = data[ np.where( data<value ) ]
        error_up = ( np.sort( np.abs( data_up - value ) ) )[ np.int( (percentile/100.0) * data_up.shape[0] ) ]
        error_down = ( np.sort( np.abs( data_down - value ) ) )[ np.int( (percentile/100.0) * data_down.shape[0] ) ]
        return [error_down, error_up]



# Function to trim an array to a given size
# Input: Array to be trimmed, i & j coordinates of centre of trimmed region, width of trimmed region
# Output: Trimmed array
def Trim(data, i_centre, j_centre, width):
    box_rad  = int(round(float(width)/2.0))
    i_cutout_min = max([0, i_centre-box_rad])
    i_cutout_max = min([(data.shape)[0], i_centre+box_rad])
    j_cutout_min = max([0, j_centre-box_rad])
    j_cutout_max = min([(data.shape)[1], j_centre+box_rad])
    trimmed = data[ int(round(i_cutout_min)):int(round(i_cutout_max)), int(round(j_cutout_min)):int(round(j_cutout_max)) ]
    i_centre = (int(round(float(trimmed.shape[0])/2.0))+1) + ( (i_centre-box_rad) - (int(round(i_cutout_min))) )
    j_centre = (int(round(float(trimmed.shape[1])/2.0))+1) + ( (j_centre-box_rad) - (int(round(j_cutout_min))) )
    return trimmed, [i_centre, j_centre]



# Function to calculate dust mass
# Input: Flux (Jy), distance (pc), wavelength (m), temperature (K), kappa at 850um (m^2 kg^-1), dust emissivity index
# Output: Dust mass (Msol)
def DustMass(S, parsecs, wavelength, T, kappa850=0.077, beta=2.0):
    c = 3E8
    h = 6.64E-34
    k = 1.38E-23
    nu = c / wavelength
    B_prefactor = (2.0 * h * nu**3.0) / c**2.0
    B_e = (h * nu) / (k * T)
    B = B_prefactor * (e**B_e - 1.0)**-1.0
    nu850 = c / 850E-6
    kappa = kappa850 * (nu / nu850)**beta
    D = parsecs * 3.26 * 9.5E15
    M = (S*1E-26 * D**2.0) / (kappa * B)
    return M / 2E30



# Function to calculate normalisation constant Omega of a particular set of greybody attributes
# Input: Dust mass (Msol), distance (pc), list of [kappa reference wavelength, kappa], beta
# Output: Omega (NOT RENDERED IN JANSKYS)
def OmegaSED(M, D, kappa_list=[850E-6, 0.077], beta=2.0):
    nu_0 = 3E8 / kappa_list[0]
    kappa_0 = kappa_list[1]
    M_kilograms = M * 2E30
    D_metres = D * 3.26 * 9.5E15
    Omega = kappa_0 * nu_0**-beta * D_metres**-2.0 * M_kilograms
    return Omega



# Function to convert the bin-edge output of np.histogram to be bin-centred (and this of same dimensions as bin totals)
# Input: Array of bin edges
# Output: Array of bin centres
def HistBinMerge(bin_edges):
    bin_edges = np.array(bin_edges)
    bin_centres = np.zeros([bin_edges.shape[0]-1])
    for i in range(0, bin_edges.shape[0]-1):
        bin_centres[i] = np.mean([bin_edges[i],bin_edges[i+1]])
    return bin_centres



# Function to plot histogram of input data
# Input: Array of data to be fit
# Output: (plot of data)
def HistPlot(data, n_bins=25, show=True):
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.hist(data, bins=n_bins, facecolor='#0174DF')
    if show==True:
        fig.canvas.draw()
    return fig, ax



# Function to perform gaussian fit to data
# Input: Array of data to be fit
# Output: Mean of fit, standard deviation of fit
def GaussFit(data, n_bins=50):
    def QuickGauss(x, *p):
        A, mu, sigma = p
        return A * np.exp( (-((x-mu)**2.0)) / (2.0*sigma**2.0) )
    hist_tots, hist_bins = np.histogram(data, bins=n_bins)
    hist_bins = ChrisFuncs.HistBinMerge(hist_bins)
    hist_guesses = [np.max(hist_tots), np.mean(data), np.std(data)]
    hist_fit = scipy.optimize.curve_fit(QuickGauss, hist_bins, hist_tots, p0=hist_guesses)
    return abs(hist_fit[0][1]), abs(hist_fit[0][2])



# Function to perform gaussian fit to data, and perform plot of data and fit
# Input: Array of data to be fit
# Output: Mean of fit, standard deviation of fit (,plot of data and fit)
def GaussFitPlot(data, n_bins=50, show=True):
    def QuickGauss(x, *p):
        A, mu, sigma = p
        return A * np.exp( (-((x-mu)**2.0)) / (2.0*sigma**2.0) )
    hist_tots, hist_bins = np.histogram(data, bins=n_bins)
    hist_bins = ChrisFuncs.HistBinMerge(hist_bins)
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



# Function to create thumbnail of cutout
# Input: Array, the downsampling factor, the colourmap to be used
# Returns: Thumbnail as APLpy object
def ThumbnailAPLpy(array, downsample=1.0, colourmap='gist_gray'):
    thumbnail = aplpy.FITSFigure(array, downsample=downsample)
    thumbnail.show_colorscale(cmap=colourmap)#, vmin=0.0, vmax=1.0)
    thumbnail.axis_labels.hide()
    thumbnail.tick_labels.hide()
    thumbnail.ticks.hide()
    return thumbnail



# Function to create thumbnail of photometry cutout
# Input: Array, semi-major axis of source aperture (pix), axial ratio, position angle (deg), i & j coords of centre of ellipse, the inner and outer bounds of the sky aperture (pix, list), width of the line to be drawn (pts), the downsampling factor, the colourmap to be used
# Returns: Thumbnail as APLpy object
def PhotThumbnailAPLpy(array, rad, axial_ratio, angle, i_centre, j_centre, sky, line_width=1.0, downsample=1.0, colourmap='gist_gray'):
    x_centre_fits, y_centre_fits = j_centre+1.0, i_centre+1.0
    thumbnail = aplpy.FITSFigure(array, downsample)
    thumbnail.show_colorscale(cmap=colourmap)
    thumbnail.show_ellipses(x_centre_fits, y_centre_fits, 2.0*float(rad), 2.0*(float(rad)/axial_ratio), angle=angle, edgecolor='cyan', facecolor='none', linewidth=line_width)
    if sky>0:
        thumbnail.show_ellipses(x_centre_fits, y_centre_fits, sky[0]*2.0*float(rad), sky[0]*2.0*(float(rad)/axial_ratio), angle=angle, edgecolor='cyan', facecolor='none', linewidth=line_width/3.0)#, linestyle='dotted')
        thumbnail.show_ellipses(x_centre_fits, y_centre_fits, sky[1]*2.0*float(rad), sky[1]*2.0*(float(rad)/axial_ratio), angle=angle, edgecolor='cyan', facecolor='none', linewidth=line_width/3.0)#, linestyle='dotted')
    thumbnail.axis_labels.hide()
    thumbnail.tick_labels.hide()
    thumbnail.ticks.hide()
    return thumbnail



# Function to quickly save a FITS image to file
# Input: Array to be saved, path to which to save file
# Output: None
def Cutout(array, path='E:\\Work\\Cutout.fits'):
    array_hdu = astropy.io.fits.PrimaryHDU(array)
    try:
        array_hdu.writeto(path)
    except:
        os.remove(path)
        array_hdu.writeto(path)



# Function to quickly write a list to a text file
# Input: List to be written
# Output: File to write to
def QuickWrite(data, outfile, sublists=False):
    outfile = open('/home/herdata/spx7cjc/Dropbox/Work/Tables/DustPedia/DustPedia_SPIRE_Cutouts_File_List.dat', 'w')
    if sublists:
        data = [ ','.join(subdata) for subdata in data ]
    outfile.write('\n'.join(data))
    outfile.close()



# Function to convert an observed brightness into a luminosity
# Input: Flux (Jy), distance (pc) frequency or boolean for nuSnu luminosity (Hz or False), boolean to switch flux input to AB magnitudes
# Output; Luminosity in bolometric solar luminosities
def FluxToLum(flux, dist, freq=False, mags=False):
    if mags==True:
        flux = ChrisFuncs.ABMagsToJy(flux)
    watts_per_hz = 10.0**-26.0 * flux * 4.0 * pi * ( dist * 3.26 * 9.5E15 )**2.0
    if freq==False:
        watts = watts_per_hz
    elif (freq!=False) and (freq>0):
        watts = watts_per_hz * freq
    Lsol = watts / 3.846E26
    return Lsol



# Function to convert SDSS-III "nanomaggies" (nMgy) into pogson magnitudes
# Input: Value to be converted (nanomaggies)
# Returns: Pogson magnitudes (mags; duh)
def nMaggiesToMags(nMaggies):
    mag = 22.5 - ( 2.5*log10(nMaggies) )
    return mag



# Function to convert GAMA data units into AB pogson magnitudes
# Input: Value to be converted (data units)
# Returns: AB pogson magnitudes (mags; duh)
def GAMACountsToMags(GAMA):
    mag = 30.0 - ( 2.5*np.log10(GAMA) )
    return mag



# Function to convert from AB pogson magnitudes into GAMA data units
# Input: Value to be converted (mags)
# Returns: AB pogson magnitudes (data units)
def GAMAMagsToCounts(mag):
    GAMA = 10.0**( (30.0-mag) / 2.5 )
    return GAMA



# Function to convert an uncertainty in AB pogson magnitudes to an uncertainty in GAMA data units
# Input: Uncertainty to be converted (mags), and its associated measurement (mags)
# Returns: Uncertainty in flux density (Jy)
def ErrGAMAMagsToCounts(err, mag):
    counts_down = GAMAMagsToCounts(mag) - GAMAMagsToCounts(mag + err)
    counts_up = GAMAMagsToCounts(mag - err) - GAMAMagsToCounts(mag)
    counts = ( counts_down + counts_up ) / 2.0
    return counts



# Function to convert from AB pogson magnitudes into flux in janskys
# Input: Value to be converted (mags)
# Returns: Source flux density (Jy)
def ABMagsToJy(mag):
    Jy = 0.000001 * 10.0**((23.9-mag)/2.5)
    return Jy



# Function to convert from flux in janskys to AB pogson magnitudes
# Input: Value to be converted (mags)
# Returns: Source flux density (Jy)
def JyToABMags(Jy):
    mag = 23.9 - ( 2.5 * np.log10( Jy * 10**6.0 ) )
    return mag



# Function to convert an uncertainty in AB pogson magnitudes to an uncertainty in janskys
# Input: Uncertainty to be converted (mags), and its associated measurement (mags)
# Returns: Uncertainty in flux density (Jy)
def ErrABMagsToJy(err, mag):
    Jy_down = ABMagsToJy(mag) - ABMagsToJy(mag + err)
    Jy_up = ABMagsToJy(mag - err) - ABMagsToJy(mag)
    Jy = ( Jy_down + Jy_up ) / 2.0
    return Jy



# Function to convery absolute AB pogson magnitudes into solar luminosities
# Input: Absolute AB pogson magnitude (Mags)
# Output: Luminosity (Lsol):
def ABAbsToLsol(Mag):
    Lsol = 10.0**( (4.58 - Mag ) / 2.51 )
    return Lsol



# Function to convert GALEX data units into AB pogson magnitudes
# Input: Value to be converted (data units), GALEX band (1 or FUV for FUV, 2 or NUV for NUV)
# Returns: AB pogson magnitudes (mags; duh)
def GALEXCountsToMags(GALEX,w):
    if w==0 or w=='FUV':
        mag = 18.82 - ( 2.5*log10(GALEX) )
    if w==1 or w=='NUV':
        mag = 20.08 - ( 2.5*log10(GALEX) )
    return mag



# Function to convert AB pogson magnitudes into GALEX data counts
# Input: Value to be converted (mags), GALEX band (1 for FUV, 2 for NUV)
# Returns: Galex  counts (data units)
def GALEXMagsToCounts(mag,w):
    if w==0:
        GALEX = 10.0**( (18.82-mag) / 2.5 )
    if w==1:
        GALEX = 10.0**( (20.08-mag) / 2.5 )
    return GALEX



# Function to convert an RMS deviation in relative linear flux to magnitudes
# Input: Relative RMS deviation in flux
# Output: RMS deviation in mangitude
def RMSFluxToMags(S_rms):
    M_rms = abs( 2.5 * np.log10(1.0-S_rms) )
    return M_rms



# Function to convert an RMS deviation in magnitude to relative linear flux
# Input: RMS deviation in magnitude
# Output: RMS deviation in relative flux
def RMSMagsToFlux(m_rms):
    S_rms = 1.0 - abs( 10.0**(m_rms/-2.5) )
    return S_rms



# New function to convert an uncertainty to log space
# Input: Value, uncertainty
# Output: Logarithmic uncertainty
def LogError(value, error):
    value, error = np.array(value), np.array(error)
    frac = 1.0 + (error/value)
    error_up = value * frac
    error_down = value / frac
    log_error_up = np.abs( np.log10(error_up) - np.log10(value) )
    log_error_down = np.abs( np.log10(value) - np.log10(error_down) )
    return np.mean([log_error_up, log_error_down])



# Function to convert a logarithmic uncertainty to linear space
# Input: Logarithmic value, logarithmic uncertainty, boolean of whether average unlogged errors or return them asymetrically
# Output: Linear uncertainty
def UnlogError(log_value, log_error, bounds=False):
    if bounds==False:
        value = 10**log_value
        log_up = log_value + log_error
        log_down = log_value - log_error
        lin_up = 10**log_up
        lin_down = 10**log_down
        rel_up = lin_up / value
        rel_down = lin_down / value
        frac = np.mean([ rel_up, rel_down**-1 ]) - 1.0
        return frac * value
    elif bounds==True:
        error_up = 10.0**(log_value + log_error) - 10.0**log_value
        error_down = 10.0**(log_value - log_error) - 10.0**log_value
        return [error_up, error_down]



# Function which takes a scipy.interpolate.interp1d interpolator object, and uses it to create a function for linear extrapolation
# Input: Interpolator object from scipy.interpolate.interp1d
# Output: Function to give result of linear extrapolation
def Extrap1D(interpolator):
    xs = interpolator.x
    ys = interpolator.y

    def pointwise(x):
        if x < xs[0]:
            return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
        elif x > xs[-1]:
            return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
        else:
            return interpolator(x)

    def ufunclike(xs):
        return array(map(pointwise, array(xs)))

    return ufunclike



# Funtion which takes a (presumably bootstrap) distribution, and a best fit value, and returns the confidence intervals up and down
# Input: Best fit value, array of distribution, boolean for whether intervals should be in log space, sigma clipping threshold, booleon for whether to sigmap clip with median
# Output: List of interval distance from best fit, down and up
def DisIntervals(best_fit, dis, log_space=False, sigma_thresh=3.0, median=False):
    dis = np.array(dis)
    if log_space==False:
        dis_clip = ChrisFuncs.SigmaClip(dis, sigma_thresh=sigma_thresh, median=median)
    if log_space==True:
        dis_clip = ChrisFuncs.SigmaClip(np.log10(dis), sigma_thresh=sigma_thresh, median=median)
    int_down = np.abs( dis_clip[1] - (dis_clip[1]-dis_clip[0]) )
    int_up = np.abs( dis_clip[1] - (dis_clip[1]-dis_clip[0]) )
    return [int_down, int_up]



# Function to generate appropriate dimensions plot divisions of a figure in along a given direction
# Input: Index of plot element in question, total number of plot elements, dimension of figure, x or y axis,
# Output: Starting position of plot, dimension of plot
def GridPos(n_plot, plot_tot, img_dim, axis='y', nonstandard=False, gaps=False):
    if nonstandard>0:
        base = nonstandard
    elif nonstandard==False:
        if axis=='y':
            base = 6.0
        elif axis=='x':
            base = 8.0
    n_plot, plot_tot, img_dim = float(n_plot), float(plot_tot), float(img_dim)
    margin_start = 0.125 * (base / img_dim)
    margin_end = (1.0 - 0.95) * (base / img_dim)
    fig_start = margin_start
    fig_end = 1.0 - margin_end
    fig_dim = fig_end - fig_start
    plot_dim = fig_dim / plot_tot
    plot_start = fig_start + ((n_plot-1.0) * plot_dim)
    if gaps>0:
        plot_start += (0.5 * gaps) * plot_dim
        plot_dim *= 1.0 - (0.5 * gaps)
    return plot_start, plot_dim



# Function to do a two-tailed chi-square test to calculate the liklihood that the null hypothesis holds true
# Input: Chi-square value of fit, degrees of freedom in model
# Output: Liklihood that nulll hypothesis holds true (where p>0.95 would be grounds for >2-sigma acceptance of the null hypothesis)
def NullHyp(chisq, DoF):

    # Do single-tailed chi-squared test
    single = scipy.stats.chi2.cdf(chisq, int(DoF))

    # If DoF is 1 or 2, chi-squared distribution is single tailed, to nothing further to do
    if DoF<=2:
        null_prob = single

    # If DoF>2, also check for over-fitting
    elif DoF>=3:
#        lower_split = (1.0 - single) / 2.0
#        upper_split = single + lower_split
#        null_prob = np.max(np.array([lower_split, upper_split]))
        chisq_red = chisq / float(DoF)
        if chisq_red<0.0:
            print 'Reduced chi-squared is less than zero - model may over-fitting data!'

    # Return result
    return null_prob



# Function to remove all NaN entries from an array
# Input: Array to be cleansed
# Output: Purified array
def Nanless(bad):
    good = bad[np.where(np.isnan(bad)==False)]
    return good



# Function to add in quadrature all of the (non-NaN) elements of an array:
# Input: Array to be added in quadrature
# Output: Quadrature sum of values
def AddInQuad(values):
    values = np.array(values)
    values = ChrisFuncs.Nanless(values)
    value_tot = 0.0
    for i in range(0, values.shape[0]):
        value_new = values[i]
        value_tot = ( value_tot**2.0 + value_new**2.0 )**0.5
    return value_tot


# Function to aappend an arbitrarily long list of arrays into 1 array:
# Input: List of numpy arrays
# Output: One big appended (1D) numpy array
def PanAppend(arr_list):
    n_arr = len(arr_list)
    arr_app = np.array([])
    for i in range(0,n_arr):
        arr_app = np.append( arr_app, np.array( arr_list[i] ) )
    return arr_app



# Function to add in quadrature all of the (non-NaN) elements of an array:
# Input: Array to be added in quadrature
# Output: Quadrature sum of values
def NanlessKS(array1,array2):
    output = scipy.stats.ks_2samp(ChrisFuncs.Nanless(array1), ChrisFuncs.Nanless(array2))
    return output[1]



# Function to quickly pickle a variable
# Input: Variable to be pickled, name of picklejar, picklejar path
# Output: None
def Pickle(var, name, path='E:\\Users\\Chris\\Dropbox\\Work\\Scripts\\Pickle Jars\\'):
    pickle.dump( var, open( path+name+'.pj', 'wb' ) )



# Function to quickly unpickle a variable
# Input: Name of picklejar, name to be assigned to unpickled variable, picklejar path
# Output: Unpickeled variable
def Unpickle(name, path='E:\\Users\\Chris\\Dropbox\\Work\\Scripts\\Pickle Jars\\'):
    var = pickle.load( open( path+name+'.pj', 'rb' ) )
    return var



# Function to install a package with PIP
# Input: String of package name
# Output: None
def PIP(package):
    pip.main(['install', package])



# Function to use PIP to upgrade all installed packages
# Input: None
# Output: None
def UpgradeAllPIP():
    for dist in pip.get_installed_distributions():
        call("pip install --upgrade --user" + dist.project_name, shell=True)



"""
# IDIUT'S GUIDE TO ELLIPTICAL APERTURES
I assume you also know the location of the ellipse's center. Call that (x0,y0).
Let t be the counterclockwise angle the major axis makes with respect to the
x-axis. Let a and b be the semi-major and semi-minor axes, respectively. If
P = (x,y) is an arbitrary point then do this:

X = (x-x0)*cos(t)+(y-y0)*sin(t); % Translate and rotate coords.
Y = -(x-x0)*sin(t)+(y-y0)*cos(t); % to align with ellipse

If

X^2/a^2+Y^2/b^2

is less than 1, the point P lies inside the ellipse. If it equals 1, it is right on
the ellipse. If it is greater than 1, P is outside.
"""

"""
target = wcs_in.wcs_world2pix(np.array([[178.54283, 0.1388639]]), 0)
cutout_inviolate = ChrisFuncs.PurgeStarPieter(cutout_inviolate, 298.49901264, 304.47718041, 0, 5)
ChrisFuncs.FitsCutout('E:\\Work\\H-ATLAS\\HAPLESS_Cutouts\\HAPLESS_1_NUV.fits', 178.55113899, 0.13662582, 1000, exten=0, variable=False, outfile='E:\\Work\\NUV.fits')
"""