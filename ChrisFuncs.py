# Import smorgasbord
import sys
import os
import pdb
sys.path.insert(0, '../')
current_module = sys.modules[__name__]
from numpy import *
import numpy as np
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
import importlib
import aplpy
#sys.path.append(os.path.join(dropbox,'Work','Scripts'))

# Import ChrisFuncs and sub-modules
import ChrisFuncs
import Photom



# Function to sum all elements in an ellipse centred on the middle of a given array
# Input: Array, semi-major axis (pix), axial ratio, position angle (deg), i & j coords of centre of ellipse
# Returns: Numpy array containing the sum of the pixel values in the ellipse, total number of pixels counted, and an array containing the pixel values
def EllipseSum(array, rad, axial_ratio, angle, i_centre, j_centre):
    return Photom.EllipseSum(array, rad, axial_ratio, angle, i_centre, j_centre)



# Function to sum all elements in an annulus centred upon the middle of the given array
# Input: Array, semi-major axis of inside edge of annulus (pix), width of annulus (pix), axial ratio, position angle (deg), i & j coords of centre of ellipse
# Returns: Numpy array containing the sum of the pixel values in the annulus, the total number of pixels counted, and an array containing the pixel values
def AnnulusSum(array, rad_inner, width, axial_ratio, angle, i_centre, j_centre):
    return Photom.AnnulusSum(array, rad_inner, width, axial_ratio, angle, i_centre, j_centre)



# Function to make annular photometry faster by pre-preparing arrays of transposed coords that are to be repeatedly used
# Input: Array, semi-major axis of inside edge of annulus (pix), width of annulus (pix), axial ratio, position angle (deg), i & j coords of centre of ellipse
# Returns: List containing i & j transposed coords
def AnnulusQuickPrepare(array, angle, i_centre, j_centre):
    return Photom.AnnulusQuickPrepare(array, angle, i_centre, j_centre)



# Function to sum all elements in an annulus centred upon the middle of the given array, usingpre-prepared transposed coord arrays
# Input: Array, semi-major axis of inside edge of annulus (pix), width of annulus (pix), axial ratio, position angle (deg), i & j coords of centre of ellipse, i & j transposed coord arrays
# Returns: Numpy array containing the sum of the pixel values in the annulus, the total number of pixels counted, and an array containing the pixel values
def AnnulusQuickSum(array, rad_inner, width, axial_ratio, angle, i_centre, j_centre, i_trans, j_trans):
    return Photom.AnnulusQuickSum(array, rad_inner, width, axial_ratio, angle, i_centre, j_centre, i_trans, j_trans)



# Function to sum all elements in an annulus centred upon the middle of the given array, using pre-prepared transposed coord arrays
# Input: Array, semi-major axis of ellipse (pix), position angle (deg), i & j coords of centre of ellipse, i & j transposed coord arrays
# Returns: Numpy array containing the sum of the pixel values in the ellipse, the total number of pixels counted, and an array containing the pixel values
def EllipseQuickSum(array, rad, axial_ratio, angle, i_centre, j_centre, i_trans, j_trans):
    return Photom.EllipseQuickSum(array, rad, axial_ratio, angle, i_centre, j_centre, i_trans, j_trans)



# Function to return a mask identifying all pixels within an ellipse of given parameters
# Input: Array, semi-major axis (pix), axial ratio, position angle (deg), i & j coords of centre of ellipse
# Output: Mask array of same dimensions as input array where pixels that lie within ellipse have value 1
def EllipseMask(array, rad, axial_ratio, angle, i_centre, j_centre):
    return Photom.EllipseMask(array, rad, axial_ratio, angle, i_centre, j_centre)



# Function to sum all pixel elements inside a given circle... the old-fashioned way
# Input: Array to be used, i & j coordinates of centre of circle, radius of circle
# Output: Sum of elements within circle, number of pixels within circle
def CircleSum(fits, i_centre, j_centre, r):
    return Photom.CircleSum(fits, i_centre, j_centre, r)



# Function to sum all pixel elements inside a given circle... the old-fashioned way
# Input: Array to be used, i & j coordinates of centre of circle, radius of circle
# Output: Sum of elements within circle, number of pixels within circle
def CircleAnnulusSum(fits, i_centre, j_centre, r, width):
    return Photom.CircleAnnulusSum(fits, i_centre, j_centre, r, width)



# Function to sum all elements in an ellipse centred on the middle of an array that has been resized to allow better pixel sampling
# Input: Array, semi-major axis (pix), axial ratio, position angle (deg), i & j coords of centre of ellipse, upscaling factor
# Returns: Numpy array containing the sum of the pixel values in the ellipse, the total number of pixels counted, and an array containing the pixel values
def EllipseSumUpscale(cutout, rad, axial_ratio, angle, i_centre, j_centre, upscale=1):
    return Photom.EllipseSumUpscale(cutout, rad, axial_ratio, angle, i_centre, j_centre, upscale=1)



# Function to sum all elements in an annulus centred upon the middle of an array that has been resized to allow better pixel sampling
# Input: Array, semi-major axis of inside edge of annulus (pix), width of annulus (pix), axial ratio, position angle (deg), i & j coords of centre of ellipse, upscaling factor
# Returns: Numpy array containing the sum of the pixel values in the annulus, the total number of pixels counted, and an array containing the pixel values
def AnnulusSumUpscale(cutout, rad_inner, width, axial_ratio, angle, i_centre, j_centre, upscale=1):
    return Photom.AnnulusSumUpscale(cutout, rad_inner, width, axial_ratio, angle, i_centre, j_centre, upscale=1)



# Function to iteratively calculate SPIRE aperture noise of photometry cutout using randomly-placed (annular-background-subtracted) circular aperture
# Input: Map, radius of aperture (pix), area of aperture (pix), boolean of whether or not to sky-subtract the noise apertures, relative radius of inner edge of annulus, relative width of annulus, angle of source ellipse, axial ratio of source ellipse
# Returns: Aperture standard deviation, and list of mean background values, list of aperture sum values
def CircularApertureStandardDeviationFinder(fits, area, ann=True, ann_inner=1.5, ann_width=1.0, angle=0.0, axial_ratio=1.0, apertures=100):
    return Photom.CircularApertureStandardDeviationFinder(fits, area, ann=True, ann_inner=1.5, ann_width=1.0, angle=0.0, axial_ratio=1.0, apertures=100)



# Function to find all contiguous pixels that lie above a given flux limit
# Input: Array, radius of guess region (pix), i & j coords of centre of guess region, cutoff value for pixel selection
# Returns: Array of ones and zeros indicating contiguous region
def ContiguousPixels(cutout, rad_initial, i_centre, j_centre, cutoff):
    return Photom.ContiguousPixels(cutout, rad_initial, i_centre, j_centre, cutoff)



# Function that combines all of the ellipse-fitting steps (finds convex hull, fits ellipse to this, then finds properties of ellipse)
# Input: x & y coordinates to which the ellipse is to be fitted
# Output: Array of x & y coordinates of ellipse centre, array of ellipse's major & minor axes, ellipse's position angle
def EllipseFit(x,y):
    return Photom.EllipseFit(x,y)



# Function to calculate the coordinates of the centre of an ellipse produced by EllipseFit
# Input: Ellipse produced by EllipseFit
# Output: Array of x & y coordinates of ellipse centre
def EllipseCentre(a):
    return Photom.EllipseCentre(a)



# Function to calculate the lengths of the axes of an ellipse produced by EllipseFit
# Input: Ellipse produced by EllipseFit
# Output: Array of ellipse's major & minor axes
def EllipseAxes(a):
    return Photom.EllipseAxes(a)



# Function to calculat the position angle of the centre of an ellipse produced by EllipseFit
# Input: Ellipse produced by EllipseFit
# Output: Ellipse's position angle
def EllipseAngle(a):
    return Photom.EllipseAngle(a)



# Pieter De Vis' function to remove a star from a given position in a map
# Input: Map to be used, i & j coordinates of star to be removed, band to be used (w index: FUV, NUV, u, g, r, i, Z, Y, J, H, K, w1, w2, w3, w4, 100, 160, 250, 350, 500)
# Outout: Map purged of unctuous stars
def PurgeStarPieter(fits, star_i, star_j, w):
    return Photom.PurgeStarPieter(fits, star_i, star_j, w)



# Function to crudely remove a star from a given position in a map
# Input: Map to be used, i & j coordinates of star to be removed
# Outout: Map purged of unctuous stars
def PurgeStar(fits, star_i, star_j, beam_pix):
    return Photom.PurgeStar(fits, star_i, star_j, beam_pix)



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



# Keflavich function to downsample an array
# Input: Array to downsample, downsampling factor, and estiamtor
# Output: Downsampled array
def Downsample(myarr, factor, estimator=np.nanmean):
    ys,xs = myarr.shape
    crarr = myarr[:ys-(ys % int(factor)),:xs-(xs % int(factor))]
    dsarr = estimator( np.concatenate([[crarr[i::factor,j::factor]
        for i in range(factor)]
        for j in range(factor)]), axis=0)
    return dsarr



# Mortcanty function that replicates IDL's congrid
def Congrid(a, newdims, method='linear', centre=False, minusone=False):
    import Congrid
    newa = Congrid.Congrid(a, newdims, method=method, centre=centre, minusone=minusone)
    return newa



# Function that produces polynomial background-removing filter for input image
# Input: Image to be processed, diamter (in pixels) of target source, order of 2-dimensional polynomial that will be fitted to background, downsampling factor (improves runtime), sigma above which bright pixels (ie, bright backgroun dsources) are masked, i-coord of target source (otherwise assumed central), j-coord of target source (otherwise assumed central)
# Output: Best-fit polynomial filter
def PolyFilter(image_in, d25, poly_order=5, downsample_factor=5, cutoff_sigma=2.0, i_centre=False, j_centre=False):

    # Downsample image to improve processing time
    downsample_factor = 5
    image_ds = Downsample(image_in, downsample_factor)
    d25 = d25 / np.float(downsample_factor)
    half_arcmin = 30.0 / np.float(downsample_factor)
    if i_centre==False:
        i_centre = np.int( np.round( ( 0.5 * ( np.float(image_ds.shape[0]) - 1.0 ) ) ) )
    if j_centre==False:
        j_centre = np.int( np.round( ( 0.5 * ( np.float(image_ds.shape[1]) - 1.0 ) ) ) )

    # Make preliminary noise measurement by sigma-clipping cutout
    clip_value = ChrisFuncs.SigmaClip(image_ds, sigma_thresh=5.0, median=True)
    noise_value = clip_value[0]
    field_value = clip_value[1]

    # Find contiguous significant pixels associated with target galaxy
    cutoff = field_value + ( cutoff_sigma * noise_value )
    cont_array = ChrisFuncs.ContiguousPixels(image_ds, half_arcmin, i_centre, j_centre, cutoff)

    # Find ellipse that best fits outline of contiguous region
    cont_x = ((np.where(cont_array == 1))[1])
    cont_y = ((np.where(cont_array == 1))[0])
    if cont_x.shape[0] > 10:
        try:
            cont_ellipse = ChrisFuncs.EllipseFit(cont_x, cont_y)
            opt_semimaj, opt_semimin = max(cont_ellipse[1]), min(cont_ellipse[1])
            opt_axial_ratio = opt_semimaj / opt_semimin
            opt_angle = cont_ellipse[2]
            opt_i_centre, opt_j_centre = cont_ellipse[0][0], cont_ellipse[0][1]
        except:
            pdb.set_trace()
    else:
        opt_semimaj, opt_semimin = half_arcmin, half_arcmin
        opt_axial_ratio = 1.0
        opt_angle = 0.0
        opt_i_centre, opt_j_centre = i_centre, j_centre


    # Mask all image pixels that lie within 2 semi-major axes (or within D25) of ellipse fitted to source
    mask_semimaj = max([ d25, 2.0*opt_semimaj ])
    image_masked = image_ds.copy()
    ellipse_mask = ChrisFuncs.EllipseMask(image_ds, mask_semimaj, opt_axial_ratio, opt_angle, opt_i_centre, opt_j_centre)
    image_masked[ np.where( ellipse_mask==1 ) ] = np.nan

    # Mask all image pixels that lie within 1 arcmin of target coords
    arcmin_mask = ChrisFuncs.EllipseMask(image_ds, half_arcmin, 1.0, 0.0, i_centre, j_centre)
    image_masked[ np.where( arcmin_mask==1 ) ] = np.nan

    # Mask all image pixels previously identified as being high SNR
    image_masked[ np.where( image_masked>cutoff ) ] = np.nan

    # Use astropy to fit 5th-order 2-dimensional polynomial to the image
    image_masked[ np.where( np.isnan(image_masked)==True ) ] = field_value
    poly_model = astropy.modeling.models.Polynomial2D(degree=poly_order)
    i_coords, j_coords = np.mgrid[:image_masked.shape[0], :image_masked.shape[1]]
    fitter = astropy.modeling.fitting.LevMarLSQFitter()
    i_coords = i_coords.flatten()
    j_coords = j_coords.flatten()
    image_flattened = image_masked.flatten()
    good = np.where(np.isnan(image_flattened)==False)
    i_coords = i_coords[good]
    j_coords = j_coords[good]
    image_flattened = image_flattened[good]
    fit = fitter(poly_model, i_coords, j_coords, image_flattened)

    # Create final polynomial filter, and return
    i_coords, j_coords = np.mgrid[:image_ds.shape[0], :image_ds.shape[1]]
    poly_fit = fit(i_coords, j_coords)
    poly_full = ChrisFuncs.Congrid(poly_fit, (image_in.shape[0], image_in.shape[1]), minusone=True)
    return poly_full



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



# Function from StackOverflow (ideasman42) that recursively reloads all submodules of a package
# Input: Package to reload submodules of
# Output: None
def ReloadPackage(package):
    assert(hasattr(package, "__package__"))
    fn = package.__file__
    fn_dir = os.path.dirname(fn) + os.sep
    module_visit = {fn}
    del fn
    def reload_recursive_ex(module):
        importlib.reload(module)
        for module_child in vars(module).values():
            if isinstance(module_child, types.ModuleType):
                fn_child = getattr(module_child, "__file__", None)
                if (fn_child is not None) and fn_child.startswith(fn_dir):
                    if fn_child not in module_visit:
                        # print("reloading:", fn_child, "from", module)
                        module_visit.add(fn_child)
                        reload_recursive_ex(module_child)
    return reload_recursive_ex(package)



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