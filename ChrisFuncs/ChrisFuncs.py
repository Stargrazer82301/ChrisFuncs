# Import smorgasbord
import pdb
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


# Function to sum all elements in an ellipse centred on the middle of a given array
# Args: Array, semi-major axis (pix), axial ratio, position angle (deg), i & j coords of centre of ellipse
# Returns: Numpy array containing the sum of the pixel values in the ellipse, total number of pixels counted, and an array containing the pixel values
def EllipseSum(array, rad, axial_ratio, angle, i_centre, j_centre):
    from . import Photom
    return Photom.EllipseSum(array, rad, axial_ratio, angle, i_centre, j_centre)



# Function to sum all elements in an annulus centred upon the middle of the given array
# Args: Array, semi-major axis of inside edge of annulus (pix), width of annulus (pix), axial ratio, position angle (deg), i & j coords of centre of ellipse
# Returns: Numpy array containing the sum of the pixel values in the annulus, the total number of pixels counted, and an array containing the pixel values
def AnnulusSum(array, rad_inner, width, axial_ratio, angle, i_centre, j_centre):
    from . import Photom
    return Photom.AnnulusSum(array, rad_inner, width, axial_ratio, angle, i_centre, j_centre)



# Function to make annular photometry faster by pre-preparing arrays of transposed coords that are to be repeatedly used
# Args: Array, semi-major axis of inside edge of annulus (pix), width of annulus (pix), axial ratio, position angle (deg), i & j coords of centre of ellipse
# Returns: List containing i & j transposed coords
def AnnulusQuickPrepare(array, angle, i_centre, j_centre):
    from . import Photom
    return Photom.AnnulusQuickPrepare(array, angle, i_centre, j_centre)



# Function to sum all elements in an annulus centred upon the middle of the given array, usingpre-prepared transposed coord arrays
# Args: Array, semi-major axis of inside edge of annulus (pix), width of annulus (pix), axial ratio, position angle (deg), i & j coords of centre of ellipse, i & j transposed coord arrays
# Returns: Numpy array containing the sum of the pixel values in the annulus, the total number of pixels counted, and an array containing the pixel values
def AnnulusQuickSum(array, rad_inner, width, axial_ratio, angle, i_centre, j_centre, i_trans, j_trans):
    from . import Photom
    return Photom.AnnulusQuickSum(array, rad_inner, width, axial_ratio, angle, i_centre, j_centre, i_trans, j_trans)



# Function to sum all elements in an annulus centred upon the middle of the given array, using pre-prepared transposed coord arrays
# Args: Array, semi-major axis of ellipse (pix), position angle (deg), i & j coords of centre of ellipse, i & j transposed coord arrays
# Returns: Numpy array containing the sum of the pixel values in the ellipse, the total number of pixels counted, and an array containing the pixel values
def EllipseQuickSum(array, rad, axial_ratio, angle, i_centre, j_centre, i_trans, j_trans):
    from . import Photom
    return Photom.EllipseQuickSum(array, rad, axial_ratio, angle, i_centre, j_centre, i_trans, j_trans)



# Function to return a mask identifying all pixels within an ellipse of given parameters
# Args: Array, semi-major axis (pix), axial ratio, position angle (deg), i & j coords of centre of ellipse
# Returns: Mask array of same dimensions as input array where pixels that lie within ellipse have value 1
def EllipseMask(array, rad, axial_ratio, angle, i_centre, j_centre):
    from . import Photom
    return Photom.EllipseMask(array, rad, axial_ratio, angle, i_centre, j_centre)



# Function to sum all pixel elements inside a given circle... the old-fashioned way
# Args: Array to be used, i & j coordinates of centre of circle, radius of circle
# Returns: Sum of elements within circle, number of pixels within circle
def CircleSum(fits, i_centre, j_centre, r):
    from . import Photom
    return Photom.CircleSum(fits, i_centre, j_centre, r)



# Function to sum all pixel elements inside a given circle... the old-fashioned way
# Args: Array to be used, i & j coordinates of centre of circle, radius of circle
# Returns: Sum of elements within circle, number of pixels within circle
def CircleAnnulusSum(fits, i_centre, j_centre, r, width):
    from . import Photom
    return Photom.CircleAnnulusSum(fits, i_centre, j_centre, r, width)



# Function to sum all elements in an ellipse centred on the middle of an array that has been resized to allow better pixel sampling
# Args: Array, semi-major axis (pix), axial ratio, position angle (deg), i & j coords of centre of ellipse, upscaling factor
# Returns: Numpy array containing the sum of the pixel values in the ellipse, the total number of pixels counted, and an array containing the pixel values
def EllipseSumUpscale(cutout, rad, axial_ratio, angle, i_centre, j_centre, upscale=1):
    from . import Photom
    return Photom.EllipseSumUpscale(cutout, rad, axial_ratio, angle, i_centre, j_centre, upscale=1)



# Function to sum all elements in an annulus centred upon the middle of an array that has been resized to allow better pixel sampling
# Args: Array, semi-major axis of inside edge of annulus (pix), width of annulus (pix), axial ratio, position angle (deg), i & j coords of centre of ellipse, upscaling factor
# Returns: Numpy array containing the sum of the pixel values in the annulus, the total number of pixels counted, and an array containing the pixel values
def AnnulusSumUpscale(cutout, rad_inner, width, axial_ratio, angle, i_centre, j_centre, upscale=1):
    from . import Photom
    return Photom.AnnulusSumUpscale(cutout, rad_inner, width, axial_ratio, angle, i_centre, j_centre, upscale=1)



# Function to iteratively calculate SPIRE aperture noise of photometry cutout using randomly-placed (annular-background-subtracted) circular aperture
# Args: Map, radius of aperture (pix), area of aperture (pix), boolean of whether or not to sky-subtract the noise apertures, relative radius of inner edge of annulus, relative width of annulus, angle of source ellipse, axial ratio of source ellipse
# Returns: Aperture standard deviation, and list of mean background values, list of aperture sum values
def CircularApertureStandardDeviationFinder(fits, area, ann=True, ann_inner=1.5, ann_width=1.0, angle=0.0, axial_ratio=1.0, apertures=100):
    from . import Photom
    return Photom.CircularApertureStandardDeviationFinder(fits, area, ann=True, ann_inner=1.5, ann_width=1.0, angle=0.0, axial_ratio=1.0, apertures=100)



# Function to find all contiguous pixels that lie above a given flux limit
# Args: Array, radius of guess region (pix), i & j coords of centre of guess region, cutoff value for pixel selection
# Returns: Array of ones and zeros indicating contiguous region
def ContiguousPixels(cutout, rad_initial, i_centre, j_centre, cutoff):
    from . import Photom
    return Photom.ContiguousPixels(cutout, rad_initial, i_centre, j_centre, cutoff)



# Function that combines all of the ellipse-fitting steps (finds convex hull, fits ellipse to this, then finds properties of ellipse)
# Args: x & y coordinates to which the ellipse is to be fitted
# Returns: Array of x & y coordinates of ellipse centre, array of ellipse's major & minor axes, ellipse's position angle
def EllipseFit(x,y):
    from . import Photom
    return Photom.EllipseFit(x,y)



# Function to calculate the coordinates of the centre of an ellipse produced by EllipseFit
# Args: Ellipse produced by EllipseFit
# Returns: Array of x & y coordinates of ellipse centre
def EllipseCentre(a):
    from . import Photom
    return Photom.EllipseCentre(a)



# Function to calculate the lengths of the axes of an ellipse produced by EllipseFit
# Args: Ellipse produced by EllipseFit
# Returns: Array of ellipse's major & minor axes
def EllipseAxes(a):
    from . import Photom
    return Photom.EllipseAxes(a)



# Function to calculat the position angle of the centre of an ellipse produced by EllipseFit
# Args: Ellipse produced by EllipseFit
# Returns: Ellipse's position angle
def EllipseAngle(a):
    from . import Photom
    return Photom.EllipseAngle(a)



# Function to create a cutout of a fits file - NOW JUST A WRAPPER OF AN ASTROPY FUNCTION
# Args: Input fits, cutout central ra (deg), cutout central dec (deg), cutout radius (arcsec), pixel width (arcsec), fits image extension, boolean of whether to reproject, boolean stating if an output variable is desired, output fits pathname if required
# Returns: HDU of new file
def FitsCutout(pathname, ra, dec, rad_arcsec, pix_width_arcsec=None, exten=0, reproj=False, variable=False, outfile=False, parallel=True, fast=True):
    from . import Fits
    return Fits.FitsCutout(pathname, ra, dec, rad_arcsec, pix_width_arcsec=pix_width_arcsec, exten=exten, reproj=reproj, variable=variable, outfile=outfile, parallel=parallel, fast=fast)



# Function to embed a fits file in a larger array of NaNs (for APLpy or the like)
# Args: Input fits pathname, margin to place around array, fits extension of interest, boolean stating if margin is in arcseconds, no pixelsboolean stating if an output variable is desired, output fits pathname
# Returns: HDU of new file
def FitsEmbed(pathname, margin, exten=0, variable=False, outfile=False):
    from . import Fits
    return Fits.FitsEmbed(pathname, margin, exten=exten, variable=variable, outfile=outfile)



# Define function to generate a generic FITS header for a given projection
# Args: Central right ascension (deg), central declination (deg), image width (deg), pixel size (arcsec)
# Returns: FITS header
def FitsHeader(ra, dec, map_width_deg, pix_width_arcsec):
    from . import Fits
    return Fits.FitsHeader(ra, dec, map_width_deg, pix_width_arcsec)



# Function to perform a sigma clip upon a set of values
# Args: Array of values, convergence tolerance, state if median instead of mean should be used for clip centrepoint, clipping threshold, boolean for whether sigma of zero can be accepted
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



# Keflavich function to downsample an array
# Args: Array to downsample, downsampling factor, and estiamtor
# Returns: Downsampled array
def Downsample(myarr, factor, estimator=np.nanmean):
    ys,xs = myarr.shape
    crarr = myarr[:ys-(ys % int(factor)),:xs-(xs % int(factor))]
    dsarr = estimator( np.concatenate([[crarr[i::factor,j::factor]
        for i in range(factor)]
        for j in range(factor)]), axis=0)
    return dsarr



# A function to fit and remove a background polynomial to an image, masking a central ellipse
# Args: Array to process, i-coord of masked central ellipse, j-coord of masked central ellipse, semimajor axis of masked central ellipse, axial ratio of masked central ellipse, position angle of masked central ellipse, order of polynomial, sigma threshold at which bright pixels cut off, downsampling factor to use, boolean of whether to only apply polynomial if it makes significant difference to image
# Returns: Poynomial-filtered array, array of the polynomial filter
def PolySub(image_in, mask_centre_i, mask_centre_j, mask_semimaj_pix, mask_axial_ratio, mask_angle, poly_order=5, cutoff_sigma=2.0, downsample=1.0, change_check=False):

    # If requested, downsample image to improve processing time
    downsample_factor = np.round(np.int(downsample))
    if downsample_factor>=2:
        image_ds = Downsample(image_in, downsample_factor)
    else:
        image_ds = image_in

    # Downsample related values accordingly
    mask_semimaj_pix = mask_semimaj_pix / downsample_factor
    mask_centre_i = int(round(float((0.5*mask_centre_i)-1.0)))
    mask_centre_j = int(round(float((0.5*mask_centre_j)-1.0)))

    # Find cutoff for excluding bright pixels by sigma-clipping map
    clip_value = SigmaClip(image_ds, tolerance=0.01, sigma_thresh=2.0, median=True)
    noise_value = clip_value[0]
    field_value = clip_value[1]
    cutoff = field_value + ( cutoff_sigma * noise_value )

    # Mask all image pixels in masking region around source
    image_masked = image_ds.copy()
    ellipse_mask = EllipseMask(image_ds, mask_semimaj_pix, mask_axial_ratio, mask_angle, mask_centre_i, mask_centre_j)
    image_masked[ np.where( ellipse_mask==1 ) ] = np.nan

    # Mask all image pixels identified as being high SNR
    image_masked[ np.where( image_masked>cutoff ) ] = np.nan

    # Use astropy to fit 2-dimensional polynomial to the image
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

    # Create final polynomial filter (undoing downsampling using lorenzoriano GitHub script)
    i_coords, j_coords = np.mgrid[:image_ds.shape[0], :image_ds.shape[1]]
    poly_fit = fit(i_coords, j_coords)
    poly_full = scipy.ndimage.interpolation.zoom(poly_fit, [ float(image_in.shape[0])/float(poly_fit.shape[0]), float(image_in.shape[1])/float(poly_fit.shape[1]) ], mode='nearest')

    # Establish background variation before application of filter
    sigma_thresh = 2.0
    clip_in = SigmaClip(image_in, tolerance=0.005, median=True, sigma_thresh=sigma_thresh)
    bg_in = image_in[ np.where( image_in<clip_in[1] ) ]
    spread_in = np.mean( np.abs( bg_in - clip_in[1] ) )

    # How much reduction in background variation there was due to application of the filter
    image_sub = image_in - poly_full
    clip_sub = SigmaClip(image_sub, tolerance=0.005, median=True, sigma_thresh=sigma_thresh)
    bg_sub = image_sub[ np.where( image_sub<clip_sub[1] ) ]
    spread_sub = np.mean( np.abs( bg_sub - clip_sub[1] ) )
    spread_diff = spread_in / spread_sub

    # If the filter made significant difference, apply to image and return it; otherwise, just return the unaltered map
    if change_check:
        if spread_diff>1.1:
            image_out = image_sub
            poly_out = poly_full
        else:
            image_out = image_ds
            poly_out = np.zeros(image_in.shape)
    else:
        image_out = image_sub
        poly_out = poly_full
    return image_out, poly_out



# Function that provides Galactic extinction correction, via IRSA dust extinction service (which uses the Schlafly & Finkbeiner 2011 prescription)
# Args: RA of target coord (deg), dec of target coord (deg), name of band of interest, (boolean of whether function should be verbose, and meaningless verbose output prefix string)
# Returns: Extinction correction factor (ie, multiply uncorrected flux by this value to yield corrected flux)
def ExtCorrrct(ra, dec, band_name, verbose=True, verbose_prefix=''):

    # Make sure there's a space at the end of the verbose prefix
    if verbose_prefix!='':
        if verbose_prefix[-1:]!=' ':
            verbose_prefix += ' '

    # Offset RA or Dec if either is exactly 0, as this can confuse IRSA
    if np.abs(ra)<0.01:
        ra = 0.01
    if np.abs(dec)<0.01:
        dec = 0.01

    # List bands for which IRSA provids corrections
    excorr_possible = ['GALEX_FUV','GALEX_NUV','SDSS_u','SDSS_g','SDSS_r','SDSS_i','SDSS_z','CTIO_U','CTIO_B','CTIO_V','CTIO_R','CTIO_I','DSS_B','DSS_R','DSS_I','2MASS_J','2MASS_H','2MASS_Ks','UKIRT_Y','UKIRT_J','UKIRT_H','UKIRT_K','Spitzer_3.6','Spitzer_4.5','Spitzer_5.8','Spitzer_8.0','WISE_3.4','WISE_4.6']

    # Check if corrections are available for this band
    photom_band_parsed = BandParse(band_name)
    if photom_band_parsed==None:
        if verbose: print(verbose_prefix+'Unable to parse band name; not conducting Galactic extinction correction for this band.')
        excorr = 1.0
        return excorr
    if photom_band_parsed not in excorr_possible:
        if verbose: print(verbose_prefix+'Galactic extinction correction not available for this band.')
        excorr = 1.0
        return excorr

    # Else if extinction correction is possible, prepare query IRSA dust extinction service
    if verbose: print(verbose_prefix+'Retreiving extinction corrections from IRSA Galactic Dust Reddening & Extinction Service.')
    query_count = 0
    query_success = False
    query_limit = 10

    # Keep trying to access extinction corrections, until it works
    while not query_success:
        if query_count>=query_limit:
            break

        # Carry out query
        try:
            sys.stdout = open(os.devnull, "w")
            irsa_query = astroquery.irsa_dust.IrsaDust.get_extinction_table( str(ra)+', '+str(dec) )
            sys.stdout = sys.__stdout__
            query_success = True
            break

        # Handle exceptions
        except Exception as exception:
            sys.stdout = sys.__stdout__
            if query_count==0:
                if hasattr(exception, 'message'):
                    print(verbose_prefix+'IRSA Galactic Dust Reddening & Extinction Service query failed with error: \"'+repr(exception.message)+'\" - reattempting.')
                else:
                    print(verbose_prefix+'IRSA Galactic Dust Reddening & Extinction Service query failed: reattempting (exception not caught).')
            query_count += 1
            time.sleep(10.0)
    if not query_success:
        print(verbose_prefix+'Unable to access IRSA Galactic Dust Reddening & Extinction Service after '+str(query_limit)+' attemps.')
        raise ValueError('Unable to access IRSA Galactic Dust Reddening & Extinction Service after '+str(query_limit)+' attemps.')

    # Loop over entries in the IRSA table, looking for the current band
    irsa_band_exists = False
    for irsa_band_raw in irsa_query['Filter_name'].tolist():
        irsa_band_parsed = BandParse(irsa_band_raw)
        if irsa_band_parsed==None:
            continue

        # If band found in IRSA table, apply quoted Schlafly & Finkbeiner extinction correction
        if irsa_band_parsed==photom_band_parsed:
            irsa_band_index = np.where( irsa_query['Filter_name']==irsa_band_raw )[0][0]
            irsa_band_excorr_mag = irsa_query['A_SandF'][irsa_band_index]
            irsa_band_excorr = 10.0**( irsa_band_excorr_mag / 2.51 )
            irsa_band_exists = True
            break

    # If band is GALEX, determine appropriate extinction correction using reddening coefficients derived from Cardelli (1989) extinction law (cf Gil de Paz 2007, arXiv:1009.4705, arXiv:1108.2837)
    if (irsa_band_exists==False) and (photom_band_parsed in ['GALEX_FUV','GALEX_NUV']):

        # Get the A(V) / E(B-V) extinction-to-excess ratio in V-band
        irsa_v_index = np.where( irsa_query['Filter_name']=='CTIO V' )[0][0]
        irsa_av_ebv_ratio = irsa_query["A_over_E_B_V_SandF"][irsa_v_index]

        # Get the A(V) attenuation in V-band
        irsa_av = irsa_query["A_SandF"][irsa_v_index]

        # Determine the factor
        if photom_band_parsed=='GALEX_FUV':
            reddening_coeff = 7.9
        elif photom_band_parsed=='GALEX_NUV':
            reddening_coeff = 8.0

        # Calculate and apply the extincton correction
        irsa_band_excorr_mag = reddening_coeff * ( irsa_av / irsa_av_ebv_ratio )
        irsa_band_excorr = 10.0**( irsa_band_excorr_mag / 2.51 )

    # If band is Y-band, use point 36.557% of way between z-band and J-band corrections
    if (irsa_band_exists==False) and (photom_band_parsed=='UKIRT_Y'):
        irsa_z_index = np.where( irsa_query['Filter_name']=='SDSS z' )[0][0]
        irsa_J_index = np.where( irsa_query['Filter_name']=='UKIRT J')[0][0]
        irsa_band_excorr_mag = irsa_query['A_SandF'][irsa_J_index] + ( (1.0-0.36557) * (irsa_query['A_SandF'][irsa_z_index] - irsa_query['A_SandF'][irsa_J_index]) )
        irsa_band_excorr = 10.0**( irsa_band_excorr_mag / 2.51 )
        irsa_band_exists = True

    # Report result and return extinction correction
    import FromGitHub.randlet
    if verbose: print(verbose_prefix+'Galactic extinction correction factor is '+str(FromGitHub.randlet.ToPrecision(irsa_band_excorr,4))+' (ie, '+str(FromGitHub.randlet.ToPrecision(irsa_band_excorr_mag,4))+' magnitudes).')
    return irsa_band_excorr



# A function to determine what particular band a given band name refers to
# Args: The band name to be parsed
# Returns: The parsed band name
def BandParse(band_name_target):

    # Define dictionary containing lists of possible alternate names for each band
    band_names_dict = {'GALEX_FUV':['GALEX_FUV','FUV','FUV-band','GALEX_f','f','f-band'],
                       'GALEX_NUV':['GALEX_NUV','NUV','NUV-band','GALEX_n','n','n-band'],
                       'SDSS_u':['SDSS_u','u','u-band','SDSS_u-band'],
                       'SDSS_g':['SDSS_g','g','g-band','SDSS_g-band'],
                       'SDSS_r':['SDSS_r','r','r-band','SDSS_r-band'],
                       'SDSS_i':['SDSS_i','i','i-band','SDSS_i-band'],
                       'SDSS_z':['SDSS_z','z','z-band','SDSS_z-band','VISTA_Z','VISTA_Z-band'],
                       'CTIO_U':['CTIO_U','CTIO_U-band'],
                       'CTIO_B':['CTIO_B','CTIO_B-band','B','B-band'],
                       'CTIO_V':['CTIO_V','CTIO_V-band','V','V-band'],
                       'CTIO_R':['CTIO_R','CTIO_R-band'],
                       'CTIO_I':['CTIO_I','CTIO_I-band'],
                       'DSS_B':['DSS_B','DSS1_B','DSSI_B','DSS2_B','DSSII_B','DSS_B-band','DSS1_B-band','DSSI_B-band','DSS2_B-band','DSSII_B-band','DSS_G','DSS1_G','DSSI_G','DSS2_G','DSSII_G','DSS_G-band','DSS1_G-band','DSSI_G-band','DSS2_G-band','DSSII_G-band'],
                       'DSS_R':['DSS_R','DSS1_R','DSSI_R','DSS2_R','DSSII_R','DSS_R-band','DSS1_R-band','DSSI_R-band','DSS2_R-band','DSSII_R-band'],
                       'DSS_I':['DSS_I','DSS1_I','DSSI_I','DSS2_I','DSSII_I','DSS_I-band','DSS1_I-band','DSSI_I-band','DSS2_I-band','DSSII_I-band'],
                       '2MASS_J':['2MASS_J','J','J-band','2MASS_J-band'],
                       '2MASS_H':['2MASS_H','H','H-band','2MASS_H-band'],
                       '2MASS_Ks':['2MASS_Ks','Ks','Ks-band','2MASS_Ks-band','2MASS_K','2MASS_K-band','VISTA_Ks','VISTA_Ks-band','VISTA_K','VISTA_K-band','VIRCAM_Ks','VIRCAM_Ks-band','VIRCAM_K','VIRCAM_K-band'],
                       'UKIRT_Y':['UKIRT_Y','UKIRT_Y-band','UKIDSS_Y','UKIDSS_Y-band','WFCAM_Y','WFCAM_Y-band','VISTA_Y','VISTA_Y-band','VIRCAM_Y','VIRCAM_Y-band'],
                       'UKIRT_J':['UKIRT_J','UKIRT_J-band','UKIDSS_J','UKIDSS_J-band','WFCAM_J','WFCAM_J-band','VISTA_J','VISTA_J-band','VIRCAM_J','VIRCAM_J-band'],
                       'UKIRT_H':['UKIRT_H','UKIRT_H-band','UKIDSS_H','UKIDSS_H-band','WFCAM_H','WFCAM_H-band','VISTA_H','VISTA_H-band','VIRCAM_H','VIRCAM_H-band'],
                       'UKIRT_K':['UKIRT_K','UKIRT_K-band','K','K-band','UKIDSS_K','UKIDSS_K-band','WFCAM_K','WFCAM_K-band'],
                       'Spitzer_3.6':['Spitzer_3.6','Spitzer_3.6um','Spitzer_3.6mu','Spitzer_IRAC_3.6','Spitzer_IRAC_3.6um','Spitzer_IRAC_3.6mu','Spitzer_IRAC1','Spitzer_I1','IRAC_3.6','IRAC_3.6um','IRAC_3.6mu','IRAC1','I1','Spitzer_IRAC1-band','IRAC1-band','I1-band','3.6','3.6um','3.6mu'],
                       'Spitzer_4.5':['Spitzer_4.5','Spitzer_4.5um','Spitzer_4.5mu','Spitzer_IRAC_4.5','Spitzer_IRAC_4.5um','Spitzer_IRAC_4.5mu','Spitzer_IRAC2','Spitzer_I2','IRAC_4.5','IRAC_4.5um','IRAC_4.5mu','IRAC2','I2','Spitzer_IRAC2-band','IRAC2-band','I2-band','4.5','4.5um','4.5mu'],
                       'Spitzer_5.8':['Spitzer_5.8','Spitzer_5.8um','Spitzer_5.8mu','Spitzer_IRAC_5.8','Spitzer_IRAC_5.8um','Spitzer_IRAC_5.8mu','Spitzer_IRAC3','Spitzer_I3','IRAC_5.8','IRAC_5.8um','IRAC_5.8mu','IRAC3','I3','Spitzer_IRAC3-band','IRAC3-band','I3-band','5.8','5.8um','5.8mu'],
                       'Spitzer_8.0':['Spitzer_8.0','Spitzer_8.0um','Spitzer_8.0mu','Spitzer_IRAC_8.0','Spitzer_IRAC_8.0um','Spitzer_IRAC_8.0mu','Spitzer_IRAC4','Spitzer_I4','IRAC_8.0','IRAC_8.0um','IRAC_8.0mu','IRAC4','I4','Spitzer_IRAC4-band','IRAC4-band','I4-band','8.0','8.0um','8.0mu','Spitzer_8','Spitzer_8m','Spitzer_8mu','Spitzer_IRAC_8','Spitzer_IRAC_8um','Spitzer_IRAC_8mu','IRAC_8','IRAC_8um','IRAC_8mu','8','8um','8mu'],
                       'WISE_3.4':['WISE_3.4','WISE_3.4um','WISE_3.4mu','WISE1','WISE1-band','W1','W1-band','WISE_W1','WISE_W1-band'],
                       'WISE_4.6':['WISE_4.6','WISE_4.6um','WISE_4.6mu','WISE2','WISE2-band','W2','W2-band','WISE_W2','WISE_W2-band'],
                       'PACS_70':['PACS_70','Herschel_70','Herschel-PACS_70'],
                       'PACS_100':['PACS_100','Herschel_100','Hercshel-PACS_100'],
                       'PACS_160':['PACS_160','Herschel_160','Herschel-PACS_160'],
                       'SPIRE_250':['SPIRE_250','Hercshel_250','Hercshel-SPIRE_250'],
                       'SPIRE_350':['SPIRE_350','Hercshel_350','Hercshel-SPIRE_350'],
                       'SPIRE_500':['SPIRE_500','Hercshel_500','Hercshel-SPIRE_500'],
                       'Planck_350':['Planck_857','Planck-HFI_350','Planck-HFI_857','HFI_350','HFI_857'],
                       'Planck_550':['Planck_545','Planck-HFI_550','Planck-HFI_545','HFI_550','HFI_545'],
                       'Planck_850':['Planck_353','Planck-HFI_850','Planck-HFI_353','HFI_850','HFI_353'],
                       'Planck_1380':['Planck_217','Planck-HFI_1380','Planck-HFI_217','HFI_1380','HFI_217'],
                       'Planck_2100':['Planck_143','Planck-HFI_2100','Planck-HFI_143','HFI_2100','HFI_143'],
                       'Planck_3000':['Planck_100','Planck-HFI_3000','Planck-HFI_100','HFI_3000','HFI_100'],
                       'Planck_4260':['Planck_70','Planck-LFI_4260','Planck_LFI_70','LFI_4260','LFI_70'],
                       'Planck_6810':['Planck_44','Planck-LFI_6810','Planck_LFI_44','LFI_6810','LFI_44'],
                       'Planck_10600':['Planck_30','Planck-LFI_10600','Planck_LFI_30','LFI_10600','LFI_30'],}

    # Loop over alternate band name dictionary entries
    band_altnames_matches = []
    for band_name_key in band_names_dict.keys():
        for band_altname in band_names_dict[band_name_key]:

            # Make band names all-lowercase and alphanumeric-only, for ease of comparison
            band_name_target_comp = re.sub(r'\W+', '', band_name_target).replace('_','').lower()
            band_altname_comp = re.sub(r'\W+', '', band_altname).replace('_','').lower()

            # Strip away wavelength and frequency units suffixes, for ease of comparison
            unit_suffixes = ['um','micron','mm','GHz','MHz']
            for unit in unit_suffixes:
                band_name_target_comp = band_name_target_comp.replace(unit,'')
                band_altname_comp = band_altname_comp.replace(unit,'')

            # If target and alternate band names match, record
            if band_name_target_comp==band_altname_comp:
                band_altnames_matches.append(band_name_key)

    # If no matches found, or more than one match found, report null output
    if len(band_altnames_matches)==0:
        return None
    elif len(band_altnames_matches)>1:
        raise Exception('Band name has multiple possible matches! Can you be more specific?')

    # Else if a good match is found, return it
    elif len(band_altnames_matches)==1:
        return band_altnames_matches[0]



# Function to read in file containing transmission functions for instruments, and convert to a dictionary
# Kwargs: Path of a file containing transmisison information for all the bands (or, optionally, leave blank to look for Transmissions.dat in same directory as script).
    # The first row for a given band takes form 'band,[BAND NAME]'.
    # The following row for a given band is optional, and gives information about that band's reference spectrum, taking the form 'ref,[SPECTRUM_DESCRIPTION]'; the spectrum description can be either nu_X, where X is replaced by a number giving the index of some frequency-dependent power law spectrum; of BB_T, where T is replaced by a number giving the temperature of a blackbody spectrum.
    # All subsequent rows for a given band then take the form '[SOME WAVELENGTH IN MICRONS],[TRANSMISSION FRACTION]'.
    # This format can be repeated to fit transmissions data for any number of bands in one file.
# Returns: Dictionary of filter transmissions
def TransmissionDict(path=None):

    # If no path given, assume Tramsissions.dat file is in same directory as script
    if path == None:
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'Transmissions.dat')

    # Read in transmission curves file, and loop over lines
    trans_dict = {'refs':{}}
    with open(path) as curve_file:
        curve_list = curve_file.readlines()
    for i in range(0,len(curve_list)):
        curve_line = curve_list[i]

        # Check if this line indicates start of a new band; if so, start recording new band
        if curve_line[:4] == 'band':
            band = curve_line[5:].replace('\n','')
            trans_dict[band] = []

        # Check if this line contains reference spectrun information; if so, place in the 'refs' dictionary entry
        elif curve_line[:3] == 'ref':
            trans_dict['refs'][band] = curve_line[4:].replace('\n','')

        # Check if this line contains regular transmission information; if so, record
        else:
            trans_dict[band].append(curve_line.replace('\n','').split(','))

    # Loop over filters in filter dict, setting them to be arrays
    for curve in trans_dict.keys():
        if curve != 'refs':
            trans_dict[curve] = np.array(trans_dict[curve]).astype(float)

    # Return finished dictionary
    return trans_dict



# Function to colour correct a flux density, for a given source SED, reference SED, and response curve
# Args: Wavelength of source flux density (in metres), the source spectrum (a Nx2 array, of wavelengths in metres, and fluxes in whatever),
    # the band filter (either a string giving name of a filter in Transmissions.dat, or a Nx2 array of wavelenghts in metres, and transmission fractions),
# Kwargs: Reference spectrum (a Nx2 array of wavelengths in metres, and fluxes; although this can be left to None if this band is in Transmissions.dat),
    # a dictionary containing transmission curves (optional, in case a custom dictionary is desired; must be in same format as yielded by TransmissionDict)
    # and a boolean for fail-unsafe mode (where a correction factor of 1 will be returned if band name not recognised)
# Returns: Colour correction factor (yes, FACTOR)
def ColourCorrect(wavelength, source_spec, band_filter, ref_spec=None, trans_dict=None, fail_unsafe=False):

    # Define physical constants
    c = 3E8
    h = 6.64E-34
    k = 1.38E-23

    # Check if a filter curve has been provided, or if a band name has been given for a common band in Transmissions.dat
    if isinstance(band_filter, np.ndarray):
        band_name = False
    elif isinstance(band_filter, str):
        band_name = copy.copy(band_filter)

        # If a dictionary of curves has already been provided, use it; else read in Transmissions.dat
        if trans_dict == None:
            trans_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'Transmissions.dat')
            if os.path.exists(trans_path):
                trans_dict = TransmissionDict(trans_path)
            else:
                raise Exception('Dictionary of band transmissions not given, and no Transmissions.dat file found in same directory as script')

        # Check that requested filter is actually in dictionary; if it is, grab it, and convert wavelengths from microns to metres
        if band_filter not in trans_dict.keys():
            if fail_unsafe:
                return 1.0
            else:
                raise Exception('Reqested filter not in database of common filters; please provide as an array instead')
        else:
            band_filter = trans_dict[band_filter].copy()
            band_filter[:,0] /= 1E6

    # Check if reference spectrum present in transmission dictionary; if it is, construct spectrum array
    if isinstance(ref_spec, np.ndarray):
        pass
    elif (isinstance(band_name, str)) and (ref_spec == None) and (band_name in trans_dict.keys()) and (band_name in trans_dict['refs'].keys()):
        nu = (c / band_filter[:,0])

        # If reference is a power law, turn into corresponding array of values at same wavelength intervals as filter curve
        if trans_dict['refs'][band_name][:2] == 'nu':
            index = float(trans_dict['refs'][band_name][3:])
            ref_spec = np.zeros(band_filter.shape)
            ref_spec[:,0] = band_filter[:,0]
            ref_spec[:,1] = nu**index
            #ref_spec[:,1] /= np.max(ref_spec[:,1])

        # If reference spectrum is a blackbody, turn into a corresponding array of values at same wavelength intervals as filter curve
        if trans_dict['refs'][band_name][:2] == 'BB':
            temp = float(trans_dict['refs'][band_name][3:])
            ref_spec = np.zeros(band_filter.shape)
            ref_spec[:,0] = band_filter[:,0]
            planck_prefactor = np.divide((2.0 * h * nu**3.0), c**2.0)
            planck_exponent = (h * nu) / (k * temp)
            ref_spec[:,1] = planck_prefactor * (np.e**planck_exponent - 1)**-1.0
            #ref_spec[:,1] /= np.max(ref_spec[:,1])

    # If reference spectrum not in Transmission.dat, nor provided by user, raise exception
    else:
        raise Exception('Reference spectrum not given, nor found in dictionary of band transmissions; please provide reference spectrum')

    # Normalise source and reference SEDs to have observed flux at (interpolated) nominal wavelength
    source_spec[:,1] /= np.interp(wavelength, source_spec[:,0], source_spec[:,1])
    ref_spec[:,1] /= np.interp(wavelength, ref_spec[:,0], ref_spec[:,1])

    # Filter SEDs by response curve (appropriately resampled in wavelength intervals), to record observed flux
    source_obs = source_spec[:,1] * np.interp(source_spec[:,0], band_filter[:,0], band_filter[:,1])
    ref_obs = ref_spec[:,1] * np.interp(ref_spec[:,0], band_filter[:,0], band_filter[:,1])

    # Integrate observed filtered SEDs (in intervals of freqency, as Jy are in terms of Hz)
    source_int = np.trapz(source_obs, x=(c/source_spec[:,0]))
    ref_int = np.trapz(ref_obs, x=(c/ref_spec[:,0]))

    # Calculate and return colour correction factor from integrals
    colour_corr_factor = ref_int / source_int
    return colour_corr_factor



# Function to find uncertainty in an array, in terms of distance from given value, out to a certain percentile limit
# Args: Array of numbers to find uncertainty of, percentile range to find uncertainty out to, boolean of whether to return up-and-down bound values
# Returns: Percentile uncertainty
def PercentileError(data, value, percentile=68.27, bounds=False):
    data = Nanless(data)
    percentile = np.float(percentile)
    if bounds==False:
        error = ( np.sort( np.abs( data - value ) ) )[ np.int( (percentile/100.0) * data.shape[0] ) ]
        return error
    elif bounds==True:
        data_up = data[ np.where( data>=value ) ]
        data_down = data[ np.where( data<value ) ]
        error_up = np.percentile( np.sort(np.abs(data_up-value)), percentile )
        error_down = np.percentile( np.sort(np.abs(data_down-value)), percentile )
        return (error_down, error_up)



# Function to trim an array to a given size
# Args: Array to be trimmed, i & j coordinates of centre of trimmed region, width of trimmed region
# Returns: Trimmed array
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



# Function that uses Driver & Robotham (2010) foruma to give percentage cosmic variance
# Args: Survey volume (in Mpc^3, assuming H0=70 km s^-1 Mpc^-1), number of survey fields, survey field aspect ratio
# Returns: Percentage cosmic variance
def CosmicVariance(v, n, x):
    v, n, x = float(v), float(n), float(x)
    first_term = 1.00 - ( 0.03 * np.sqrt(x-1.0) )
    second_term = ( 219.7 - (52.4*np.log10(v)) + (3.21*(np.log10(v))**2.0) ) / n**0.5
    cv = first_term * second_term
    return cv



# Function to convert the bin-edge output of np.histogram to be bin-centred (and this of same dimensions as bin totals)
# Args: Array of bin edges
# Returns: Array of bin centres
def HistBinMerge(bin_edges):
    bin_edges = np.array(bin_edges)
    bin_centres = np.zeros([bin_edges.shape[0]-1])
    for i in range(0, bin_edges.shape[0]-1):
        bin_centres[i] = np.mean([bin_edges[i],bin_edges[i+1]])
    return bin_centres



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



# Function to perform gaussian fit to data
# Args: Array of data to be fit
# Returns: Mean of fit, standard deviation of fit
def GaussFit(data, n_bins=50):
    def QuickGauss(x, *p):
        A, mu, sigma = p
        return A * np.exp( (-((x-mu)**2.0)) / (2.0*sigma**2.0) )
    hist_tots, hist_bins = np.histogram(data, bins=n_bins)
    hist_bins = HistBinMerge(hist_bins)
    hist_guesses = [np.max(hist_tots), np.mean(data), np.std(data)]
    hist_fit = scipy.optimize.curve_fit(QuickGauss, hist_bins, hist_tots, p0=hist_guesses)
    return abs(hist_fit[0][1]), abs(hist_fit[0][2])



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



# Function to convert an observed brightness into a luminosity
# Args: Flux (Jy), distance (pc) frequency or boolean for nuSnu luminosity (Hz or False), boolean to switch flux input to AB magnitudes
# Output; Luminosity in bolometric solar luminosities
def FluxToLum(flux, dist, freq=False, mags=False):
    if mags==True:
        flux = ABMagsToJy(flux)
    watts_per_hz = 10.0**-26.0 * flux * 4.0 * np.pi * ( dist * 3.26 * 9.5E15 )**2.0
    if freq==False:
        watts = watts_per_hz
    elif (freq!=False) and (freq>0):
        watts = watts_per_hz * freq
    Lsol = watts / 3.846E26
    return Lsol



# Function to convert SDSS-III "nanomaggies" (nMgy) into pogson magnitudes
# Args: Value to be converted (nanomaggies)
# Returns: Pogson magnitudes (mags; duh)
def nMaggiesToMags(nMaggies):
    mag = 22.5 - ( 2.5*np.log10(nMaggies) )
    return mag



# Function to convert GAMA data units into AB pogson magnitudes
# Args: Value to be converted (data units)
# Returns: AB pogson magnitudes (mags; duh)
def GAMACountsToMags(GAMA):
    mag = 30.0 - ( 2.5*np.log10(GAMA) )
    return mag



# Function to convert from AB pogson magnitudes into GAMA data units
# Args: Value to be converted (mags)
# Returns: AB pogson magnitudes (data units)
def GAMAMagsToCounts(mag):
    GAMA = 10.0**( (30.0-mag) / 2.5 )
    return GAMA



# Function to convert from AB pogson magnitudes into flux in janskys
# Args: Value to be converted (mags)
# Returns: Source flux density (Jy)
def ABMagsToJy(mag):
    Jy = 1E-6 * 10.0**((23.9-mag)/2.5)
    return Jy



# Function to convert from flux in janskys to AB pogson magnitudes
# Args: Value to be converted (mags)
# Returns: Source flux density (Jy)
def JyToABMags(Jy):
    mag = 23.9 - ( 2.5 * np.log10( Jy * 10**6.0 ) )
    return mag



# Function to convert an uncertainty in AB pogson magnitudes to an uncertainty in janskys
# Args: Uncertainty to be converted (mags), and its associated measurement (mags)
# Returns: Uncertainty in flux density (Jy)
def ErrABMagsToJy(err, mag):
    Jy_down = ABMagsToJy(mag) - ABMagsToJy(mag + err)
    Jy_up = ABMagsToJy(mag - err) - ABMagsToJy(mag)
    Jy = ( Jy_down + Jy_up ) / 2.0
    return Jy



# Function to convery absolute AB pogson magnitudes into solar luminosities
# Args: Absolute AB pogson magnitude (Mags)
# Returns: Luminosity (Lsol):
def ABAbsToLsol(Mag):
    Lsol = 10.0**( (4.58 - Mag ) / 2.51 )
    return Lsol



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



# Function to convert an RMS deviation in relative linear flux to magnitudes
# Args: Relative RMS deviation in flux
# Returns: RMS deviation in mangitude
def RMSFluxToMags(S_rms):
    M_rms = abs( 2.5 * np.log10(1.0-S_rms) )
    return M_rms



# Function to convert an RMS deviation in magnitude to relative linear flux
# Args: RMS deviation in magnitude
# Returns: RMS deviation in relative flux
def RMSMagsToFlux(m_rms):
    S_rms = 1.0 - abs( 10.0**(m_rms/-2.5) )
    return S_rms



# New function to convert an uncertainty to log space
# Args: Value, uncertainty
# Returns: Logarithmic uncertainty
def LogError(value, error):
    value, error = np.array(value), np.array(error)
    frac = 1.0 + (error/value)
    error_up = value * frac
    error_down = value / frac
    log_error_up = np.abs( np.log10(error_up) - np.log10(value) )
    log_error_down = np.abs( np.log10(value) - np.log10(error_down) )
    return 0.5*(log_error_up+log_error_down)



# Function to convert a logarithmic uncertainty to linear space
# Args: Logarithmic value, logarithmic uncertainty, boolean of whether average unlogged errors or return them asymetrically
# Returns: Linear uncertainty
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



# Function to generate appropriate dimensions plot divisions of a figure in along a given direction
# Args: Index of plot element in question, total number of plot elements, dimension of figure, x or y axis,
# Returns: Starting position of plot, dimension of plot
def GridPos(n_plot, n_tot, img_dim, axis='y', nonstandard=False, gaps=False):
    if nonstandard>0:
        base = nonstandard
    elif nonstandard==False:
        if axis=='y':
            base = 6.0
        elif axis=='x':
            base = 8.0
    n_plot, n_tot, img_dim = float(n_plot), float(n_tot), float(img_dim)
    margin_start = 0.125 * (base / img_dim)
    margin_end = (1.0 - 0.95) * (base / img_dim)
    fig_start = margin_start
    fig_end = 1.0 - margin_end
    fig_dim = fig_end - fig_start
    plot_dim = fig_dim / n_tot
    plot_start = fig_start + (n_plot * plot_dim)
    if gaps>0:
        plot_start += (0.5 * gaps) * plot_dim
        plot_dim *= 1.0 - (0.5 * gaps)
    return plot_start, plot_dim



# Function to find the Sheather-Jones bandwidth estimator (Sheather & Jones, 1991), adapted from: https://github.com/Neojume/pythonABC
# Args: Array of values of which bandwidth will be found
# Returns: Sheather-Jones bandwidth of array
def SheatherJonesBW(x, weights=None):

    # Define Equation 12 from Sheather & Jones (1991)
    def sj12(x, h):
        phi6 = lambda x: (x ** 6 - 15 * x ** 4 + 45 * x ** 2 - 15) * scipy.stats.norm.pdf(x, loc=0.0, scale=1.0)
        phi4 = lambda x: (x ** 4 - 6 * x ** 2 + 3) * scipy.stats.norm.pdf(x, loc=0.0, scale=1.0)
        n = len(x)
        one = np.ones((1, n))
        lam = np.percentile(x, 75) - np.percentile(x, 25)
        a = 0.92 * lam * n ** (-1 / 7.0)
        b = 0.912 * lam * n ** (-1 / 9.0)
        W = np.tile(x, (n, 1))
        W = W - W.T
        W1 = phi6(W / b)
        tdb = np.dot(np.dot(one, W1), one.T)
        tdb = -tdb / (n * (n - 1) * b ** 7)
        W1 = phi4(W / a)
        sda = np.dot(np.dot(one, W1), one.T)
        sda = sda / (n * (n - 1) * a ** 5)
        alpha2 = 1.357 * (abs(sda / tdb)) ** (1 / 7.0) * h ** (5 / 7.0)
        W1 = phi4(W / alpha2)
        sdalpha2 = np.dot(np.dot(one, W1), one.T)
        sdalpha2 = sdalpha2 / (n * (n - 1) * alpha2 ** 5)
        return (scipy.stats.norm.pdf(0, loc=0, scale=np.sqrt(2)) /
                (n * abs(sdalpha2[0, 0]))) ** 0.2 - h

    # Bandwidth estimator for normal case (From paragraph 2.4.2 of Bowman & Azzalini, 1997)
    def hnorm(x, weights=None):
        wmean = lambda x,w: sum(x * w) / float(sum(w))
        wvar = lambda x,w: sum(w * (x - wmean(x, w)) ** 2) / float(sum(w) - 1)
        x = np.asarray(x)
        if weights is None:
            weights = np.ones(len(x))
        n = float(sum(weights))
        if len(x.shape) == 1:
            sd = np.sqrt(wvar(x, weights))
            return sd * (4 / (3 * n)) ** (1 / 5.0)
        if len(x.shape) == 2:
            ndim = x.shape[1]
            sd = np.sqrt(np.apply_along_axis(wvar, 1, x, weights))
            return (4.0 / ((ndim + 2.0) * n) ** (1.0 / (ndim + 4.0))) * sd

    # Actual calculator of bandwidth
    h0 = hnorm(x)
    v0 = sj12(x, h0)
    if v0 > 0:
        hstep = 1.1
    else:
        hstep = 0.9
    h1 = h0 * hstep
    v1 = sj12(x, h1)
    while v1 * v0 > 0:
        h0 = h1
        v0 = v1
        h1 = h0 * hstep
        v1 = sj12(x, h1)
    return h0 + (h1 - h0) * abs(v0) / (abs(v0) + abs(v1))



# Function to remove all NaN entries from an array
# Args: Array to be cleansed
# Returns: Purified array
def Nanless(bad):
    bad = np.array(bad)
    good = bad[np.where(np.isnan(bad)==False)]
    return good




# Function to aappend an arbitrarily long list of arrays into 1 array:
# Args: List of numpy arrays
# Returns: One big appended (1D) numpy array
def PanAppend(arr_list):
    n_arr = len(arr_list)
    arr_app = np.array([])
    for i in range(0,n_arr):
        arr_app = np.append( arr_app, np.array( arr_list[i] ) )
    return arr_app



# Function to wget a file from a given URL to a given directory
# Args: String of target url, string of output filepath, boolean for clobbering, boolean for auto-retrying, boolean for verbosity
# Returns: None
def wgetURL(url, filename, overwrite=True, auto_retry=False):
    if os.path.exists(filename):
        if overwrite:
            os.remove(filename)
        else:
            raise Exception('Output file already exists; if you\'re happy to overwrite it, re-run wgetURL with overwrite=True.')
    success = False
    while success==False:
        try:
            try:
                wget.download(url, out=filename)
            except:
                os.system('wget \"'+url+'\" -O '+filename)
            print('Successful acquisition of '+url)
            success = True
        except:
            print('Failure! Retrying acquistion of '+url)
            time.sleep(0.1)
            success = False
            if not auto_retry:
                raise Exception('Unsuccessful wget attempt.')



# Function to delete a directory "as best it can"; fully if possible, else by recursievely removing all delete-able files
# Args: Path of directory to delete
# Returns: None
def RemoveCrawl(directory):
    try:
        shutil.rmtree(directory)
    except:
        for glob_path in glob.glob(os.path.join(directory,'**'), recursive=True):
            if os.path.isfile(glob_path):
                try:
                    os.remove(glob_path)
                except:
                    pass



# Function to estimate time until a task completes
# Args: List of time taken by each iteration in units of seconds since Unix epoch, total number of iterations
# Returns: Python time string of estimated time/date of completion
def TimeEst(time_list, total, plot=False, raw=False):

    # Convert times into log space, fit trend, project, and un-log
    time_list_log = np.log10(np.array(time_list))
    iter_list = np.arange( 0.0, float(len(time_list)) )
    time_fit_log = np.polyfit(iter_list, time_list_log, 1)

    # Find offset between most recent time value, and fit at that point
    time_latest_actual = time_list[-1:][0]
    time_latest_predicted_log = ( time_fit_log[0] * (float(np.array(time_list).shape[0])-1.0) ) + time_fit_log[1]
    time_latest_offset_log = np.log10(time_latest_actual) - time_latest_predicted_log
    time_fit_log[1] += time_latest_offset_log

    # Predict time of completion
    time_end_log = ( time_fit_log[0] * total ) + time_fit_log[1]
    time_end = 10.0**time_end_log

    # If requested, create plot
    if plot:
        fig = plt.figure(figsize=(8,6))
        ax_dims = [0.125, 0.125, 0.825, 0.825]
        ax = fig.add_axes(ax_dims)

        # Create plotting arrays
        time_list_hours = ( np.array(time_list) - time_list[0] ) / 3600.0
        time_end_hours = ( time_end - time_list[0] ) / 3600.0
        line_x = np.linspace(0, total, 10000)
        line_y_log = ( time_fit_log[0] * line_x ) + time_fit_log[1]
        line_y = ( 10.0**( line_y_log ) - time_list[0] ) / 3600.0

        # Plot points, and line of best fit
        ax.scatter(iter_list, time_list_hours, c='#4D78C9', marker='o', s=25, linewidths=0)
        ax.scatter(total, time_end_hours, c='#C03030', marker='H', s=100, linewidths=0)
        ax.plot(line_x, line_y, ls='--', lw=1.0, c='#C03030')

        # Format axis limts and labels
        ax.set_xlabel(r'Iteration', fontsize=15)
        ax.set_ylabel(r'Time Since Start (hrs)', fontsize=15)
        ax.set_xlim(0.0,1.1*line_x.max())
        ax.set_ylim(0.0,1.1*line_y.max())
        for xlabel in ax.get_xticklabels():
            xlabel.set_fontproperties(matplotlib.font_manager.FontProperties(size=15))
        for ylabel in ax.get_yticklabels():
            ylabel.set_fontproperties(matplotlib.font_manager.FontProperties(size=15))
        ax.grid(linestyle='dotted')
        ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

    # Produce human-readable output, if required
    if not raw:
        time_end = time.strftime('%H:%M:%S %a %d %b %Y', time.localtime(time_end))

    # Return estimate (and plot, if requested)
    if plot:
        return time_end, fig
    elif not plot:
        return time_end



# Function that uses small files in a temporary folder to track progress of parallel functions
# Input; Progress directory to use, total number of iterations to be completed
# Outut: How many iterations have completed, and estimated completion time
def ProgressDir(prog_dir, iter_total, raw=False):

    # If directory doens't seem to exist, wait a bit to see if some parallel process makes it first
    prog_dir_exists = False
    for i in range(0,5):
        if os.path.exists(prog_dir):
            prog_dir_exists = True
            break
        else:
            time.sleep(1.0 + (5.0 * np.random.random()))

    # If directory definately doesn't already exist, create it, add time file, and finish up
    if not prog_dir_exists:
        try:
            os.mkdir(prog_dir)
            prog_file = open( os.path.join(prog_dir, str(time.time())), 'w')
            prog_file.close()
            return 1, 'pending'
        except:
            prog_dir_exists = True

    # Else if progress directroy does indeed exist, carry on
    if prog_dir_exists:

        # Create file in directory, with filename recording the current time (assuming no identically-named file exists)
        while True:
            prog_filename = os.path.join(prog_dir, str(time.time()))
            if not os.path.exists(prog_filename):
                prog_file = open(prog_filename, 'w')
                prog_file.close()
                break

        # List of all files in directory, and convert to list of completion times, and record completed iterations
        prog_list = np.array([ float(prog_time) for prog_time in os.listdir(prog_dir) ])
        prog_list.sort()
        prog_list = prog_list.tolist()
        iter_complete = len(prog_list)

        # Estimate time until completion
        time_est = TimeEst(prog_list, iter_total, raw=raw)

        # If this was the final iteration, clean up
        if iter_complete == iter_total:
            time.sleep(10)
            shutil.rmtree(prog_dir)

        # Return results
        return iter_complete, time_est




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
