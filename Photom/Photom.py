# Import smorgasbord
import sys
import os
import pdb
current_module = sys.modules[__name__]
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import astropy.io.fits
import astropy.wcs
import astropy.convolution
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
    ellipse_where = np.where( (ellipse_check<=1) & (np.isnan(array)==False) )
    ellipse_tot = sum( array[ ellipse_where ] )
    ellipse_count = ellipse_where[0].shape[0]
    ellipse_pix = array[ ellipse_where ]

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
    annulus_where = np.where( (ellipse_check_outer<=1) & (ellipse_check_inner>1) & (np.isnan(array)==False) )
    annulus_tot = sum( array[ annulus_where ] )
    annulus_count = annulus_where[0].shape[0]
    annulus_pix = array[ annulus_where ]

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
    annulus_where = np.where( (ellipse_check_outer<=1) & (ellipse_check_inner>1) & (np.isnan(array)==False) )
    annulus_tot = sum( array[ annulus_where ] )
    annulus_count = annulus_where[0].shape[0]
    annulus_pix = array[ annulus_where ]

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
    ellipse_where = np.where( (ellipse_check<=1) & (np.isnan(array)==False) )
    ellipse_tot = sum( array[ ellipse_where ] )
    ellipse_count = ellipse_where[0].shape[0]
    ellipse_pix = array[ ellipse_where ]

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
    ellipse_mask[ np.where( ellipse_check<=1 ) ] = 1.0

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
    ellipse_where = np.where( (ellipse_check<=1) & (np.isnan(cutout)==False) )
    ellipse_tot = sum( cutout[ ellipse_where ] )
    ellipse_count = ellipse_where[0].shape[0]
    ellipse_pix = cutout[ ellipse_where ]

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
    annulus_where = np.where( (ellipse_check_outer<=1) & (ellipse_check_inner>1) & (np.isnan(cutout)==False) )
    annulus_tot = sum( cutout[ annulus_where ] )
    annulus_count = annulus_where[0].shape[0]
    annulus_pix = cutout[ annulus_where ]

    # Scale output values down to what they would've been for original array
    annulus_count *= float(upscale)**-2.0

    # Return results
    return [annulus_tot, annulus_count, annulus_pix]



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
            i_centre = np.random.randint(ap_border, fits.shape[0]-ap_border)
            j_centre = np.random.randint(ap_border, fits.shape[1]-ap_border)

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



# Function to find all contiguous pixels that lie above a given flux limit
# Input: Array, radius of guess region (pix), i & j coords of centre of guess region, cutoff value for pixel selection
# Returns: Array of ones and zeros indicating contiguous region
def ContiguousPixels(cutout, rad_initial, i_centre, j_centre, cutoff):

    # Create version of cutout where significant pixels have value 1, insignificant pixels have value 0
    cont_array_binary = np.zeros([(cutout.shape)[0], (cutout.shape)[1]])
    cont_array_binary[np.where(cutout>=cutoff)[0], np.where(cutout>=cutoff)[1]] = 1

    # Use SciPy's label function to identify contiguous features in binary map
    cont_structure = np.array([[1,1,1], [1,1,1], [1,1,1]])
    cont_array = np.zeros([(cutout.shape)[0], (cutout.shape)[1]])
    scipy.ndimage.measurements.label(cont_array_binary, structure=cont_structure, output=cont_array)

    # Identify primary contiguous feature within specified radius of given coordinates
    cont_array_mask = ChrisFuncs.EllipseMask(cont_array, rad_initial, 1.0, 0.0, i_centre, j_centre)
    cont_search_values = cont_array[ np.where( cont_array_mask==1 ) ]

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
        cont_search_values = np.array(cont_search_values)
        cont_target = scipy.stats.mode(cont_search_values[np.where(cont_search_values>0)])[0][0]

        # Remove all features other than primary, set value of primary feature to 1
        cont_array[np.where(cont_array!=cont_target)] = 0
        cont_array[np.where(cont_array!=0)] = 1

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
    E, V =  np.linalg.eig(np.dot(np.linalg.inv(S), C))
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
							fits[star_i+ii,star_j+jj]=np.random.choice(bg_calc)  # 	exchange the stellar pixels for background pixels from the surrounding area
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