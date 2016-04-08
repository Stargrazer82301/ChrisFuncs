# Import smorgasbord
import sys
import os
import pdb
current_module = sys.modules[__name__]
import numpy as np
import scipy.stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import astropy.io.fits
import astropy.wcs
import astropy.convolution
import lmfit
import FITS_tools
import ChrisFuncs



# Function to set a set of maps to the same level but least-squares fitting a plane to their levels to estimate and remove their backgrounds
# Input: Directory of files to be levelled, suffix identifying files to be levelled, optioal directory if a set of smoothed files is to be used to determine the level
# Output: None
def LevelFITS(fitsfile_dir, target_suffix, convfile_dir=False):

    # Define sill ysubfunction that fits flat plane to image, to find level
    def ChisqLevelFITS(level_params, image):
        level = level_params['level'].value
        chi = image - level
        chisq = chi**2.0
        return chisq

    # See if a specific convfile dir is specified
    if not convfile_dir:
        convfile_dir = fitsfile_dir

    # Make list of files in target directory that have target suffix
    allfile_list = os.listdir(fitsfile_dir)
    fitsfile_list = []
    for allfile in allfile_list:
        if target_suffix in allfile:
            fitsfile_list.append(allfile)

    # Loop over each file
    for i in range(0, len(fitsfile_list)):
        print 'Matching backgorund of map '+fitsfile_list[i]

        # Read in corresponding map from directory containing convolved images
        fitsdata_conv = astropy.io.fits.open( os.path.join(convfile_dir,fitsfile_list[i]) )
        image_conv = fitsdata_conv[0].data
        fitsdata_conv.close()

        # Fit to level of image; save if first image, otherwise calculate appropriate offset
        level_params = lmfit.Parameters()
        level_params.add('level', value=np.nanmedian(image_conv), vary=True)
        image_conv_clipped = ChrisFuncs.SigmaClip(image_conv, tolerance=0.005, median=False, sigma_thresh=3.0)[2]
        level_result = lmfit.minimize(FITS_Level_Chisq, level_params, args=(image_conv_clipped.flatten(),))
        level = level_result.params['level'].value
        if i==0:
            level_ref = level
            continue
        average_offset = level_ref - level
        #print 'Applying offset of '+str(average_offset)+' to '+fitsfile_list[i]

        """
        # Save floor and peak values
        floor_value = np.nanmin(image_conv)
        peak_value = ChrisFuncs.SigmaClip( image_conv, tolerance=0.00025, median=False, sigma_thresh=3.0)[1]
        floor_value_list.append(floor_value)
        peak_value_list.append(peak_value)
        if i==0:
            floor_value_ref = floor_value
            peak_value_ref = peak_value
            continue

        # Calculate offsets
        floor_offset = floor_value_ref - floor_value
        peak_offset = peak_value_ref - peak_value
        average_offset = peak_offset#np.mean([ floor_offset, peak_offset ])
        """

        # Read in unconvolved file, and apply offset
        fitsdata_in = astropy.io.fits.open( os.path.join(fitsfile_dir,fitsfile_list[i]) )
        image_in = fitsdata_in[0].data
        header_in = fitsdata_in[0].header
        fitsdata_in.close()
        image_out = image_in + average_offset
        #print 'Map mean of '+fitsfile_list[i]+' changed from '+str(np.nanmean(image_in))+' to '+str(np.nanmean(image_out))

        # Save corrected file
        image_out_hdu = astropy.io.fits.PrimaryHDU(data=image_out, header=header_in)
        image_out_hdulist = astropy.io.fits.HDUList([image_out_hdu])
        image_out_hdulist.writeto( os.path.join(fitsfile_dir,fitsfile_list[i]), clobber=True )
