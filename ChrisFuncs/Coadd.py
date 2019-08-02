# Import smorgasbord
import pdb
import sys
import os
current_module = sys.modules[__name__]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import astropy.io.fits
import astropy.wcs
import astropy.convolution
import multiprocessing as mp
import time
import copy
import lmfit
import ChrisFuncs



# Function to set a set of maps to the same level by simply least-squares fitting a constant to their pixel value distributions, to estimate and remove their backgrounds
# Args: Directory of files to be levelled, suffix identifying files to be levelled, optioal directory if a set of smoothed files is to be used to determine the level
# Returns: None
def LevelFITS(fitsfile_dir, target_suffix, convfile_dir=False):

    # Define subfunction that fits flat plane to image, to find level
    def LevelChi(level_params, image):
        level = level_params['level'].value
        chi = np.abs(image - level)
        return chi

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

        # Read in corresponding map from directory containing convolved images
        image_conv = astropy.io.fits.getdata(os.path.join(convfile_dir,fitsfile_list[i]))

        # Fit to level of image; save if first image, otherwise calculate appropriate offset
        image_conv_clipped = image_conv[np.where(image_conv < np.percentile(image_conv, 80))]
        image_conv_clipped = ChrisFuncs.SigmaClip(image_conv_clipped, median=False, sigma_thresh=3.0)[2]

        # Do first round of fitting with a brute force grid search, to get us in the region of the global minimum
        brute_level_params = lmfit.Parameters()
        brute_level_params.add('level', value=np.nanmedian(image_conv), vary=True, min=-1.0*np.nanmax(image_conv), max=np.nanmax(image_conv))
        brute_level_result = lmfit.minimize(LevelChi, brute_level_params, method='brute', args=(image_conv_clipped.flatten(),), Ns=250)#fit_kws={'Ns':20, 'finish':None}
        brute_level = brute_level_result.params['level'].value

        # Now do standard Levenberg-Marquardt optimisation from the grid minimum
        level_params = lmfit.Parameters()
        level_params.add('level', value=brute_level, vary=True)
        level_result = lmfit.minimize(LevelChi, level_params, args=(image_conv_clipped.flatten(),))
        level = level_result.params['level'].value
        if i==0:
            level_ref = level
            continue
        average_offset = level_ref - level

        # Read in unconvolved file, and apply offset
        image_in, header_in = astropy.io.fits.getdata(os.path.join(fitsfile_dir,fitsfile_list[i]), header=True)
        image_out = image_in + average_offset

        # Save corrected file
        save_success = False
        save_fail = 0
        while not save_success:
            try:
                astropy.io.fits.writeto(os.path.join(fitsfile_dir,fitsfile_list[i]), image_out, header=header_in, overwrite=True)
                save_success = True
            except:
                print('Unable to save corrected file '+fitsfile_list[i]+'; reattempting')
                save_fail += 1
                time.sleep(10)
                if save_fail>=5:
                    pdb.set_trace()

