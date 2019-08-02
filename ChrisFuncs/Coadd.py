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



# Function to perform full plane-fitting to a set of maps, to match their background levels
# Args: Directory of files to be levelled, suffix identifying files to be levelled, optioal directory if a set of smoothed files is to be used to determine the level
# Returns: None
def FullLevelFITS(fits_dir, target_suffix, conv_dir=False):

    # Define subfunction that subtracts background levels from a set of images, and evaluates the residuals
    def FullLevelChi(level_params, fits_dict):
        chi_list = []
        for fits_file in fits_dict.keys():
            for i in range(len(fits_dict[fits_file]['overlap_files'])):
                comp_index = fits_dict[fits_file]['overlap_indices'][i]
                fits_sub = fits_dict[fits_file]['overlap_map_values'][i] - level_params['level_'+str(fits_dict[fits_file]['index'])].value
                comp_sub = fits_dict[fits_file]['overlap_comp_values'][i] - level_params['level_'+str(comp_index)].value
                resid = fits_sub - comp_sub
                chi = resid / np.sqrt(np.abs(fits_dict[fits_file]['overlap_map_values'][i]))
                chi_list += chi.tolist()
        return np.array(chi_list)

    # As a first pass, and to provide at least a "best effort" processing in case of no/minimal overlap, process all files through basic LevelFITS
    LevelFITS(fits_dir, target_suffix, convfile_dir=False)

    # See if a specific convfile dir is specified
    if not conv_dir:
        conv_dir = fits_dir

    # Make list of files in target directory that have target suffix
    all_files = os.listdir(conv_dir)
    fits_files = []
    for all_file in all_files:
        if target_suffix in all_file:
            fits_files.append(all_file)

    # Loop over each map, and record it to dictionary
    fits_dict = {}
    for i in range(len(fits_files)):
        fits_file = fits_files[i]
        fits_dict[fits_file] = {}
        fits_dict[fits_file]['map'], fits_dict[fits_file]['hdr'] = astropy.io.fits.getdata(os.path.join(conv_dir,fits_file), header=True)
        fits_dict[fits_file]['index'] = i

        # Create binary coverage array for map
        fits_dict[fits_file]['bin'] = fits_dict[fits_file]['map'].copy()
        fits_dict[fits_file]['bin'][np.where(np.isnan(fits_dict[fits_file]['bin']) == False)] = 1.0

    # Loop over each map again, seeing what other maps each overlaps with
    for fits_file in fits_dict.keys():
        fits_dict[fits_file]['overlap_files'] = []
        fits_dict[fits_file]['overlap_indices'] = []
        fits_dict[fits_file]['overlap_map_values'] = []
        fits_dict[fits_file]['overlap_comp_values'] = []
        for comp_file in fits_dict.keys():
            if comp_file == fits_file:
                continue
            overlap = fits_dict[fits_file]['bin'] + fits_dict[comp_file]['bin'] - 1.0

            # If there is overlap, record the pixels in question
            if np.nanmax(overlap) == 1:
                overlap_where = np.where(overlap == 1)
                fits_dict[fits_file]['overlap_files'].append(comp_file)
                fits_dict[fits_file]['overlap_indices'].append(fits_dict[comp_file]['index'])
                fits_dict[fits_file]['overlap_map_values'].append(fits_dict[fits_file]['map'][overlap_where])
                fits_dict[fits_file]['overlap_comp_values'].append(fits_dict[comp_file]['map'][overlap_where])

    # Calculate bounds of possible range of levels
    level_max_list = []
    for fits_file in fits_dict.keys():
        level_max_list.append(np.nanmax(np.abs(fits_dict[fits_file]['map'])))
    level_max_range = max(level_max_list) - (-1.0 * min(level_max_list))

    # Prepare parameter objects and assign guess values to setup LMFit
    level_params = lmfit.Parameters()
    for fits_file in fits_dict.keys():
        i = fits_dict[fits_file]['index']
        level_guess = ChrisFuncs.SigmaClip(fits_dict[fits_file]['map'], median=False, sigma_thresh=3.0)[1]
        level_params.add('level_'+str(i), value=level_guess, vary=True, min=-1.0*level_max_range, max=1.0*level_max_range)

    # Remove maps from fit_dict now they're not longer needed, to speed up optimisation
    for fits_file in fits_dict.keys():
        fits_dict[fits_file].pop('map')
        fits_dict[fits_file].pop('bin')

    # Run LMFit to... well... fit
    print('Optimising background levels')
    level_result = lmfit.minimize(FullLevelChi, level_params, method='differential_evolution', args=(fits_dict,),
                                 fit_kws={'maxiter':20, 'popsize':5, 'tol':0.05, 'polish':False, 'workers':int(0.75*mp.cpu_count())})

    # Apply result levels, and save corrected map to file
    for fits_file in fits_dict.keys():
        i = fits_dict[fits_file]['index']
        fits_dict[fits_file]['map'] -= level_result.params['level_'+str(i)]
        astropy.io.fits.writeto(os.path.join(fits_dir,fits_file), data=fits_dict[fits_file]['map'], header=fits_dict[fits_file]['hdr'], overwrite=True)
    return

