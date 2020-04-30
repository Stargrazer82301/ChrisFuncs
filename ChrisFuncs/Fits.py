# Import smorgasbord
import sys
import os
import pdb
current_module = sys.modules[__name__]
import numpy as np
import scipy.ndimage
import astropy.io.fits
import astropy.wcs
import astropy.convolution
import reproject
import aplpy
import tempfile
import time
from ChrisFuncs import SigmaClip, Nanless, RemoveCrawl, ImputeImage

# Handle the lack of the basestring class in Python 3
try:
  basestring
except NameError:
  basestring = str



# Function to create a cutout of a fits file - NOW JUST A WRAPPER OF AN ASTROPY FUNCTION
# Args: Input fits, cutout central ra (deg), cutout central dec (deg), cutout radius (arcsec), pixel width (arcsec), fits image extension, boolean of whether to reproject, boolean stating if an output variable is desired, output fits pathname if required
# Returns: HDU of new file
def FitsCutout(pathname, ra, dec, rad_arcsec, pix_width_arcsec=None, exten=0, reproj=False, variable=False, outfile=False, parallel=True, fast=True):

    # Open input fits and extract data
    if isinstance(pathname,basestring):
        in_fitsdata = astropy.io.fits.open(pathname)
    elif isinstance(pathname,astropy.io.fits.HDUList):
        in_fitsdata = pathname
    in_map = in_fitsdata[exten].data
    in_header = in_fitsdata[exten].header
    in_wcs = astropy.wcs.WCS(in_header)

    # If reprojection not requesed, pass input parameters to astropy cutout function
    if reproj==False:
        pos = astropy.coordinates.SkyCoord(ra, dec, unit='deg')
        size = astropy.units.Quantity(2.0*rad_arcsec, astropy.units.arcsec)
        cutout_obj = astropy.nddata.utils.Cutout2D(in_map, pos, size, wcs=in_wcs, mode='partial', fill_value=np.NaN)

        # Extract outputs of interest
        out_map = cutout_obj.data
        out_wcs = cutout_obj.wcs
        out_header = out_wcs.to_header()

    # If reporjection requested, pass input parameters to reprojection function (fast or thorough, as specified)
    if reproj==True:
        import reproject
        width_deg = ( 2.0 * float(rad_arcsec) ) / 3600.0
        if pix_width_arcsec == None:
            pix_width_arcsec = 3600.0*np.mean(np.abs(np.diagonal(in_wcs.pixel_scale_matrix)))
        cutout_header = FitsHeader(ra, dec, width_deg, pix_width_arcsec)
        cutout_shape = ( cutout_header['NAXIS1'],  cutout_header['NAXIS1'] )
        try:
            if fast==False:
                cutout_tuple = reproject.reproject_exact(in_fitsdata, cutout_header, shape_out=cutout_shape, hdu_in=exten, parallel=parallel)
            elif fast==True:
                cutout_tuple = reproject.reproject_interp(in_fitsdata, cutout_header, shape_out=cutout_shape, hdu_in=exten)
        except Exception as exception:
            print(exception.message)

        # Extract outputs of interest
        try:
            out_map = cutout_tuple[0]
        except:
            pdb.set_trace()
        out_wcs = astropy.wcs.WCS(cutout_header)
        out_header = cutout_header

    # Save, tidy, and return; all to taste
    if outfile!=False:
        out_hdu = astropy.io.fits.PrimaryHDU(data=out_map, header=out_header)
        out_hdulist = astropy.io.fits.HDUList([out_hdu])
        out_hdulist.writeto(outfile, overwrite=True)
    if isinstance(pathname,basestring):
        in_fitsdata.close()
    if variable==True:
        out_hdu = astropy.io.fits.PrimaryHDU(data=out_map, header=out_header)
        out_hdulist = astropy.io.fits.HDUList([out_hdu])
        return out_hdulist



# Function to embed a fits file in a larger array of NaNs (for APLpy or the like)
# Args: Input fits pathname, margin to place around array, fits extension of interest, boolean stating if margin is in arcseconds, no pixelsboolean stating if an output variable is desired, output fits pathname
# Returns: HDU of new file
def FitsEmbed(pathname, margin, exten=0, variable=False, outfile=False):

    # Open fits file and extract data
    if isinstance(pathname,basestring):
        fitsdata = astropy.io.fits.open(pathname)
    elif isinstance(pathname,astropy.io.fits.HDUList):
        fitsdata = pathname
    fits_old = fitsdata[exten].data
    wcs_old = astropy.wcs.WCS(fitsdata[exten].header)
    fitsdata.close()

    # Create larger array
    fits_new = np.zeros([ fits_old.shape[0]+(2*int(margin)), fits_old.shape[1]+(2*int(margin)) ])
    fits_new[:] = np.NaN

    # Plonck old array into new
    if margin>=0:
        fits_new[margin:margin+fits_old.shape[0], margin:margin+fits_old.shape[1]] = fits_old
    elif margin<0:
        fits_new = fits_old[-margin:margin+fits_old.shape[0], -margin:margin+fits_old.shape[1]]

    # Populate header
    new_wcs = astropy.wcs.WCS(naxis=2)
    new_wcs.wcs.crpix = [margin+wcs_old.wcs.crpix[0], margin+wcs_old.wcs.crpix[1]]
    new_wcs.wcs.cdelt = wcs_old.wcs.cdelt
    new_wcs.wcs.crval = wcs_old.wcs.crval
    new_wcs.wcs.ctype = wcs_old.wcs.ctype
    new_header = new_wcs.to_header()

    # Construct fits HDU
    new_hdu = astropy.io.fits.PrimaryHDU(data=fits_new, header=new_header)
    new_hdulist = astropy.io.fits.HDUList([new_hdu])

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



# Define function to generate a generic FITS header for a given projection
# Args: Central right ascension (deg), central declination (deg), image width (deg), pixel size (arcsec), rotation (optional, deg), output directory (optional)
# Returns: FITS header
def FitsHeader(ra, dec, map_width_deg, pix_width_arcsec, rotation=None, out_path=False):

    # Calculate map dimensions
    map_width_arcsec = float(map_width_deg) * 3600.0
    map_width_pix = int( np.ceil( map_width_arcsec / float(pix_width_arcsec) ) )
    map_centre_pix = map_width_pix / 2.0
    pix_width_deg = float(pix_width_arcsec) / 3600.0

    # Set up WCS object
    wcs = astropy.wcs.WCS(naxis=2)
    wcs.wcs.crval = [ float(ra), float(dec) ]
    wcs.wcs.crpix = [ map_centre_pix, map_centre_pix ]
    wcs.wcs.cdelt = np.array([ -float(pix_width_deg), float(pix_width_deg) ])
    wcs.wcs.ctype = [ 'RA---TAN', 'DEC--TAN' ]

    # Create empty header, and set map dimensions in it
    header = astropy.io.fits.Header()
    header.set('BITPIX', -64)
    header.set('WCSAXES', 2)
    header.set('NAXIS', 2)
    header.set('NAXIS1', map_width_pix)
    header.set('NAXIS2', map_width_pix)

    # Add WCS parameters to header
    header.set('CRVAL1', wcs.wcs.crval[0])
    header.set('CRVAL2', wcs.wcs.crval[1])
    header.set('CRPIX1', wcs.wcs.crpix[0])
    header.set('CRPIX2', wcs.wcs.crpix[1])
    header.set('CDELT1', wcs.wcs.cdelt[0])
    header.set('CDELT2', wcs.wcs.cdelt[1])
    header.set('CTYPE1', wcs.wcs.ctype[0])
    header.set('CTYPE2', wcs.wcs.ctype[1])

    # If requested, add rotation information
    if rotation:
        header.set('CROTA2', rotation)

    # If requested, write header to file (a la mHdr)
    if isinstance(out_path, str):
        header.totextfile(out_path)

    # Return header
    return header



# Define function create a three-colour PNG image from some FITS files
# Args: Central RA for output image, central Dec of output image, radius of output image in arcseconds, list of paths (or list of lists when giving order-of-preference) to red green and blue files, output directory, (pmin threshold or list of 3 threshold values, pmax threshold or list of 3 threshold values, stretch type or list of stretch types)
# Returns: None
def FitsRGB(ra, dec, rad_arcsec, in_paths, out_dir, pmin=False, pmax=False, stretch='log'):

    # Prepare storage list for paths, then loop over bands
    path_list = [''] * 3
    for b in range(len(path_list)):

        # If this band gives a single path for a specific file, record it
        if in_paths[b].__class__ == str:
            path_list[b] = in_paths[b]

        # Else if this band gives a list of paths (ie, an order of preference), find the first file that exists and record it
        elif hasattr(in_paths[b], '__iter__'):
            for path in in_paths[b]:
                if not os.path.exists(path):
                    continue
                else:
                    path_list[b] = path
                    break

    # Find which band's data has the smallest pixel size
    pix_width_arcsec = 3600
    for k in range(len(path_list)):
        temp_header = astropy.io.fits.getheader(path_list[k])
        temp_wcs = astropy.wcs.WCS(temp_header)
        temp_wcs_pix_matrix = temp_wcs.pixel_scale_matrix
        temp_pix_width_arcsec = 3600.0 * np.mean(np.abs(temp_wcs_pix_matrix[np.where(temp_wcs_pix_matrix!=0)]))
        if temp_pix_width_arcsec<pix_width_arcsec:
            pix_width_arcsec = temp_pix_width_arcsec

    # Loop over data, reprojecting to a common projection that uses the smallest pixel size
    path_zoomed_list = []
    for k in range(len(path_list)):
        path_zoomed_list.append(os.path.join(out_dir, 'RGB_Temp_'+str(k)+'.fits.gz'))
        FitsCutout(path_list[k], ra, dec, rad_arcsec=rad_arcsec, pix_width_arcsec=pix_width_arcsec, reproj=True, outfile=path_zoomed_list[k])

    # Extract pmin values
    if pmin == False:
        pass
    elif hasattr(pmin, '__iter__'):
        pmin_list = pmin
    elif isinstance(float(pmin), float):
        pmin_list = [pmin] * 3

    # Extract pmax values
    if pmax == False:
        pmax_list = [99.75, 99.75, 99.75]
    elif hasattr(pmax, '__iter__'):
        pmax_list = pmax
    elif isinstance(float(pmax), float):
        pmax_list = [pmax] * 3

    # Set up vmin and vmax storage lists, then loop over each band
    vmin_list = [0.0] * 3
    vmax_list = [0.0] * 3
    for k in range(len(path_list)):

        # Calculate band vmin
        vmin_phot = astropy.io.fits.getdata(path_list[k])
        if pmin == False:
            vmin_phot_clip = SigmaClip(vmin_phot, median=True, sigma_thresh=3.0, tolerance=0.0005)
            vmin_list[k] = vmin_phot_clip[1] + (1.75 * vmin_phot_clip[0])
        else:
            vmin_list[k] = np.percentile(Nanless(vmin_phot), pmin_list[k])

        # Calculate band vmax, making sure vmin and vmax end up in right order
        vmax_phot = astropy.io.fits.getdata(path_zoomed_list[k])
        vmax_list[k] = np.percentile(Nanless(vmax_phot), pmax_list[k])
        if vmax_list[k]<vmin_list[k]:
            vmin_list[k] = np.percentile(vmax_phot, pmax_list[k])
            vmax_list[k] = vmin_phot_clip[1] + (1.25 * vmin_phot_clip[0])
        if vmin_list[k]>vmax_list[k]:
            vmin_temp = vmin_list[k]
            vmin_list[k] = vmax_list[k]
            vmax_list[k] = vmin_temp

    # Extract stretch types
    if isinstance(stretch, str):
        stretch_list = [stretch] * 3
    elif hasattr(stretch, '__iter__'):
        stretch_list = stretch

    # Generate RGB image
    aplpy.make_rgb_image(path_zoomed_list,
                         os.path.join(out_dir, 'RGB.png'),
                         stretch_r=stretch_list[0], stretch_g=stretch_list[1], stretch_b=stretch_list[2],
                         vmin_r=vmin_list[0], vmin_g=vmin_list[1], vmin_b=vmin_list[2],
                         vmax_r=vmax_list[0], vmax_g=vmax_list[1], vmax_b=vmax_list[2])

    # Clean up temporary files
    [os.remove(path_zoomed) for path_zoomed in path_zoomed_list]



# Define function to run montage_wrapper in a temp file, and tidy up when done (a Montage wrapper wrapper, if you will)
# Inputs: FITS data to be reprojected, being either an astropy.io.fits.HDU object, or path to FITS file; header to reproject to, being either an astropy.io.fits.Header object, or a string to an mHdr-type text file; (path to directory holding Montage commands, in case this is not already part of the system PATH; path to directory to place temporary files in, in case you have a strong perference in this regard; which FITS extension HDU to use)
# Returns: Array of reprojected data
def MontageWrapperWrapper(in_fitsdata, in_hdr, montage_path=None, temp_path=None, exten=None):

    # Handle Montage path, if kwargs provided
    if montage_path != None:
        os.environ['PATH'] += ':'+montage_path
    import montage_wrapper

    # Produce on-the-fly temporary file
    timestamp = str(time.time()).replace('.','-')
    if isinstance(temp_path, str):
        temp_dir = os.path.join(temp_path, timestamp)
        os.mkdir(temp_dir)
    else:
        temp_dir = tempfile.mkdtemp()

    # If FITS data provided is path to file, record that path; else if FITS data provided is an astropy HDUList object, write it to the temporary directory
    if isinstance(in_fitsdata, basestring):
        if os.path.exists(in_fitsdata):
            in_fits_path = in_fitsdata
        else:
            raise Exception('No FITS file at path provided')
    elif isinstance(in_fitsdata, astropy.io.fits.hdu.image.PrimaryHDU) or isinstance(in_fitsdata, astropy.io.fits.hdu.image.ImageHDU):
        in_fits_path = os.path.join(temp_dir,'temp_in_'+timestamp+'.fits')
        in_fitsdata = astropy.io.fits.HDUList([in_fitsdata])
        in_fitsdata.writeto(in_fits_path)

    # If header provided is path to file, record that path; else if header previded is a astropy Header object, write it to the temporary directory
    if isinstance(in_hdr, basestring):
        if os.path.exists(in_hdr):
            hdr_path = in_hdr
        else:
            raise Exception('No header file at path provided')
    elif isinstance(in_hdr, astropy.io.fits.Header):
        hdr_path = os.path.join(temp_dir,'temp_header_'+timestamp+'.hdr')
        in_hdr.totextfile(hdr_path)

    # Reproject data with montage_wrapper
    out_fits_path = os.path.join(temp_dir,'temp_out_'+timestamp+'.fits')
    montage_wrapper.reproject(in_fits_path, out_fits_path, hdr_path, exact_size=True, hdu=exten)
    out_img = astropy.io.fits.getdata(out_fits_path)

    # If temporary files were placed inside user-supplied temporary directory, delete those files individually
    if isinstance(temp_path, str):
        os.remove(hdr_path)
        os.remove(out_fits_path)

    # Else if using temporary directory produced with tempfile, delete it wholesale
    else:
        RemoveCrawl(temp_dir)

    # Return output array
    return out_img



# Define function to put Montage path into PATH, for use by montage_wrapper
# Inputs: None
# Outputs: Montage path
def MontagePath():

    # Determine machine
    import socket
    location = socket.gethostname()

    # List all combinations of machines and paths we want to cover; first entry in each row should be location of list of locations, second entry should be corresponding Montage directory path (for single location, the location name can start with a * wildcard)
    paths_lists = ([['*replicators', '/Users/cclark/Soft/Montage/bin/'],
                    [['science0.stsci.edu','science3.stsci.edu','science4.stsci.edu','science5.stsci.edu','science6.stsci.edu','science7.stsci.edu'], '/grp/software/linux/rhel6/x86_64/montage/6.0/bin/'],
                    [['science1.stsci.edu','science2.stsci.edu','science8.stsci.edu','science9.stsci.edu','science10.stsci.edu'], '/grp/software/linux/rhel7/x86_64/montage/6.0/bin/'],
                    ['*.stsci.edu', '/grp/software/linux/rhel7/x86_64/montage/6.0/bin/']
                    ])
    paths_array = np.array(paths_lists)

    # Loop over rows, checking to see if location conditions are met
    for i in range(paths_array.shape[0]):
        if isinstance(paths_array[i,0], list) and (location in paths_array[i,0]):
            montage_path = paths_array[i,1]
            break
        elif isinstance(paths_array[i,0], str) and ('*' in paths_array[i,0]) and (paths_array[i,0].replace('*','') in location):
            montage_path = paths_array[i,1]
            break
        elif isinstance(paths_array[i,0], str) and (paths_array[i,0] == location):
            montage_path = paths_array[i,1]
            break

    # Append montage path to PATH environment variable
    os.environ['PATH'] += ':'+montage_path

    # Do test import
    import montage_wrapper
    montage_wrapper.installed

    # Return path
    return montage_path



# Define function for clever fourier combination of images,
# Inputs: HDU containing low-res data; HDU containing high-res data; array of low-res beam gridded to the pixel scale of high-res image; array of high-res beam gridded to pixel scale of high-res image(); optional boolean/float for giving angular scale in degrees at which to apply a tapering transition; boolean of whether to employ subpixel low-pass filter to low-res image to remove pixel edge artefacts; boolean/string for saving combined image to file instead of returning)
# Outputs: The combined image
def FourierCombine(lores_hdu, hires_hdu, lores_beam_img, hires_beam_img, taper_cutoffs_deg=False, apodise=False, to_file=False):

    # Grab high-resolution data, and calculate pixel size
    hires_img = hires_hdu.data.copy()
    hires_hdr = hires_hdu.header
    hires_wcs = astropy.wcs.WCS(hires_hdr)
    hires_pix_width_arcsec = 3600.0 * np.abs(np.max(hires_wcs.pixel_scale_matrix))
    hires_pix_width_deg = hires_pix_width_arcsec / 3600.0

    # Grab low-resolution data, and calculate pixel size
    lores_hdr = lores_hdu.header
    lores_wcs = astropy.wcs.WCS(lores_hdr)
    lores_pix_width_arcsec = 3600.0 * np.abs(np.max(lores_wcs.pixel_scale_matrix))

    # Impute (temporarily) any NaNs surrounding the coverage region with the clipped average of the data (so that the fourier transformers play nice)
    hires_img[np.where(np.isnan(hires_img))] = SigmaClip(hires_img, median=True, sigma_thresh=1.0)[1]

    # Grab low-resolution data, temporarily interpolate over any NaNs, reproject to high-resolution pixel scale
    lores_img = ImputeImage(lores_hdu.data.copy())
    lores_img = astropy.convolution.interpolate_replace_nans(lores_img, astropy.convolution.Gaussian2DKernel(3),
                                                             astropy.convolution.convolve_fft, allow_huge=True, boundary='wrap')
    lores_img = reproject.reproject_interp((lores_img, lores_hdr), hires_hdr, order='bicubic')[0] # Ie, following how SWarp supersamples images
    where_edge = np.where(np.isnan(lores_img))
    lores_img[where_edge] = np.nanmedian(lores_img)

    # If requested, low-pass filter low-resolution data to remove pixel-edge effects
    if apodise:
        lores_apodisation_kernel_sigma = 0.5 * 2.0**-0.5 * (lores_pix_width_arcsec / hires_pix_width_arcsec) # 0.5 * 2.0**-0.5
        lores_apodisation_kernel = astropy.convolution.Gaussian2DKernel(lores_apodisation_kernel_sigma).array
        lores_img = astropy.convolution.convolve_fft(lores_img, lores_apodisation_kernel,
                                                     boundary='reflect', allow_huge=True, preserve_nan=False) # As NaNs already removed

        # Incorporate apodisation filter into the low-resolution beam (as we have to account for the fact that its resolution is now ever-so-slightly lower)
        lores_beam_img = astropy.convolution.convolve_fft(lores_beam_img, lores_apodisation_kernel,
                                                     boundary='reflect', allow_huge=True, preserve_nan=False)
        lores_beam_img -= np.min(lores_beam_img)
        lores_beam_img /= np.sum(lores_beam_img)

    # Fourier transform all the things
    hires_beam_fourier = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(hires_beam_img)))
    lores_beam_fourier = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(lores_beam_img)))
    hires_fourier = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(hires_img)))
    lores_fourier = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(lores_img)))

    # Add miniscule offset to any zero-value elements to stop inf and nan values from appearing later.
    lores_beam_fourier.real[np.where(lores_beam_fourier.real == 0)] = 1E-50


    # Divide the low-resolution data by the low-resolution beam (ie, deconvolve it), then multiply by the high-resoluiton beam, to normalise amplitudes
    fourier_norm = 1 / lores_beam_fourier
    fourier_norm *= hires_beam_fourier
    lores_fourier *= fourier_norm

    # If requested, start by cross-calibrating the hires and lores data within the tapering angular window
    if taper_cutoffs_deg != False:
        hires_fourier_corr = FourierCalibrate(lores_fourier, hires_fourier, taper_cutoffs_deg, hires_pix_width_deg)
        hires_fourier = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(hires_img * hires_fourier_corr)))

        # Perform tapering between specificed angular scales to weight data in Fourier space, following a Hann filter profile
        taper_filter = FourierTaper(taper_cutoffs_deg, hires_wcs)
        hires_weight = 1.0 - taper_filter
        hires_fourier_weighted = hires_fourier.copy()
        hires_fourier_weighted *= hires_weight
        lores_weight = taper_filter
        lores_fourier_weighted = lores_fourier.copy()
        lores_fourier_weighted *= lores_weight

    # Otherwise, in standard operation, use low-resolution beam to weight the tapering from low-resolution to high-resolution data
    else:
        hires_weight = 1.0 - lores_beam_fourier
        hires_fourier_weighted = hires_fourier * hires_weight
        lores_weight = 1.0 * lores_beam_fourier
        lores_fourier_weighted = lores_fourier * lores_weight

    # Combine the images, then convert back out of Fourier space
    comb_fourier = lores_fourier_weighted + hires_fourier_weighted
    comb_fourier_shift = np.fft.ifftshift(comb_fourier)
    comb_img = np.fft.fftshift(np.real(np.fft.ifft2(comb_fourier_shift)))

    # Estimate the size of the low-resolution beam
    lores_beam_demislice = lores_beam_img[int(round(0.5*lores_beam_img.shape[1])):,int(round(0.5*lores_beam_img.shape[1]))]
    lores_beam_width_pix = float(np.argmin(np.abs(lores_beam_demislice - (0.5 * lores_beam_demislice.max()))))

    # Purely for prettiness sake, identify any steppy edge region, and replace with simple interpolation
    hires_mask = hires_hdu.data.copy()
    hires_mask = (hires_mask * 0.0) + 1.0
    hires_mask[np.where(np.isnan(hires_mask))] = 0.0
    hires_mask_dilated_out = scipy.ndimage.binary_dilation(hires_mask, iterations=int(2.0*round(lores_beam_width_pix))).astype(int)
    hires_mask = -1.0 * (hires_mask - 1.0)
    hires_mask_dilated_in = scipy.ndimage.binary_dilation(hires_mask, iterations=int(0.4*round(lores_beam_width_pix))).astype(int)
    hires_mask_border = (hires_mask_dilated_out + hires_mask_dilated_in) - 1
    comb_img[np.where(hires_mask_border)] = np.nan
    comb_img[where_edge] = np.nan
    comb_img[np.where(comb_img == 0)] = np.nan
    comb_img = astropy.convolution.interpolate_replace_nans(comb_img, astropy.convolution.Gaussian2DKernel(round(2.0*lores_beam_width_pix)),
                                                            astropy.convolution.convolve_fft, allow_huge=True, boundary='wrap')

    pdb.set_trace()
    # Return combined image (or save to file if that was requested)
    if not to_file:
        return comb_img
    else:
        astropy.io.fits.writeto(to_file, data=comb_img, header=hires_hdr, overwrite=True)
        return to_file

    # Various tests (included intentionall here after the return, just to save re-typing if need be in future)
    astropy.io.fits.writeto('/astro/dust_kg/cclark/Quest/comb_img.fits', data=comb_img, header=hires_hdr, overwrite=True)
    astropy.io.fits.writeto('/astro/dust_kg/cclark/Quest/hires_img.fits', data=hires_img, header=hires_hdr, overwrite=True)
    astropy.io.fits.writeto('/astro/dust_kg/cclark/Quest/lores_img.fits', data=lores_img, header=hires_hdr, overwrite=True)
    astropy.io.fits.writeto('/astro/dust_kg/cclark/Quest/hires_fourier.fits', data=hires_fourier.real, header=hires_hdr, overwrite=True)
    astropy.io.fits.writeto('/astro/dust_kg/cclark/Quest/lores_fourier.fits', data=lores_fourier.real, header=hires_hdr, overwrite=True)
    astropy.io.fits.writeto('/astro/dust_kg/cclark/Quest/comb_fourier.fits', data=comb_fourier.real, header=hires_hdr, overwrite=True)
    astropy.io.fits.writeto('/astro/dust_kg/cclark/Quest/hires_fourier_weighted.fits', data=hires_fourier_weighted.real, header=hires_hdr, overwrite=True)
    astropy.io.fits.writeto('/astro/dust_kg/cclark/Quest/lores_fourier_weighted.fits', data=lores_fourier_weighted.real, header=hires_hdr, overwrite=True)
    hires_weighted_img = np.fft.fftshift(np.real(np.fft.ifft2(np.fft.ifftshift(hires_fourier_weighted))))
    astropy.io.fits.writeto('/astro/dust_kg/cclark/Quest/hires_weighted_img.fits', data=hires_weighted_img, header=hires_hdr, overwrite=True)
    lores_weighted_img = np.fft.fftshift(np.real(np.fft.ifft2(np.fft.ifftshift(lores_fourier_weighted))))
    astropy.io.fits.writeto('/astro/dust_kg/cclark/Quest/lores_weighted_img.fits', data=lores_weighted_img, header=hires_hdr, overwrite=True)
    pdb.set_trace()

    lores_fourier_unnorm_weighted = lores_fourier * lores_weight
    lores_norm_ratio = lores_weighted_img / np.fft.fftshift(np.real(np.fft.ifft2(np.fft.ifftshift(lores_fourier_unnorm_weighted))))
    astropy.io.fits.writeto('/astro/dust_kg/cclark/Quest/lores_norm_ratio.fits', data=lores_norm_ratio, header=hires_hdr, overwrite=True)



# Function to create a 2D tapering fourier-space filter, transitioning between two defined angular scales, according to a Hann (ie, cosine bell) filter
# Inputs: Iterable giving angular scale of high- and low resolution cutoffs (in deg); WCS of the data to be combined
# Outputs: Array containing requested filter, with low-frequency passpand and high-frequency stopband
def FourierTaper(taper_cutoffs_deg, in_wcs):

    # Use provided WCS to construct array to hold the output filter
    if in_wcs.array_shape[0] != in_wcs.array_shape[1]:
        raise Exception('Input data to be combined are not square; this will not end well')
    out_filter = np.zeros([in_wcs.array_shape[1], in_wcs.array_shape[0]])
    out_i_centre = (0.5*out_filter.shape[0])-0.5
    out_j_centre = (0.5*out_filter.shape[1])-0.5

    # Convert cutoff scales to units of fourier-pixels
    pix_width_deg = np.abs( np.max( in_wcs.pixel_scale_matrix ) )
    cutoff_min_deg = min(taper_cutoffs_deg)
    cutoff_max_deg = max(taper_cutoffs_deg)
    cutoff_max_frac = (cutoff_max_deg / pix_width_deg) / out_filter.shape[0]
    cutoff_min_frac = (cutoff_min_deg / pix_width_deg) / out_filter.shape[0]
    cutoff_max_pix = 1.0 * (1.0 / cutoff_max_frac)
    cutoff_min_pix = 1.0 * (1.0 / cutoff_min_frac)

    # Use meshgrids to find distance of each pixel from centre of filter array
    i_linespace = np.linspace(0, out_filter.shape[0]-1, out_filter.shape[0])
    j_linespace = np.linspace(0, out_filter.shape[1]-1, out_filter.shape[1])
    i_grid, j_grid = np.meshgrid(i_linespace, j_linespace, indexing='ij')
    i_grid -= out_i_centre
    j_grid -= out_j_centre
    rad_grid = np.sqrt(i_grid**2.0 + j_grid**2.0)

    # Rejigger the distance grid to give distance past inner edge of taper region
    rad_grid -= cutoff_max_pix

    # Set central region (ie, low-resolution regime) to be entirely passband
    out_filter[np.where(rad_grid <= 0)] = 1.0

    # Construct well-sampled Hann filter to compute taper for transition region
    hann_filter = np.hanning(2000)[1000:]
    hann_pix = np.linspace(0, cutoff_min_pix-cutoff_max_pix, num=1000)
    hann_interp = scipy.interpolate.interp1d(hann_pix, hann_filter, bounds_error=False, fill_value=np.nan)

    # Identify pixels where Hann filter is to be applied, and compute taper
    hann_where = np.where((rad_grid > 0) & (rad_grid <= cutoff_min_pix))
    out_filter[hann_where] = hann_interp(rad_grid[hann_where])
    out_filter[np.where(np.isnan(out_filter))] = 0.0

    # Return final filter
    return out_filter



# Function to cross-calibrate two data sets' power over a range of angular scales, as a precursor to fourier combination
# Input: Fourier transform of low-resolution data; fourier transform of high-resolution data; Iterable giving angular scale of high- and low resolution cutoffs (in deg)
# Output: Correction factor to be applied to high-resolution data; uncertainty on the correction factor
def FourierCalibrate(lores_fourier, hires_fourier, taper_cutoffs_deg, pix_width_deg):

    # Find centre of fourier arrays (ie, the zeroth order fourier frequency)
    freq_i_centre = (0.5*hires_fourier.shape[0])-0.5
    freq_j_centre = (0.5*hires_fourier.shape[1])-0.5

    # Calculate grid of radial fourier frequency distance from zeroth order
    i_linespace = np.linspace(0, hires_fourier.shape[0]-1, hires_fourier.shape[0])
    j_linespace = np.linspace(0, hires_fourier.shape[1]-1, hires_fourier.shape[1])
    i_grid, j_grid = np.meshgrid(i_linespace, j_linespace, indexing='ij')
    i_grid -= freq_i_centre
    j_grid -= freq_j_centre
    rad_grid = np.sqrt(i_grid**2.0 + j_grid**2.0)

    # Convert cutoff scales from degrees to fourier frequencies
    cutoff_min_deg = min(taper_cutoffs_deg)
    cutoff_max_deg = max(taper_cutoffs_deg)
    cutoff_min_frac = (cutoff_min_deg / pix_width_deg) / hires_fourier.shape[0]
    cutoff_max_frac = (cutoff_max_deg / pix_width_deg) / hires_fourier.shape[0]
    cutoff_min_pix = 1.0 / cutoff_min_frac
    cutoff_max_pix = 1.0 / cutoff_max_frac

    # Slice out subsets of radial distance grid corresponding to cutoffs, and their overlap
    rad_grid_cutoff_min = rad_grid[np.where(rad_grid<cutoff_min_pix)]
    #rad_grid_cutoff_max = rad_grid[np.where(rad_grid>cutoff_max_pix)]
    rad_grid_overlap = rad_grid[np.where((rad_grid>cutoff_max_pix) & (rad_grid<cutoff_min_pix))]

    # Calculate power of each scale (square-rooted, so that it's power, not the power spectrum)
    power_lores = np.sqrt((lores_fourier.real)**2.0)
    power_hires = np.sqrt((hires_fourier.real)**2.0)
    power_lores_cutoff_min = power_lores[np.where(rad_grid<cutoff_min_pix)]

    # Slice out power at regiosn corresponding to cutoffs, and their otherlap
    power_lores_overlap = power_lores[np.where((rad_grid>cutoff_max_pix) & (rad_grid<cutoff_min_pix))]
    power_hires_overlap = power_hires[np.where((rad_grid>cutoff_max_pix) & (rad_grid<cutoff_min_pix))]

    # Caltulate correction factor (in log space, as distribution in power space is logarithmic)
    power_dex_overlap = np.log10(power_hires_overlap) - np.log10(power_lores_overlap)
    power_hires_corr_factor = 1.0 / 10.0**SigmaClip(power_dex_overlap, median=True, sigma_thresh=1.0)[1]

    # Calculate uncertainty on correction factor by bootstrapping
    power_hires_corr_bs = []
    for b in range(100):
        power_hires_corr_bs.append(SigmaClip(np.random.choice(power_dex_overlap, size=len(power_dex_overlap)), median=True, sigma_thresh=1.0)[1])
    power_hires_corr_factor_unc = np.std(1.0 / (10.0**np.array(power_hires_corr_bs)))

    #Return results
    return (power_hires_corr_factor, power_hires_corr_factor_unc)

    # Test plotting
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(rad_grid, power_hires, s=0.25, c='dodgerblue', alpha=0.85)
    ax.scatter(rad_grid_cutoff_min, power_lores_cutoff_min, s=0.25, c='orangered', alpha=0.85)
    #ax.scatter(rad_grid_overlap, power_lores_overlap, s=0.25, c='limegreen', alpha=0.85)
    power_comb = np.sqrt((comb_fourier.real)**2.0)
    ax.scatter(rad_grid, power_comb, s=0.25, c='fuchsia', alpha=0.85)
    ax.plot([cutoff_min_pix,cutoff_min_pix], [1E-50,1E50], ls=':', c='gray')
    ax.plot([cutoff_max_pix,cutoff_max_pix], [1E-50,1E50], ls=':', c='gray')
    ax.set_xlim([1,50])
    ax.set_ylim([1E4,1E7])
    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.savefig('/astro/dust_kg/cclark/Quest/fourier.png', dpi=300)



# Function to get the pixel width of some FITS data
# Input: Etiher a string to a FITS file, or an astropy.io.fits.Header object, or an astropy.wcs.WCS object
# Output: Pixel width in arcseconds
def PixWidthArcsec(in_data):

    # If input is string, read in header from file, and grab WCS
    if isinstance(in_data, str):
        in_hdr = astropy.io.fits.getheader(in_data)
        in_wcs = astropy.wcs.WCS(in_hdr)

    # If input is header, grab WCS
    if isinstance(in_data, astropy.io.fits.Header):
        in_hdr = in_data
        in_wcs = astropy.wcs.WCS(in_hdr)

    # If input is WCS object, note as such
    if isinstance(in_data, astropy.wcs.WCS):
        in_wcs = in_data

    # Calculate pixel width, and return
    pix_width_arcsec = 3600.0 * np.abs( np.max( in_wcs.pixel_scale_matrix ) )
    return pix_width_arcsec



# Define function to convert data from Msol/sqpc to Msol/pix
# Inputs: Numpy array (or just a float) of data in units of Msol/sqpc; the fits header for those data; distance to source in pc; (a boolean for if to calculate Msol/pix to Msol/sqpc instead)
# Outputs: Numpy array (or just a float) with data in units of Msol/pix
def MsolSqpcToMsolPix(img, hdr, dist, inverse=False):
    wcs = astropy.wcs.WCS(hdr)
    wcs_pix_matrix = wcs.pixel_scale_matrix
    pix_width_deg = np.mean(np.abs(wcs_pix_matrix[np.where(wcs_pix_matrix!=0)]))
    pix_width_pc = dist * 2.0 * np.tan(0.5 * np.deg2rad(pix_width_deg)) # As (O/2) = A tan(theta/2)
    pix_area_sqpc = pix_width_pc**2.0
    img = np.array(img)
    if not inverse:
        return img * pix_area_sqpc
    elif inverse:
        return img / pix_area_sqpc



# Define function to convert data from Msol/pix to Msol/sqpc (this just calls MsolSqpcToPix with the inverse keyword set)
# Inputs: Numpy array (or just a float) of data in units of Msol/sqpc; the fits header for those data; distance to source in pc
# Outputs: Numpy array (or just a float) with data in units of Msol/pix
def MsolPixToMsolSqpc(img, hdr, dist):
    return MsolSqpcToMsolPix(img, hdr, dist, inverse=True)



# Define function to convert MJy/sr surface brightness to Jy flux
# Inputs: Numpy array (or just a float) of data in units of MJy/sr; the fits header for those data; (a boolean for if to calculate Jy to MJy/sr instead)
# Outputs: Numpy array (or just a float) with data in units of Jy
def MJySrToJyPix(img, hdr, inverse=False):
    wcs = astropy.wcs.WCS(hdr)
    pix_scale_matrix = wcs.pixel_scale_matrix
    pix_width_deg = np.mean(np.abs(pix_scale_matrix[np.where(pix_scale_matrix!=0)]))
    pix_area_sqdeg = pix_width_deg**2.0
    pix_area_sr = pix_area_sqdeg * (np.pi / 180.0)**2.0
    conversion = pix_area_sr * 1E6
    if not inverse:
        return img * conversion
    elif inverse:
        return img / conversion



# Define function to convert MJy/sr surface brightness to Jy flux
# Inputs: Numpy array (or just a float) of data in units of MJy/sr; the fits header for those data; (a boolean for if to calculate Jy to MJy/sr instead)
# Outputs: Numpy array (or just a float) with data in units of Jy
def JyPixToMJySr(img, hdr):
    return MJySrToJyPix(img, hdr, inverse=True)
