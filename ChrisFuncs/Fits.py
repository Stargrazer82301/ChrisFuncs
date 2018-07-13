from __future__ import print_function
# Import smorgasbord
import sys
import os
import pdb
current_module = sys.modules[__name__]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import astropy.io.fits
import astropy.wcs
import astropy.convolution
import aplpy
import reproject
from ChrisFuncs import SigmaClip, Nanless



# Function to create a cutout of a fits file - NOW JUST A WRAPPER OF AN ASTROPY FUNCTION
# Input: Input fits, cutout central ra (deg), cutout central dec (deg), cutout radius (arcsec), pixel width (arcsec), fits image extension, boolean of whether to reproject, boolean stating if an output variable is desired, output fits pathname if required
# Output: HDU of new file
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
# Input: Input fits pathname, margin to place around array, fits extension of interest, boolean stating if margin is in arcseconds, no pixelsboolean stating if an output variable is desired, output fits pathname
# Output: HDU of new file
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
# Input: Central right ascension (deg), central declination (deg), image width (deg), pixel size (arcsec)
# Output: FITS header
def FitsHeader(ra, dec, map_width_deg, pix_width_arcsec):

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

    # Return header
    return header



# Define function create a three-colour PNG image from some FITS files
# Input: Central RA for output image, central Dec of output image, radius of output image in arcseconds, list of paths (or list of lists when giving order-of-preference) to red green and blue files, output directory, (pmin threshold or list of 3 threshold values, pmax threshold or list of 3 threshold values, stretch type or list of stretch types)
# Output: None
def FitsRGB(ra, dec, rad_arcsec, in_paths, out_dir, pmin=False, pmax=False, stretch='log'):

    # Prepare storage list for paths, then loop over bands
    path_list = [''] * 3
    for b in range(len(path_list)):

        # If this band gives a single path for a specific file, record it
        if not hasattr(in_paths[b], '__iter__'):
            path_list[b] = in_paths[b]

        # Else if this band gives a list of paths (ie, an order of preference), find the first file that exists and record it
        else:
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
        if pmin == False:
            vmin_phot = astropy.io.fits.getdata(path_list[k])
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
