# Import smorgasbord
import numpy as np
import matplotlib
from matplotlib import pyplot as pl, cm, colors



# Function to extract a colourmap from cmap object, from https://gist.github.com/denis-bz/8052855
def get_cmap( cmap, name=None, n=256 ):
    """ in: a name "Blues" "BuGn_r" ... of a builtin cmap (case-sensitive)
        or a filename, np.loadtxt() n x 3 or 4  ints 0..255 or floats 0..1
        or a cmap already
        or a numpy array.
        See http://wiki.scipy.org/Cookbook/Matplotlib/Show_colormaps
        or in IPython, pl.cm.<tab>
    """
    if isinstance( cmap, colors.Colormap ):
        return cmap
    if isinstance( cmap, basestring ):
        if cmap in cm.cmap_d:
            return pl.get_cmap( cmap )  # "Blues" ...
        A = np.loadtxt( cmap, delimiter=None )  # None: white space
        name = name or cmap.split("/")[-1] .split(".")[0]  # .../xx.csv -> xx
    else:
        A = cmap  # numpy array or array-like
    return array_cmap( A, name, n=n )



# Function to create a truncated version of an existing colourmap, from https://gist.github.com/denis-bz/8052855
def truncate_colormap( cmap, minval=0.0, maxval=1.0, n=256 ):
    """ mycolormap = truncate_colormap(
            cmap name or file or ndarray,
            minval=0.2, maxval=0.8 ): subset
            minval=1, maxval=0 )    : reverse
    by unutbu http://stackoverflow.com/questions/18926031/how-to-extract-a-subset-of-a-colormap-as-a-new-colormap-in-matplotlib
    """
    cmap = get_cmap( cmap )
    name = "%s-trunc-%.2g-%.2g" % (cmap.name, minval, maxval)
    return colors.LinearSegmentedColormap.from_list(
    name, cmap( np.linspace( minval, maxval, n )))



# Function to apply an arbitrary function to a colourmap
def cmap_map(function, cmap):
    """ Applies function (which should operate on vectors of shape 3: [r, g, b]), on colormap cmap.
        This routine will break any discontinuous points in a colormap.
    by http://scipy-cookbook.readthedocs.io/items/Matplotlib_ColormapTransformations.html
    """
    cdict = cmap._segmentdata
    step_dict = {}

    # Firt get the list of points where the segments start or end
    for key in ('red', 'green', 'blue'):
        step_dict[key] = list(map(lambda x: x[0], cdict[key]))
    step_list = sum(step_dict.values(), [])
    step_list = np.array(list(set(step_list)))

    # Then compute the LUT, and apply the function to the LUT
    reduced_cmap = lambda step : np.array(cmap(step)[0:3])
    old_LUT = np.array(list(map(reduced_cmap, step_list)))
    new_LUT = np.array(list(map(function, old_LUT)))

    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i, key in enumerate(['red','green','blue']):
        this_cdict = {}
        for j, step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j, i]
            elif new_LUT[j,i] != old_LUT[j, i]:
                this_cdict[step] = new_LUT[j, i]
        colorvector = list(map(lambda x: x + (x[1], ), this_cdict.items()))
        colorvector.sort()
        cdict[key] = colorvector
    return matplotlib.colors.LinearSegmentedColormap('colormap',cdict,1024)