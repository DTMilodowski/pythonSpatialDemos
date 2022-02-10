"""
pt1_intro_to_python_spatial.py
--------------------------------------------------------------------------------
WORKSHOP PART 1: INTRODUCTION TO SPATIAL ANALYSIS IN PYTHON
Part 1 of the python demonstration gives a very quick introduction to the open
source programming language python, and utilises the geospatial library xarray 
(http://xarray.pydata.org/en/stable/), which provides a lot of useful 
functionality for dealing with raster datasets.

04/02/2022 - D. T. Milodowski
--------------------------------------------------------------------------------
"""

"""
This is a comment block, bookended by three double quotation marks
Comment blocks and comments are ignored by python, but are useful for explaining
what the code is doing
"""

# this is a comment, with a hash at the start. The line after the hash is
# ignored

"""
# Import the necessary packages
"""
import numpy as np                  # standard package for scientific computing
import xarray as xr                 # xarray geospatial package
import matplotlib.pyplot as plt     # plotting package
import seaborn as sns               # another useful plotting package
sns.set()                           # set some nice default plotting options

"""
Part 1A: Loading a dataset using xarray
We are going to start by loading a raster dataset into python using xarray,
before exploring how to interact with this xarray object. For later scripts, a
lot of the processing will be tucked away inside other functions, but it is
useful to know what we are dealing with.
"""

# To open a raster dataset is easy enough to do. We need the name of the file in
# question, alongside it's full path
path2s1 = '/disk/scratch/local.2/Sentinel-1/' # this is the path to the sentinel 1 data
path2filtered = '{}processed_to_dB/cawdor/processed_temporal_filter/'.format(path2s1)
print(path2filtered) # print path2data to screen

# an example file, in this case, the VV polarised backscatter (median from 
# temporal stack between 2019-06-01 and 2019-08-01)
s1file = '{}/S1__IW__D_20190601_20190801_VV_gamma0-rtc_dB_temporal_median.tif'.format(path2filtered)
print(s1file) # print filename to screen

# open file and store data in an xarray called agb
ds = xr.open_rasterio(s1file)
print(type(ds))

# Let's explore the xarray structure a little
# The key properties for a data array are:
# 1) the values numpy array containing the gridded observations
print(type(ds.values))
# 2) the dimensions of the array
print(ds.dims)
# 3) the coordinates of the data
print(ds.coords)
# 4) the meta data e.g. coordinate system
print(ds.attrs)
# 5) the nodata value
print(ds.nodatavals)

# There was only one band in the geotiff, with the VV backscatter values. We
# can select this band very easily using the sel() function
s1 = ds.sel(band=1)

# convert nodatavalues to numpy-recognised nodata (np.nan)
s1.values[s1.values==s1.nodatavals[0]]=np.nan

# This new xarray now only has the dimensions y and x
print(ds.coords)
print(s1.coords)

# We'll explore xarray interactions further in due course, but for now, it is
# worthwhile simply plotting a map of the data. With xarray, this is really easy.

"""
Basic plotting with xarray
"""
fig1, axis1 = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
s1.plot(ax=axis1, vmax=0, cmap='viridis', add_colorbar=True,
                    extend='max', cbar_kwargs={'label': 'VV backscatter / dB$',
                                            'orientation':'vertical'})
axis1.set_aspect("equal")
axis1.set_title('Study area, VV backscatter, median 2019-06 to 2019-08')
fig1.show()

"""
Subsets of xarrays
# OK, now we can try to manipulate this a bit. It is quite easy to select a
# subset of an xarray - see http://xarray.pydata.org/en/stable/indexing.html
# We are not going to go through every example here, but a simple spatial
# subset can be taken if we know the coordinate bounds
"""

# first of all, let's check the coordinate ranges of the raster, before choosing
# a subset...
print('min x: ', np.min(s1.x.values))
print('max x: ', np.max(s1.x.values))
print('min y: ', np.min(s1.y.values))
print('max y: ', np.max(s1.y.values))

# Choose a subset based on min and max coordinates
min_x = 445005.00; max_x = 459995.0
min_y = 6370005.0; max_y = 6380495.0

# use the sel function to extract the spatial subset
s1_subset = s1.sel(x=slice(min_x,max_x),y = slice(max_y,min_y))
# Note that since our y coordinates are listed in decreasing order, we take the
# slice from max to min, which might seem initially counter-intuitive.

# Great, now let's plot that up and see what it looks like!
fig2, axis2 = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
s1_subset.plot(ax=axis2, vmax=0, cmap='viridis', add_colorbar=True,
                    extend='max', cbar_kwargs={'label': 'VV backscatter / dB$',
                                            'orientation':'vertical'})
axis2.set_aspect("equal")
axis2.set_title('Spatial subset, VV backscatter, median 2019-06 to 2019-08')
fig2.show()


# of course, we could have achieved the same plot without creating a new object:
fig3, axis3 = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
s1.sel(x=slice(min_x,max_x),y = slice(max_y,min_y)).plot(ax=axis3,
                    vmax=0, cmap='viridis', add_colorbar=True, extend='max',
                    cbar_kwargs={'label': 'VV backscatter / dB$',
                                'orientation':'vertical'})
axis3.set_aspect("equal")
axis3.set_title('Spatial subset, VV backscatter, median 2019-06 to 2019-08')
fig3.show()
# Perhaps it is clearer to split the subset and plotting stage

# Sometimes visualisation of radar data can be made a bit more interesting by
# creating "false colour composites". Basically this revolves around creating
# an RGB image that can be used to 


# First, let's create a little helper function
# rescale - useful for generating RGB normalisations for multi-band visualisation
def rescale(array,llim=None,ulim=None):
    if llim is None:
        llim=np.nanmin(array)
    if ulim is None:
        ulim=np.nanmax(array)
    return   np.interp(array, (llim, ulim), (0, 1))

# Next, let's load in three rasters for the area
s1afile = '{}/S1__IW__D_20190401_20190601_VV_gamma0-rtc_dB_temporal_median.tif'.format(path2filtered)
s1bfile = '{}/S1__IW__D_20190601_20190801_VV_gamma0-rtc_dB_temporal_median.tif'.format(path2filtered)
s1cfile = '{}/S1__IW__D_20190801_20191001_VV_gamma0-rtc_dB_temporal_median.tif'.format(path2filtered)

s1a = xr.open_rasterio(s1afile)
s1b = xr.open_rasterio(s1bfile)
s1c = xr.open_rasterio(s1cfile)

# Set the nodatavalues appropriately
s1a.values[s1a.values==s1a.nodatavals[0]]=np.nan
s1b.values[s1b.values==s1b.nodatavals[0]]=np.nan
s1c.values[s1c.values==s1c.nodatavals[0]]=np.nan

# Now use the 'concat' function to combine these into a single 
# first let's check the shape of one of our rasters:
print(s1a.shape)
# concatenate three rasters along the dimension 'band'
s1combined = xr.concat((s1a,s1b,s1c),dim='band')
# check shape of new object
print(s1combined.shape)

# for RGB plots, it's important to scale the values so that they lie between 0 and 1
# One common visualisation trick is to scale so that the dynamic range spans +/- two
# standard deviations of the mean
mean = np.nanmean(s1combined.values)
std = np.nanstd(s1combined.values)
# note that I have used the nanmean and nanstd functions as they handle 'nodata' pixels

# Use these to set the limits for the "rescale" function
llim = mean-2*std
ulim = mean+2*std
s1rgb = s1combined.copy(deep=True)
s1rgb.values = rescale(s1combined.values, llim=llim, ulim=ulim)

# now plot the rgb and see what it looks like!
fig4, axis4 = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
s1rgb.plot.imshow(ax=axis4)
axis4.set_aspect("equal")
axis4.set_title('Spatial subset, VV backscatter false colour temporal composite')
fig4.show()
fig4.savefig('S1_multitemporal_composite.png')