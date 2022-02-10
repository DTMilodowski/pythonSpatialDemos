"""
pt2_basic_time_series.py
--------------------------------------------------------------------------------
WORKSHOP PART 2: INTRODUCTION TO BASIC TIME SERIES ANALYSIS
Part 2 of the demonstration provides an introduction to extracting time series
information from a set of raster datasets, representing multiple "views" of a
landscape over time.

Steps:
1) load and visualise a raster dataset to find a coordinates for an interesting
    target location to analyse later

2) extract a pixel value for this target coordinate

3) extract a pixel value for a target coordinate over multiple timesteps

4) visualise the time series in a simple plot for VV and VH

5) repeat for other pixels...

07/02/2022 - D. T. Milodowski
--------------------------------------------------------------------------------
"""

"""
# Import the necessary packages
"""
import os
import glob
import numpy as np                  # standard package for scientific computing
import xarray as xr                 # xarray geospatial package
import matplotlib.pyplot as plt     # plotting package
import seaborn as sns               # another useful plotting package
sns.set()                           # set some nice default plotting options

# Some useful functions for dealing with dates
def get_year(date):
    return date.astype('datetime64[Y]').astype(int) + 1970
def get_month(date):
    return date.astype('datetime64[M]').astype(int) % 12 + 1
def get_day(date):
    return (date - date.astype('datetime64[M]') + 1).astype(int)

"""
Part 2A: Initial visualisation
We are going to start by loading a raster dataset into python using xarray,
as before. We will load one of the temporally filtered datasets, because it
is a bit easier to see features of interest once the filtering stage has 
been applied
"""

# To open a raster dataset is easy enough to do. We need the name of the file in
# question, alongside it's full path
path2s1 = '/disk/scratch/local.2/Sentinel-1/' # this is the path to the sentinel 1 data
path2timesteps = '{}processed_to_dB/cawdor/processed_no_filter/'.format(path2s1)
path2filtered = '{}processed_to_dB/cawdor/processed_temporal_filter/'.format(path2s1)

# an example file, in this case, the VV polarised backscatter (median from 
# temporal stack between 2019-06-01 and 2019-08-01)
s1file = '{}/S1__IW__D_20190601_20190801_VV_gamma0-rtc_dB_temporal_median.tif'.format(path2filtered)

# open file and store data in an xarray called agb
ds = xr.open_rasterio(s1file)

# There was only one band in the geotiff, with the VV backscatter values. We
# can select this band very easily using the sel() function
s1 = ds.sel(band=1)

# convert nodatavalues to numpy-recognised nodata (np.nan)
s1.values[s1.values==s1.nodatavals[0]]=np.nan

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
Part 2B: For a single location, extract the pixel value
Look at the figure plotted in the previous step. Find an area you think might be
interesting and extract the pixel value associate with this
"""
target_x = 453841.0
target_y = 6377702.0

"""
Find the nearest pixel to your target point. This might be a little fiddly!
We use the np.argmin function to find the row and column locations in the
array
"""
nearest_column = np.argmin(np.abs(s1.x.values-target_x))
nearest_row = np.argmin(np.abs(s1.y.values-target_y))

# print the nearest value to screen...
print(s1[nearest_column,nearest_row])

# that wasn't so tricky after all!

"""
Part 2C: For a single location, plot a time series of VV polarised backscatter
We are going to load in datasets for the full time series. This will provide a
great opportunity to view a basic time series (i.e. repeated measurements of
the same location over time). This is the first step to understanding the
temporal patterns in the data, and how these may relate to the phenology of the
underlying vegetation.

Note that we need to pay close attention to the file paths. These are systematic
with date, so should be straighforward :)
"""
start_date = np.datetime64('2018-01-01')
end_date = np.datetime64('2022-01-01')

# start an empty list to host the available raster data within the time period
# we will fill this list in the following loop
scenes_vh = []
scenes_vv = []
dates = []

for date in np.arange(start_date,end_date):
    # this loops through every date between "start_date" and "end_date"
    # We need to find out if there is data for a particular date. There
    # are a many different ways to do this. In this case, we will use a
    # simple check to see if there is a directory for this date, in 
    # which case we will have data!
    day = get_day(date)
    month = get_month(date)
    year = get_year(date)
    path2date = '%s/%04i/%02i/%02i/' % (path2timesteps,year,month,day)
    if os.path.isdir(path2date): # this returns TRUE if there is a directory
        vv_files_for_this_date = glob.glob('%s/*_%04i%02i%02i*VV*.tif' % (path2date, year, month, day))
        vh_files_for_this_date = glob.glob('%s/*_%04i%02i%02i*VH*.tif' % (path2date, year, month, day))
        scenes_vv += vv_files_for_this_date # adding two lists together like this appends one to the other
        scenes_vh += vh_files_for_this_date # adding two lists together like this appends one to the other
        for ii in np.arange(0,len(vh_files_for_this_date)): # sometimes there may be two shots on a given day (two S1 satellites!)
            dates += [date]

# how many scenes do we have?
print("number of scenes = {}".format(len(dates)))

# Now loop through the scenes and extract the value for the target location
VHvalues = np.zeros(len(scenes_vh))*np.nan # create an empty array (nodata values)
VVvalues = np.zeros(len(scenes_vv))*np.nan
for ii,date in enumerate(dates):
    # open file and store data in an xarray called agb
    vhfile = scenes_vh[ii]
    vvfile = scenes_vv[ii]
    s1VH_iter = xr.open_rasterio(vhfile).sel(band=1)
    s1VV_iter = xr.open_rasterio(vvfile).sel(band=1)

    nearest_column = np.argmin(np.abs(s1.x.values-target_x))
    nearest_row = np.argmin(np.abs(s1.y.values-target_y))
    
    VHvalues[ii]=s1VH_iter.values[nearest_column,nearest_row]
    VVvalues[ii]=s1VV_iter.values[nearest_column,nearest_row]
    
# Great, now we have a list of dates, and the VH and VV backscatter amplitudes on
# those dates. Let's make a plot and see what they look like
# Note for colours, I've taken colour-blind friendly combinations
# (e.g. http://mkweb.bcgsc.ca/colorblind/palettes.mhtml#page-container)
fig2C,ax2C = plt.subplots(nrows=1, ncols=1, figsize=(12,5))
ax2C.plot(dates,VHvalues,marker='o',linestyle='None',color='#9F0162',label='VH')
ax2C.plot(dates,VVvalues,marker='^',linestyle='None',color='#0079FA',label='VV')
ax2C.set_xlabel('')
ax2C.set_ylabel('backscatter amplitude / dB')
ax2C.legend()
fig2C.tight_layout()
fig2C.show()
fig2C.savefig('S1_time_series_example.png')

"""
Part 2D: Now time for a short exercise. See if you can make a plot that
compares the backscatter time series of different land cover types. You
will need to play around with the colours and markers etc. to make it as
clear as possible. You could even try making multiple subplots in the same
figure.
"""

# you will need to write this code, but the code above gives you an easy headstart!
