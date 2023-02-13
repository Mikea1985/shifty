# -*- coding: utf-8 -*-
# shifty/shifty/cleaner.py

'''
   Classes / methods for reading raw image fits files and cleaning them
   Provides methods to
   - Read in images (using imagehandler's DataEnsemble)
   - align the images via reprojection (slow!)
   - background normalisation
   - template subtraction
   - masking
'''

# -----------------------------------------------------------------------------
# Third party imports
# -----------------------------------------------------------------------------
import os
import sys
# import gc
from datetime import datetime
import copy
import tracemalloc
import numpy as np

from astropy.io import fits
# from astropy import wcs
from astropy.nddata import CCDData
from ccdproc import wcs_project  # , Combiner


# -----------------------------------------------------------------------------
# Any local imports
# -----------------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(
                os.path.realpath(__file__))))
from shifty.imagehandler import DataEnsemble


# -----------------------------------------------------------------------------
# Various class definitions for *data import * in shifty
# -----------------------------------------------------------------------------

class DataCleaner():
    '''
    (1) Loads a list of fits-files, via DataEnsemble
    (2) Cleans data in various ways
    (3) Returns a cleaned DataEnsemble object and/or saves to fits files.

    methods:
    --------
    reproject_data
    subtract_background_level
    template_subtract
    _subtract_provided_template
    _subtract_overall
    _subtract_local
    _subtract_local_but_not_too_local

    main public method:
    -------------------
    reproject_data
    subtract_background_level
    template_subtract

    '''
    # Turning off some stupid syntax-checker warnings:
    # pylint: disable=too-many-instance-attributes
    # Why on earth should an object only have 7 attributes?
    # That seems dumb. Turning off this warning.
    # pylint: disable=too-few-public-methods
    # I get why an object should have at least two public methods in order to
    # not be pointless, but it's an annoying warning during dev. Turning off.
    # pylint: disable=attribute-defined-outside-init
    # Why should a method not be allowed to redefine an attribute?
    # It doesn't make any sense. Then what's the point of methods?
    # pylint: disable=too-many-locals
    # OK, this one I understand (methods should not be too long),
    # but it's currently just annoying.

    def __init__(self, filename=None, extno=0, verbose=False, **kwargs):
        '''
        inputs:
        -------
        filename       - list of str  - list of filepaths to valid fits-files
        extno          - int OR       - Extension number to use for all images
                         list of int  - list of extension to use for each image
        verbose        - bool         - Print extra stuff if True
        EXPTIME        - str OR float - Exposure time in seconds
                                      - (keyword or value)
        EXPUNIT        - str OR float - Units on exposure time
                                      - ('s','m','h','d' or float seconds/unit)
        MAGZERO        - str OR float - Zeropoint magnitude
                                      - (keyword or value)
        MJD_START      - str OR float - MJD at start of exposure
                                      - (keyword or value)
        GAIN           - str OR float - Gain value (keyword or value)
        FILTER         - str          - Filter name (keyword or '-name')
        NAXIS1         - str OR int   - Number of pixels along axis 1
        NAXIS2         - str OR int   - Number of pixels along axis 2
        INSTRUMENT     - str          - Instrument name (keyword)

        A float/integer value can be defined for most keywords,
        rather than a keyword name; this will use that value
        rather than searching for the keyword in the headers.
        INSTRUMENT and FILTER obviously can't be floats/integers,
        so use a leading '-' to specify a value
        rather than a keyword name to use.
        Not yet supported: list of a keyword values/names for each image, which
        hopefully should only ever be useful if attempting to stack images
        from different instruments; so low priority.
        '''
        # Set some values
        self.filename = filename
        self.extno = extno
        self.verbose = verbose
        if filename is not None:
            self.cleaned_data = DataEnsemble(filename, extno, verbose, **kwargs)
        else:
            self.cleaned_data = DataEnsemble()
        self.aligned = False
        self.reprojected = False
        self.bglevel_subtracted = False
        self.template_subtracted = False
        # Do some awesome stuff!!!

    def reproject_data_old(self, InputEnsemble, target=0, padmean=False):
        '''
        Reprojects each layer of InputEnsemble.data to be aligned, using their WCS.
        By default aligns everything else with the first image,
        but this can be changed by setting target.

        inputs:
        -------
        InputEnsemble - a DataEnsemble object
        target        - int  - Index of the target image to align relative to
        padmean       - bool - Whether to pad with the mean value (True) or
                               NaN values (False, default).

        outputs:
        --------
        InputEnsemble, having been modified directly
        (should take less memory than making a copy to modify)
        '''
        # Offsets
        offsetsl = []
        for i, w in enumerate(InputEnsemble.WCS):
            offsetsl.append(InputEnsemble.WCS[target].all_world2pix(*w.all_pix2world(0, 0, 0), 0))
        offsets = np.array(offsetsl).round().astype(int)[:, ::-1]
        # Make sure offsets are all positive (by subtracting smallest value)
        offsets[:, 0] -= offsets[:, 0].min()
        offsets[:, 1] -= offsets[:, 1].min()
        # Find max offset, to know how much to pad
        ymax = offsets[:, 0].max()
        xmax = offsets[:, 1].max()

        padded = []
        for i, dat in enumerate(InputEnsemble.data):
            pad_value = np.nanmean(dat) if padmean else np.nan  # Mean or NaN
            pad_size = ((offsets[i, 0], ymax - offsets[i, 0]),  # Size of pad
                        (offsets[i, 1], xmax - offsets[i, 1]))  # on 4 sides
            # Align the array by adding a padding around the edge
            paddedi = np.pad(dat, pad_size, constant_values=pad_value)
            padded.append(paddedi)
            # Update WCS, both in .wcs and .header
            InputEnsemble.WCS[i].wcs.crpix += (offsets[i, 1], offsets[i, 0])

        reprojected = []
        for i, dat in enumerate(InputEnsemble.data):
            print(f"Reprojecting image {i}", end='\r')
            # Do the reprojection
            reprojecti = wcs_project(CCDData(padded[i], wcs=InputEnsemble.WCS[i],
                                     unit='adu'), InputEnsemble.WCS[target])
            # Append to list
            reprojected.append(reprojecti.data)
            # Update WCS, both in InputEnsemble.wcs and InputEnsemble.header
            InputEnsemble.WCS[i] = InputEnsemble.WCS[target]
            del InputEnsemble.header[i]['CD?_?']  # required for update to work
            # Required to display correctly in ds9:
            InputEnsemble.header[i].remove('DATASEC', ignore_missing=True)
            new_wcs_header = InputEnsemble.WCS[i].to_header(relax=True)
            for key in ['TIMESYS', 'TIMEUNIT', 'MJDREF', 'DATE-OBS',
                        'MJD-OBS', 'TSTART', 'DATE-END', 'MJD-END',
                        'TSTOP', 'TELAPSE', 'TIMEDEL', 'TIMEPIXR']:
                new_wcs_header.remove(key, ignore_missing=True)
            del new_wcs_header['???MJD???']
            del new_wcs_header['???TIME???']
            del new_wcs_header['???DATE???']
            InputEnsemble.header[i].update(new_wcs_header)
            # Add a comment to the header about the reprojection
            now = str(datetime.today())[:19]
            InputEnsemble.header[i]['COMMENT'] = (f'Data was reprojected to WCS '
                                                  f'of file {target} at {now}')
        print("\nDone")
        InputEnsemble.data = np.array(reprojected)
        # ??? Having a separate array like this means we're taking up twice
        # the memory for the data. I wonder whether we can refactor this
        # to edit the raw data array directly. ???

    def reproject_data_old2(self, InputEnsemble, target=0, padmean=False):
        '''
        Reprojects each layer of InputEnsemble.data to be aligned, using their WCS.
        By default aligns everything else with the first image,
        but this can be changed by setting target.

        inputs:
        -------
        InputEnsemble - a DataEnsemble object
        target        - int  - Index of the target image to align relative to
        padmean       - bool - Whether to pad with the mean value (True) or
                               NaN values (False, default).

        outputs:
        --------
        InputEnsemble, having been modified directly
        (should take less memory than making a copy to modify)

        This was an attempt to improve reproject_old to use less memory.
        This does use less memory, but it only works if the array size is
        does not change, ie. the padding is 0 pixels... So no good.
        '''
        # Offsets
        offsetsl = []
        for i, w in enumerate(InputEnsemble.WCS):
            offsetsl.append(InputEnsemble.WCS[target].all_world2pix(*w.all_pix2world(0, 0, 0), 0))
        offsets = np.array(offsetsl).round().astype(int)[:, ::-1]
        # Make sure offsets are all positive (by subtracting smallest value)
        offsets[:, 0] -= offsets[:, 0].min()
        offsets[:, 1] -= offsets[:, 1].min()
        # Find max offset, to know how much to pad
        ymax = offsets[:, 0].max()
        xmax = offsets[:, 1].max()

        padded = []
        for i, dat in enumerate(InputEnsemble.data):
            pad_value = np.nanmean(dat) if padmean else np.nan  # Mean or NaN
            pad_size = ((offsets[i, 0], ymax - offsets[i, 0]),  # Size of pad
                        (offsets[i, 1], xmax - offsets[i, 1]))  # on 4 sides
            # Align the array by adding a padding around the edge
            paddedi = np.pad(dat, pad_size, constant_values=pad_value)
            padded.append(paddedi)
            # Update WCS, both in .wcs and .header
            InputEnsemble.WCS[i].wcs.crpix += (offsets[i, 1], offsets[i, 0])

        for i, dat in enumerate(InputEnsemble.data):
            print(f"Reprojecting image {i}", end='\r')
            # Do the reprojection
            reprojecti = wcs_project(CCDData(padded[i], wcs=InputEnsemble.WCS[i],
                                     unit='adu'), InputEnsemble.WCS[target])
            # Append to list
            InputEnsemble.data[i] = reprojecti.data
            # Oh right, that doesn't work if the array changes size...
            # Update WCS, both in InputEnsemble.wcs and InputEnsemble.header
            InputEnsemble.WCS[i] = InputEnsemble.WCS[target]
            del InputEnsemble.header[i]['CD?_?']  # required for update to work
            # Required to display correctly in ds9:
            InputEnsemble.header[i].remove('DATASEC', ignore_missing=True)
            new_wcs_header = InputEnsemble.WCS[i].to_header(relax=True)
            for key in ['TIMESYS', 'TIMEUNIT', 'MJDREF', 'DATE-OBS',
                        'MJD-OBS', 'TSTART', 'DATE-END', 'MJD-END',
                        'TSTOP', 'TELAPSE', 'TIMEDEL', 'TIMEPIXR']:
                new_wcs_header.remove(key, ignore_missing=True)
            del new_wcs_header['???MJD???']
            del new_wcs_header['???TIME???']
            del new_wcs_header['???DATE???']
            InputEnsemble.header[i].update(new_wcs_header)
            # Add a comment to the header about the reprojection
            now = str(datetime.today())[:19]
            InputEnsemble.header[i]['COMMENT'] = (f'Data was reprojected to WCS '
                                                  f'of file {target} at {now}')
        print("\nDone")

    def reproject_data(self, InputEnsemble, target=0, padmean=False):
        '''
        Reprojects each layer of InputEnsemble.data to be aligned, using their WCS.
        By default aligns everything else with the first image,
        but this can be changed by setting target.

        inputs:
        -------
        InputEnsemble - a DataEnsemble object
        target        - int  - Index of the target image to align relative to
        padmean       - bool - Whether to pad with the mean value (True) or
                               NaN values (False, default).

        outputs:
        --------
        InputEnsemble, having been modified directly
        (should take less memory than making a copy to modify)
        '''
        # Do rough pixel alignment and padding.
        self.rough_align(InputEnsemble, target, padmean)

        for i, wcsi in enumerate(InputEnsemble.WCS):
            print(f"Reprojecting image {i}", end='\r')
            # Do the reprojection
            InputEnsemble.data[i] = wcs_project(CCDData(InputEnsemble.data[i],
                                                        wcs=wcsi, unit='adu'),
                                                InputEnsemble.WCS[target]).data
            # Update WCS, both in InputEnsemble.wcs and InputEnsemble.header
            InputEnsemble.WCS[i] = InputEnsemble.WCS[target]
            del InputEnsemble.header[i]['CD?_?']  # required for update to work
            new_wcs_header = InputEnsemble.WCS[i].to_header(relax=True)
            for key in ['TIMESYS', 'TIMEUNIT', 'MJDREF', 'DATE-OBS',
                        'MJD-OBS', 'TSTART', 'DATE-END', 'MJD-END',
                        'TSTOP', 'TELAPSE', 'TIMEDEL', 'TIMEPIXR']:
                new_wcs_header.remove(key, ignore_missing=True)
            del new_wcs_header['???MJD???']
            del new_wcs_header['???TIME???']
            del new_wcs_header['???DATE???']
            InputEnsemble.header[i].update(new_wcs_header)
            # Add a comment to the header about the reprojection
            now = str(datetime.today())[:19]
            InputEnsemble.header[i]['COMMENT'] = (f'Data was reprojected to WCS '
                                                  f'of file {target} at {now}')
        print("\nDone")

    def rough_align(self, InputEnsemble, target=0, padmean=False):
        '''
        Aligns images to integer pixel accuracy.

        inputs:
        -------
        InputEnsemble - a DataEnsemble object
        target        - int  - Index of the target image to align relative to
        padmean       - bool - Whether to pad with the mean value (True) or
                               NaN values (False, default).

        outputs:
        --------
        InputEnsemble, having been modified directly
        (should take less memory than making a copy to modify)
        '''
        tracemalloc.start()
        # Offsets
        offsetsl = []
        for i, w in enumerate(InputEnsemble.WCS):
            offsetsl.append(InputEnsemble.WCS[target].all_world2pix(*w.all_pix2world(0, 0, 0), 0))
        offsets = np.array(offsetsl).round().astype(int)[:, ::-1]
        offsets -= offsets.min(0)
        xymax = np.max(offsets, 0)

        padded = []
        for i in np.arange(len(InputEnsemble.data)):
            print(f"Aligning and padding image {i}", end='\r')
            pad_value = (np.nanmean(InputEnsemble.data[i]) if padmean
                         else np.nan)  # Mean or NaN
            # Size of pad on 4 sides
            pad_size = ((offsets[i, 0], xymax[0] - offsets[i, 0]),
                        (offsets[i, 1], xymax[1] - offsets[i, 1]))
            # Align the array by adding a padding around the edge
            paddedi = np.pad(InputEnsemble.data[i], pad_size,
                             constant_values=pad_value)
            padded.append(paddedi)
            # Update WCS to account for padding, both in .wcs and .header
            InputEnsemble.WCS[i].wcs.crpix += (offsets[i, 1], offsets[i, 0])
            # Update header
            del InputEnsemble.header[i]['CD?_?']  # required for update to work
            # Required to display correctly in ds9:
            InputEnsemble.header[i].remove('DATASEC', ignore_missing=True)
            # Make new header:
            new_wcs_header = InputEnsemble.WCS[i].to_header(relax=True)
            # Don't change things that aren't WCP related.
            for key in ['TIMESYS', 'TIMEUNIT', 'MJDREF', 'DATE-OBS',
                        'MJD-OBS', 'TSTART', 'DATE-END', 'MJD-END',
                        'TSTOP', 'TELAPSE', 'TIMEDEL', 'TIMEPIXR']:
                new_wcs_header.remove(key, ignore_missing=True)
            del new_wcs_header['???MJD???']
            del new_wcs_header['???TIME???']
            del new_wcs_header['???DATE???']
            InputEnsemble.header[i].update(new_wcs_header)
            # Add a comment to the header about the reprojection
            now = str(datetime.today())[:19]
            InputEnsemble.header[i]['COMMENT'] = (f'Data was alligned to '
                                                  f'integer pixel at {now}')
        del InputEnsemble.data
        InputEnsemble.data = np.array(padded)
        del padded
        print("\nDone")

    def subtract_background_level(self, InputEnsemble, usemean=False):
        '''
        Subtract the background level (effectively making the background 0)

        inputs:
        -------
        InputEnsemble - a DataEnsemble object

        outputs:
        --------
        InputEnsemble, having been modified directly
        (should take less memory than making a copy to modify)
        '''
        for i, dat in enumerate(InputEnsemble.data):
            print(f"Subtracting background level in image {i}", end='\r')
            background_value = (np.nanmean(dat) if usemean
                                else np.nanmedian(dat))
            InputEnsemble.data[i] -= background_value
            # Add a comment to the header about the subtraction
            now = str(datetime.today())[:19]
            InputEnsemble.header[i]['COMMENT'] = (f'Background level '
                                                  f'subtracted at {now}')
        print("\nDone")

    def subtract_background_level_better(self, InputEnsemble, usemean=False):
        '''
        Subtract the background level (effectively making the background 0)

        inputs:
        -------
        InputEnsemble - a DataEnsemble object

        outputs:
        --------
        InputEnsemble, having been modified directly
        (should take less memory than making a copy to modify)
        '''
        for i, dat in enumerate(InputEnsemble.data):
            print(f"Subtracting background level in image {i}", end='\r')
            background_value = (np.nanmean(dat) if usemean
                                else np.nanmedian(dat))
            std_value = np.nanstd(dat)
            idx_old = (dat > -np.inf)
            idx = (np.abs(dat - background_value) < 5 * std_value)
            j = 0
            #print(j, background_value, std_value, np.shape(idx), np.shape(dat))
            while not np.all(idx_old == idx) & (j < 25):
                background_value = (np.nanmean(dat[idx]) if usemean
                                    else np.nanmedian(dat[idx]))
                std_value = np.nanstd(dat[idx])
                idx_old = idx
                idx = (np.abs(dat - background_value) < 5 * std_value)
                j += 1
                #print(j, background_value, std_value)
                
            InputEnsemble.data[i] -= background_value
            # Add a comment to the header about the subtraction
            now = str(datetime.today())[:19]
            InputEnsemble.header[i]['COMMENT'] = (f'Background level '
                                                  f'subtracted at {now}')
        print("\nDone")

    def _subtract_overall(self, InputEnsemble, usemean=False):
        '''
        Create a template from all of the observations
        and subtract it from every image.
        This is the fastest, but least good, template subtraction.

        inputs:
        -------
        InputEnsemble - a DataEnsemble object
        usemean       - bool - Whether to use mean (True)
                               or median (False, default)

        outputs:
        --------
        InputEnsemble, having been modified directly
        (should take less memory than making a copy to modify)
        '''
        print("Creating a universal template from all images.")
        universal_template = (np.nanmean(InputEnsemble.data, 0) if usemean
                              else np.nanmedian(InputEnsemble.data, 0))
        print("Subtracting template")
        InputEnsemble.data -= universal_template
        for i in np.arange(len(InputEnsemble.data)):
            # Add a comment to the header about the subtraction
            now = str(datetime.today())[:19]
            InputEnsemble.header[i]['COMMENT'] = (f'Template '
                                                  f'subtracted at {now}')
        print("\nDone")

    def _subtract_local_average(self, InputEnsemble, usemean=False, nobs=100):
        '''
        Create a 'local' (temporally) template from the closest nobs
        observations on each side of each image, and subtract it.

        inputs:
        -------
        InputEnsemble - a DataEnsemble object
        usemean       - bool - Whether to use mean (True)
                               or median (False, default)
        nobs          - int  - number of observations before and after

        outputs:
        --------
        InputEnsemble, having been modified directly
        (should take less memory than making a copy to modify)
        '''
        original_data = copy.deepcopy(InputEnsemble.data)
        for i in np.arange(len(InputEnsemble.data)):
            print(f"Subtracting template from image {i}", end='\r')
            i_min = np.max([0, i - nobs])
            i_max = np.min([i + nobs, len(original_data)])
            InputEnsemble.data[i] -= (np.nanmean(original_data[i_min:i_max], 0)
                                      if usemean
                                      else np.nanmedian(original_data[i_min:i_max], 0))
            # Add a comment to the header about the subtraction
            now = str(datetime.today())[:19]
            InputEnsemble.header[i]['COMMENT'] = (f'Template '
                                                  f'subtracted at {now}')
        print("\nDone")

    def _subtract_donut(self, InputEnsemble, usemean=False, nouter=100, ninner=25):
        '''
        Create a 'local' (temporally) template from the closest nobs
        observations on each side of each image, and subtract it.

        inputs:
        -------
        InputEnsemble - a DataEnsemble object
        usemean       - bool - Whether to use mean (True)
                               or median (False, default)
        nouter          - int  - number of observations before and after
        ninner          - int  - number of observations before and after

        outputs:
        --------
        InputEnsemble, having been modified directly
        (should take less memory than making a copy to modify)
        '''
        original_data = copy.deepcopy(InputEnsemble.data)
        nobs = len(original_data)
        for i in np.arange(len(InputEnsemble.data)):
            print(f"Subtracting template from image {i}", end='\r')
            i_min = np.max([0, i - nouter])
            i_mid1 = np.max([0, i - ninner])
            i_mid2 = np.min([i + ninner, nobs])
            i_max = np.min([i + nouter, nobs])
            use_data = (list(original_data[i_min:i_mid1])
                        + list(original_data[i_mid2:i_max]))
            donut_template = (np.mean(use_data, 0) if usemean
                              else np.median(use_data, 0))
            InputEnsemble.data[i] -= donut_template
            # Add a comment to the header about the subtraction
            now = str(datetime.today())[:19]
            InputEnsemble.header[i]['COMMENT'] = (f'Template '
                                                  f'subtracted at {now}')
        print("\nDone")

    def save_cleaned(self, filename='clean'):
        '''Save the shifted images to fits files.'''
        save_fits(self.cleaned_data, filename, self.verbose)


# -------------------------------------------------------------------------
# These functions really don't need to be methods, and therefore aren't.
# No need to over-complicate things.
# -------------------------------------------------------------------------

def save_fits(DataEnsembleObject, filename='data.fits', verbose=True):
    '''
    Save some data to [a] fits file[s].

    input:
    ------
    DataEnsembleObject - DataEnsemble Object - contains data and header.
    filename           - str                 - name or shared start of filename

    output:
    -------
    one or more fits file.
    If multiple, they are numbered *_000.fits, *_001.fits, etc.
    '''
    if len(DataEnsembleObject.data.shape) == 2:
        filename = filename if '.fits' in filename else filename + '.fits'
        hdu = fits.PrimaryHDU(data=DataEnsembleObject.data,
                              header=DataEnsembleObject.header)
        if verbose:
            print(f'Saving to file {filename}')
        hdu.writeto(filename, overwrite=True)
    elif len(DataEnsembleObject.data.shape) == 3:
        nfiles = len(str(len(DataEnsembleObject.data)))
        print(nfiles)
        for i, data in enumerate(DataEnsembleObject.data):
            filenamei = filename.replace('.fits', '')
            filenamei += f'_{i:0{nfiles}.0f}.fits'
            hdu = fits.PrimaryHDU(data=data,
                                  header=DataEnsembleObject.header[i])
            if verbose:
                print(f'Saving to file {filenamei}', end='\r')
            hdu.writeto(filenamei, overwrite=True)
    else:
        raise ValueError('The input is not a valid DataEnsemble object.')
    print('\nDone!')


# END
