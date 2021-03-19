"""Google Earth Engine Time Series Extractor class"""

import ee
import numpy as np
import os
import pandas as pd


class GeeTimeseriesExtractor:
    """Google Earth Engine Time Series Extractor class
    
    Parameters
    ----------
    product : str
        The Google Earth Engine product name.
    bands : list
        A list of the band names required.
    start_date : str
        The start date for the time series.
    end_date : str
        The end date for the time series.
    freq : str, optional
        The frequency of time series entries. The default is '1D' for
        daily entries.
    gap_fill : str or bool, optional
        DESCRIPTION. The default is True.
    max_gap : int, optional
        The maximum size of gap that will be filled. Filling is done in
        both directions, so the maximum actual size filled is
        ``2 * max_gap``. If None, there if no limit on the gap size.
        The default is None.
    dir_name : str
        The directory where the extracted files will be stored. The
        default is ''.

    Returns
    -------
    None.
    """
    
    def __init__(self, product, bands, start_date, end_date,
                 freq='1D', gap_fill=True, max_gap=None, dir_name=''):
        self.product = product
        self.bands = bands
        self.collection = ee.ImageCollection(product).select(bands)
        self.set_date_range(start_date, end_date, freq, gap_fill, max_gap)
        self.save_band_info()
        self.set_output_dir(dir_name)
        self.set_default_proj_dir()
        
    def set_date_range(self, start_date, end_date, freq='1D', gap_fill=True, max_gap=None):
        """Sets the date range for the extracts
        
        Parameters
        ----------
        start_date : str
            The start date for the time series.
        end_date : str
            The end date for the time series.
        freq : str, optional
            The frequency of time series entries. The default is '1D'
            for daily entries.
        gap_fill : str or bool, optional
            DESCRIPTION. The default is True.
        max_gap : int, optional
            The maximum size of gap that will be filled. Filling is
            done in both directions, so the maximum actual size filled
            is ``2 * max_gap``. If None, there if no limit on the gap
            size. The default is None.

        Returns
        -------
        None.
        """
        self.start_date = start_date
        self.end_date = end_date
        if gap_fill:
            date_range = pd.date_range(start_date, end_date, freq=freq, closed="left")
            self.days = pd.Series(date_range, name="id")
            self.fill = pd.DataFrame(date_range, columns=["id"])
            self.gap_fill = gap_fill
            self.max_gap = max_gap
        else:
            self.gap_fill = False

    def filtered_collection(self):
        """Filters the GEE collection by date
        
        Returns
        -------
        ee.Collection
            The filtered GEE collection.
        """
        return self.collection.filterDate(self.start_date, self.end_date)
        
    def set_output_dir(self, dir_name):
        """Sets the extract directory
        
        Parameters
        ----------
        dir_name : str
            Thr name of the extract directory.

        Returns
        -------
        None.
        """
        if dir_name is None or dir_name == '':
            self.dir_name = ''
        else:
            self.dir_name = os.path.join(dir_name, '')
        
    def save_band_info(self):
        """Saves bands information from the GEE collection
        
        Returns
        -------
        None.
        """
        image_info = self.filtered_collection().first().getInfo()
        self.band_info = image_info['bands']
        data_type = self.band_info[0]['data_type']  # Assumes all bands have the same data type
        self.data_type = 'Int64' if data_type['precision'] == 'int' else np.float
        
    def _int_data(self):
        return self.data_type == 'Int64'
        
    def set_default_proj_dir(self):
        """Sets the default extract projection and scale
        
        Sets the extract projection and scale to the collection's
        native projection and scale.

        Returns
        -------
        None.

        """
        band = self.band_info[0]  # Assumes all bands have the same proj/scale
        self.projection = band['crs']
        self.scale = abs(band['crs_transform'][0]) # crs_tranform is [+/-scale, 0, x, 0 , +/-scale, y]
        
    def set_proj_scale(self, proj, scale):
        """Sets the required projection and scale for the extract
        
        Parameters
        ----------
        proj : str
            The required projection.
        scale : float
            The required scale.

        Returns
        -------
        None.
        """
        self.projection = proj
        self.scale = scale
        
    def download_data(self, location):
        """Download the GEE data for a location
        
        Downloads the GEE data for a location, converts it to a data
        frame and gap fills if required.
        
        Parameters
        ----------
        location : Pandas series
            A Pandas series containing the location ``Longitude`` and
            ``Latitude``.

        Returns
        -------
        bands_df : Pandas data frame
            A data frame. The columns are the bands and the index is
            the dates.

        """
        geometry = ee.Geometry.Point([location.Longitude, location.Latitude])
        data = self.filtered_collection().getRegion(
            geometry, self.scale, self.projection).getInfo()
        data_df = pd.DataFrame(data[1:], columns=data[0])
        self.last_longitude = data_df.longitude[0]
        self.last_latitude = data_df.latitude[0]
        bands_index = pd.DatetimeIndex(pd.to_datetime(data_df.time, unit='ms').dt.date)
        bands_df = data_df[self.bands].set_index(bands_index).rename_axis(index='id').sort_index()
        if self.gap_fill:
            bands_df = bands_df.merge(self.days, how="right",
                                      left_index=True, right_on='id').set_index('id')
            method = 'linear' if self.gap_fill == True else self.gap_fill
            bands_df = bands_df[self.bands].interpolate(axis=0, method=method, limit=self.max_gap,
                                                        limit_direction="both")
        if self._int_data():
            bands_df = bands_df.round().astype(self.data_type)
        return bands_df
            
    def get_and_save_data(self, location):
        """Get and save the GEE data for a location
        
        Checks if data has already been extracted for the location (a
        file called ``<Site>.csv`` already exists). If the file exists,
        it is read into a data frame and returned. If it doesn't exist,
        the data for the location will be downloaded, saved to
        ``<Site>.csv`` and the data frame returned.

        Parameters
        ----------
        location : Pandas series
            A Pandas series containing the ``Site`` name and location
            ``Longitude`` and ``Latitude``.

        Returns
        -------
        point_df : Pands data frame
            A data frame. The columns are the bands and the index is
            the dates.

        """
        file_name = f'{self.dir_name}{location.Site}.csv'
        try:  # If we already have the location data, read it
            dtypes = {band: self.data_type for band in self.bands}
            point_df = pd.read_csv(file_name, index_col="id", parse_dates=True, dtype=dtypes)
        except:  # Otherwise extract it from GEE
            print(f'Extracting data for {location.Site} (lat: {location.Latitude} ' + \
                  f'long: {location.Longitude})')
            point_df = self.download_data(location)
            point_df.to_csv(file_name)
        return point_df