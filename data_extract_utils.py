"""Data extraction utilities."""

import pandas as pd
import re
from datetime import datetime
from datetime import timedelta


def get_sample_data(date_str, site_data, ts_offset=1, ts_length=365, ts_freq=1):
    """Retrieves a timeseries for a sample from the full site data.
    
    Parameters
    ----------
    date_str : str
        The sampling date in YYYY-MM-DD format.
    site_data : DataFrame
        The full data extract for a site. Columns are the channels or
        bands of the source data. The dataframe should have a datetime
        index with 1 row per day.
    ts_offset : int, optional
        The number of days before the sampling date the time series
        should end. The default is 1 - the time series will end the day
        before the sampling date.
    ts_length : int, optional
        The length of the timeseries. The number of steps in the
        timeseries (not the number of days spanned by the timeseries).
        The default is 365.
    ts_freq : int, optional
        The frequency (in days) of the time series data. The default is
        1 - a daily time series.

    Returns
    -------
    np.array or None
        The flattened timeseries, in date then band order. ``None`` if
        the timeseries cannot be extracted from ``site_data``.
    """
    sample_date = datetime.strptime(date_str, '%Y-%m-%d')
    end_date = sample_date - timedelta(days=ts_offset)
    date_range = pd.date_range(end=end_date, periods=ts_length, freq=timedelta(days=ts_freq))
    try:
        return site_data.loc[date_range].values.flatten()
    except:
        print(f'Invalid date: {date_str} outside valid date range')
        return None


def sort_key(key):
    """Returns a numeric representation of a site or sample key
    
    Generates an integer key from a site or sample key. Assumes the key
    is a character followed by numbers between 1 and 999 separated by
    underscores. The leading character is ignored and the numeric parts
    processed from left to right.
    
    Example:
      - key: C13_1_4; generated key: 13001004
      - key: C6_1_13; generated key: 6001013
      - key: C6_1_5; generated key: 6001005
     
    These will be ordered as C6_1_5, C6_1_13, and C13_1_4 when sorted
    by the ``sortKey()`` value, a more natural ordering than the
    original alphanumeric sorting.
    
    Parameters
    ----------
    key : str
        Either a site (e.g. C6_1) or sample key (e.g. C6_1_13).

    Returns
    -------
    genKey : int
        A numeric representation of the key that can be used to sort
        keys into logical order.
    """
    key_parts = key.split("_")
    genKey = int(re.findall("\d+", key_parts[0])[0])
    for key_part in key_parts[1:]:
        genKey = genKey * 1000 + int(key_part)
    return genKey