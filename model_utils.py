"""Model building and evaluation utitlities"""

import random
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import interpn
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold


def _stratify_sites(data, split_col, stratify, min_sites):
    print(f'Split by {split_col}, stratify by {stratify}')
    # Assign a unique landcover value to each site
    sites = data.groupby([split_col], as_index=False).agg(
        stratify=(stratify, lambda x: pd.Series.mode(x)[0]))
    lc = sites.groupby(['stratify'], as_index=False).agg(counts=(split_col, 'count'))
    # Group landcover classes with fewer than min_sites sites into one group
    lc = lc.stratify[lc.counts < min_sites]
    sites.loc[sites.stratify.isin(lc), 'stratify'] = 0
    # Return the stratified sites
    return sites
    
    
def reshape_data(X, nchannels):
    """Reshape an array
    
    Reshape a two dimensional array of rows with flattened timeseries
    and channels to three dimensions.

    Parameters
    ----------
    X : np.array
        A two dimensional array. The first dimension is the row, the
        second the features ordered by date then band/channel - e.g.
        d1.b1 d1.b2 d1.b3 d2.b1 d2.b2 d2.b3 ...
    nchannels : int
        Number of channels.

    Returns
    -------
    np.array
        A 3 dimensional array where the first dimension are rows, the
        second time and the third the band or channel.
    """
    return X.reshape(X.shape[0], int(X.shape[1] / nchannels), nchannels)

	
def random_split(num_samples, split_sizes, val_data=False, random_seed=None):
    """Splits samples randomly
    
    Randomly assigns numbers in range(0, num_samples) to the test,
    training and validation set in accordance with the required split
    sizes. Returns an index for each set that indicates which samples
    are in each set.

    Parameters
    ----------
    num_samples : int
        The number of samples.
    split_sizes : tuple of length 1 or 2
        The size of the test and optionally the validation set. If the
        tuples are integers, they are interpreted as the number of
        samples. If they are floats, they are interpreted as the
        proportion of samples.
    val_data : bool, optional
        Indicates if validation data is required. The default is False.
    random_seed : int, optional
        The seed for the random number generator. The default is None.

    Returns
    -------
    train_index : array of bool
        An array of length num_samples. True if the sample should be
        included in the training set, false otherwise.
    val_index : array of bool
        An array of length num_samples. True if the sample should be
        included in the validation set, false otherwise.
    test_index : array of bool
        An array of length num_samples. True if the sample should be
        included in the test set, false otherwise.
    """
    if random_seed is not None:
        random.seed(random_seed)
    sample_list = list(range(num_samples))
    random.shuffle(sample_list)

    # Generate the test index
    test_size = split_sizes[0]
    if type(test_size) is int:
        break1 = test_size
    else:
        break1 = int(np.floor(num_samples * test_size))
    test_index = np.zeros(num_samples, dtype=bool)
    test_index[sample_list[:break1]] = True

    # Generate the validation index if used
    if val_data:
        val_size = split_sizes[1]
        if type(val_size) is int:
            break2 = break1 + val_size
        else:
            break2 = break1 + int(np.floor(num_samples * (val_size)))
        val_index = np.zeros(num_samples, dtype=bool)
        val_index[sample_list[break1:break2]] = True
    else:
        break2 = break1
        val_index = None

    # Generate the training index
    train_index = np.zeros(num_samples, dtype=bool)
    train_index[sample_list[break2:]] = True
    return train_index, val_index, test_index


def split_by_site(data, split_sizes, split_col='Site', stratify=None,
                val_data=False, min_sites=6, random_seed=None):
    """Splits sample sites randomly.
    
    Randomly assigns sites to the test, training and validation sets in
    accordance with the required split sizes. If stratified splitting
    is requested, sites are grouped according to the stratified column
    value and split so each group is evenly represented in each set.
    After the sites have been split, samples are assigned to the
    appropriate set based on their site.
    
    Parameters
    ----------
    data : DataFrame
        Data Frame of samples. Must have a column labelled with
        ``split_col``, and one labelled with ``stratify`` if specified.
    split_sizes : tuple of length 1 or 2
        The size of the test and optionally the validation set. If the
        tuples are ints, they are interpreted as the number of sites.
        If floats, they are interpreted as the proportion of sites.
    split_col : str, optional
        The name of the column containing the sample site. The default
        is 'Site'.
    stratify : str or None, optional
        The name of the column to use for stratified splitting. None
        for no stratified splits. The default is None.
    val_data : bool, optional
        Indicates if validation data is required. The default is False.
    min_sites : int, optional
        The minimum number of sites in a stratified group. Groups with
        fewer sites are combined into a single group. The default is 6.
    random_seed : int, optional
        The seed for the random number generator. The default is None.

    Returns
    -------
    train_index : array of bool
        An array of length equal to rows in data. True if the sample
        should be included in the training set, false otherwise.
    val_index : array of bool
        An array of length equal to rows in data. True if the sample
        should be included in the validation set, false otherwise.
    test_index : array of bool
        An array of length equal to rows in data. True if the sample
        should be included in the test set, false otherwise.
    """
    if stratify:
        # print(f'Split by {split_col}, stratify by {stratify}')
        # # Assign a unique landcover value to each site
        # sites = data.groupby([split_col], as_index=False).agg(
        #     stratify=(stratify, lambda x: pd.Series.mode(x)[0]))
        # lc = sites.groupby(['stratify'], as_index=False).agg(counts=(split_col, 'count'))
        # # Group landcover classes with fewer than min_sites sites into one group
        # lc = lc.stratify[lc.counts < min_sites]
        # sites.loc[sites.stratify.isin(lc), 'stratify'] = 0
        # # Set the stratify labels and splitter class
        sites =_stratify_sites(data, split_col, stratify, min_sites)
        y = sites.stratify
        Splitter = StratifiedShuffleSplit
        
    else:
        print(f'Split by {split_col}, no stratify')
        sites = data.groupby([split_col], as_index=False).agg(counts=(split_col, 'count'))
        y = sites[split_col]
        Splitter = ShuffleSplit

    temp = data.reset_index()[split_col]
    
    # Generate the test index
    test_size = split_sizes[0]
    if type(test_size) is float:
        test_size = int(np.floor(y.size * test_size))
    sss = Splitter(n_splits=1, test_size=test_size, random_state=random_seed)
    trainSites, testSites = next(sss.split(y, y))
    test_index = np.zeros(temp.size, dtype=bool)
    test_index[temp.isin(sites.loc[testSites, split_col])] = True

    # Generate the validation index if used
    if val_data:
        val_size = split_sizes[1]
        if type(val_size) is float:
            val_size = int(np.floor(y.size * val_size))
        y1 = y.iloc[trainSites]
        rs = None if random_seed is None else random_seed * 2
        sss = Splitter(n_splits=1, test_size=val_size, random_state=rs)
        ti, vi = next(sss.split(y1, y1))
        valSites = trainSites[vi]
        trainSites = trainSites[ti]
        val_index = np.zeros(temp.size, dtype=bool)
        val_index[temp.isin(sites.loc[valSites, split_col])] = True
    else:
        val_index = None

    # Generate the training index
    train_index = np.zeros(temp.size, dtype=bool)
    train_index[temp.isin(sites.loc[trainSites, split_col])] = True
    return train_index, val_index, test_index	


def kfold_by_site(data, nFolds, split_col='Site', stratify=None,
                  val_size=0, min_sites=6, random_seed=None):
    """Creates k-fold splits
    
    Creates ``nFold`` splits of the "sites" by randomly assigning sites
    to a fold. Then, in turn, each fold is assigned to the test set and
    remaining folds assigned to the training set. (Validation data is
    randomly selected from training data, if required). If stratified
    splitting is requested, sites are grouped by the stratified column
    value and split so each group is evenly represented in each fold.
    After the sites have been split, samples are assigned to the
    appropriate set based on their site.
    
    Note that if split_col is set to the name of the data index (or
    "index" if unnamed), the folds will be random (stratified) splits
    of the samples, rather than the sites.

    Parameters
    ----------
    data : DataFrame
        Data Frame of samples. Must have a column labelled with
        ``split_col``, and one labelled with ``stratify`` if specified.
    nFolds : int
        The number of folds (``K``) to create.
    split_col : str, optional
        The name of the column containing the sample site. The default
        is 'Site'.
    stratify : str or None, optional
        The name of the column to use for stratified splitting. None
        for no stratified splits. The default is None.
    val_size : int or float, optional
        The number (int) or proportion (float) of samples to assign to
        the validation set in each fold. The validation samples are
        randomly selected from the training set. If 0, no validation
        sets are created. The default is 0.
    min_sites : int, optional
        The minimum number of sites in a stratified group. Groups with
        fewer sites are combined into a single group. The default is 6.
    random_seed : int, optional
        The seed for the random number generator. The default is None.

    Returns
    -------
    train_index : array of bool
        An array of length equal to rows in data. True if the sample
        should be included in the training set, false otherwise.
    val_index : array of bool
        An array of length equal to rows in data. True if the sample
        should be included in the validation set, false otherwise.
    test_index : array of bool
        An array of length equal to rows in data. True if the sample
        should be included in the test set, false otherwise.
    """
    if stratify:
        # print(f'Split by {split_col}, stratify by {stratify}')
        # sites = data.groupby([split_col], as_index=False).agg(
        #     stratify=(stratify, lambda x: pd.Series.mode(x)[0]))
        # lc = sites.groupby(['stratify'], as_index=False).agg(counts=(split_col, 'count'))
        # lc = lc.stratify[lc.counts < min_sites]
        # sites.loc[sites.stratify.isin(lc), 'stratify'] = 0
        # y = sites.stratify
        sites =_stratify_sites(data, split_col, stratify, min_sites)
        y = sites.stratify
        Splitter = StratifiedKFold
        ValSplitter = StratifiedShuffleSplit
        
    else:
        print(f'Split by {split_col}, no stratify')
        sites = data.groupby([split_col], as_index=False).agg(counts=(split_col, 'count'))
        y = sites[split_col]
        Splitter = KFold
        ValSplitter = ShuffleSplit

    temp = data.reset_index()[split_col]
    
    # Generate the test indexes
    sss = Splitter(n_splits=nFolds, shuffle=True, random_state=random_seed)
    folds = [{'train': i1, 'test': i2} for i1, i2 in sss.split(y, y)]
    test_index = [np.zeros(temp.size, dtype=bool) for _ in range(nFolds)]
    for n, fold in enumerate(folds):
        test_index[n][temp.isin(sites.loc[fold['test'], split_col])] = True

    # Generate the validation indexes if used
    if val_size:
        if type(val_size) is float:
            val_size = int(np.floor(y.size * val_size))
        rs = None if random_seed is None else random_seed * 2
        val_index = [np.zeros(temp.size, dtype=bool) for _ in range(nFolds)]
        for n, fold in enumerate(folds):
            y1 = y.iloc[fold['train']]
            sss = ValSplitter(n_splits=1, test_size=val_size, random_state=rs)
            ti, vi = next(sss.split(y1, y1))
            valSites = fold['train'][vi]
            fold['train'] = fold['train'][ti]
            val_index[n][temp.isin(sites.loc[valSites, split_col])] = True
    else:
        val_index = [None] * nFolds

    # Generate the training indexes
    train_index = [np.zeros(temp.size, dtype=bool) for n in range(nFolds)]
    for n, fold in enumerate(folds):
        train_index[n][temp.isin(sites.loc[fold['train'], split_col])] = True
    return train_index, val_index, test_index


def split_data(data, train_index, test_index, val_index=None):
    """Splits data into train, test and validation sets
    
    Splits data into train, test and (optionally) validation sets using
    the index parameters.

    Parameters
    ----------
    data : array
        DESCRIPTION.
    train_index : array of bool
        Array values should be True for entries in the training set,
        False otherwise.
    test_index : array of bool
        Array values should be True for entries in the test set, False
        otherwise.
    val_index : array of bool, optional
        Array values should be True for entries in the validation set,
        False otherwise. If val_index is None, no validation set is
        created. The default is None.

    Returns
    -------
    train_data : array
        The training data.
    test_data : array
        The test data.
    val_data : array or None
        The validation data (if any).
    """
    #split the sample set into training, validation and testing sets
    test_data = data[test_index]
    val_data = data[val_index] if val_index is not None else None
    train_data = data[train_index]
    return train_data, test_data, val_data


def normalise(data, method='minMax', percentiles=0, range=[0, 10000]):
    """Normalises the data.
    
    Parameters
    ----------
    data : array
        Array of the data to be normalised.
    method : str, optional
        Normalisation method to use. Currently implemented methods are:
          - minMax: Default. Normalise so the lower percentile is 0 and
            the upper percentile is 1.
          - range: Normal across a set range, so lower range is 0 and
            upper range is 1.
    percentiles : int or list-like, optional
        The percentile to use with the minMax method, If a single int
        value, this is treated as the lower percentile and the upper
        percentile is 100 - the value. The default is 0.
    range : list-like, optional
        Two values corresponding to the lower and upper bounds of the
        normalisation range. The default is [0, 10000].

    Returns
    -------
    array
        The normalised data.
    """
    if method == 'minMax':
        if type(percentiles) is int:
            percentiles = [percentiles, 100-percentiles]
        bounds = np.percentile(data, percentiles, axis=(0, 1))
        return (data - bounds[0]) / (bounds[1] - bounds[0])
    if method == 'range':
        return (data - range[0]) / (range[1] - range[0])
    else: # method not implemented
        return data


def calc_statistics(x, y):
    """Calculate model evaluation statistics
    
    Calculates the following statistics:
      - Bias: The difference between the mean prediction and the mean
        label
      - RMSE: The root mean squared error of the predictions
      - ubRMSE (Unbiased RMSE): The RMSE obtained if each prediction is
        adjusted by the Bias
      - R: The correlation coefficient
      - R2 (R-squared): The percent of variance explained

    Parameters
    ----------
    x : array
        The sample labels.
    y : TYPE
        The sample predictions.

    Returns
    -------
    dict
        The calculated statistics.

    """
    bias = np.mean(y) - np.mean(x)
    bias = np.round(bias, 2)
    rmse = np.sqrt(np.mean(np.square(y - x)))
    rmse = np.round(rmse, 2)
    r = np.corrcoef(x, y)
    r2 = r[0, 1] ** 2
    r = np.round(r[0, 1], 2)
    r2 = np.round(r2, 2)
    ubrmse = np.sqrt(np.mean(np.square(y - x - bias)))
    ubrmse = np.round(ubrmse, 2)
    return {'Bias': bias, 'R':r, 'R2':r2, 'RMSE': rmse, 'ubRMSE': ubrmse}


def plot_results(fig_name, y, yhat, metrics, ax=None,
                 bins=30, lower=0, upper=300, vmin=0, vmax=25):
    """Create a plot of labels versus predictions
    
    Create a plot of labels versus predictions and save to a png file.
    The figure (or axes) is not displayed, but returned so the caller
    can display it if required.
    
    A 2-D histogram of the labels and predictions is generated, and
    interpolated to give a smoothed result. The plot is coloured
    according to the number of samples in each bin.

    Parameters
    ----------
    fig_name : str
        A name for the figure. Used to generate a title and filename
        for the figure.
    y : array
        The sample labels.
    yhat : array
        The sample predictions.
    metrics : dict
        The evaluation statistics.
    ax : MatPlotLib axes, optional
        An axes for the figure. Used if the plot is to be added as a
        sub-plot to a figure. If this is set, a figure object is not
        created and an axes object is returned. The default is None.
    bins : int, optional
        The number of bins (in each dimension) to use when binning the
        data. The default is 30.
    lower : int, optional
        The lower bound to use for binning. The default is 0.
    upper : int, optional
        The upper bound to use for binning. The default is 300.
    vmin : int, optional
        The minimum bin size. Only used for the colour scale, so does
        not cause an error if the number of values in a bin is less
        than this limit. The default is 0.
    vmax : int, optional
        The maximum bin size. Only used for the colour scale, so does
        not cause an error if the number of values in a bin exceeds
        this limit. The default is 25.

    Returns
    -------
    MatPlotLib axes or figure
        Returns the axes if one specified, else the figure.

    """
    # Create the sub-plot, if none
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = None
    # Plot the data
    plot_data = pd.DataFrame({'y': np.round(y), 'yhat':np.round(yhat)})
    plot_data = plot_data[(plot_data.y <= upper) & (plot_data.yhat <= upper) ]
    bin_width = (upper-lower)//bins
    data, x_bins, y_bins = np.histogram2d(plot_data.y,
                                          plot_data.yhat,
                                          bins=list(range(0, upper+1, bin_width)),
                                          density=False)
    z = interpn((x_bins[:-1]+(bin_width/2), y_bins[:-1]+(bin_width/2)),
                data/(bin_width**2),
                np.vstack([plot_data.y, plot_data.yhat]).T,
                method = "splinef2d",
                bounds_error=False)
    z[np.where(np.isnan(z))] = 0.0
    idx = z.argsort()
    x1, y1, z1 = plot_data.y[idx], plot_data.yhat[idx], z[idx]
    plot_data = pd.DataFrame({'x': x1, 'y': y1, 'z':z1}).drop_duplicates()
    ax.scatter(plot_data.x, plot_data.y, c=plot_data.z, s=5, vmin=vmin, vmax=vmax)
    # Add the statistics
    text = f"Bias: {metrics['Bias']}%\nR: {metrics['R']}\n" + \
           f"RMSE: {metrics['RMSE']}%\nubRMSE: {metrics['ubRMSE']}%"
    xloc = lower + 10
    yloc = upper - 60
    ax.text(xloc, yloc, text, size=10)
    # Plot the regression line and equation
    slope, intercept, _, _, _ = stats.linregress(y, yhat)
    linex = np.asarray(list(range(lower+25, upper-50)))
    ax.plot(linex, linex * slope + intercept, '--', color='red')
    text = f"y={slope:.2f}x+{intercept:.2f}"
    xloc = upper // 2
    yloc = lower + 50
    ax.text(xloc, yloc, text, size=10, color='firebrick')
    ax.plot([lower, upper], [lower, upper], '--', color=(0.5, 0.5, 0.5))
    # Add the labels, title etc
    ax.axis([lower, upper, lower, upper])
    ax.set_ylabel('Estimated LFMC[%]', fontsize=12)
    ax.set_xlabel('Measured LFMC[%]', fontsize=12)
    ax.grid(True)
    ax.set_title(fig_name)
    if fig is None:
        return ax
    else:
        return fig


def plot_all_results(all_results, all_stats, model_dir):
    """Plot results for all models
    
    Parameters
    ----------
    all_results : Data Frame
        A data frame contains the results to plot. The first column
        should be named ``y`` and contain the labels. The other columns
        should be named with the model name.
    all_stats : Data Frame
        A data frame containing the evaluation statistics for each
        model. The index should be the model names and the columns the
        statistic names.
    model_dir : str
        The name of the directory where the plots should be saved.

    Returns
    -------
    None.

    """
    for y in all_results.columns[1:]:
        fig = plot_results(f'{y} Results', all_results.y, all_results[y], all_stats.loc[y])
        fig.savefig(os.path.join(model_dir, y + '.png'), dpi=300)
        plt.close(fig)
