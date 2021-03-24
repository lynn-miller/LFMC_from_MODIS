"""Functions to build and evaluate LFMC models"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from lfmc_model import LfmcTempCnn, ModelList
from model_parameters import ModelParams
from model_utils import calc_statistics, plot_results, plot_all_results
from model_utils import random_split, split_by_site, kfold_by_site, split_data


def train_test_split(model_params, samples, X, y):
    """Splits data into test and training sets

    Splits data into test, training and optionally validation sets. The
    model parameters determine how to split the data.
    
    validationSet:
        Indicates if a validation set should be created
    
    splitMethod:
      - random: samples are split randomly
      - bySite: sites are split into test and training sites, then all
        samples from the site are allocated to the respective set
      - byYear: samples are allocated to the test and training sets
        based on the sampling year
    
    splitSizes:
        The proportion or number of samples to allocate to the test and
        validation sets. Used for random and bySite splits.
    
    siteColumn:
        The column in the samples containing the site. Used for bySite
        splits.
    
    yearColumn:
        The column in the samples containing the sampling year. Used
        for byYear splits.
    
    splitStratify:
        Indicates the column to use for stratified splitting, if this
        is needed. Stratified splitting groups the sites by value and
        ensures each split contains a roughly equal proportion of each
        group. If None, stratified splittig is not used. Used for
        bySite splits.
    
    splitYear:
        The year to use for byYear splitting. Samples later than this
        year go into the test set. Samples earlier than this year go
        into the training set. Samples for this year go into the
        validation set if used, otherwise go into the training set.
        Used for byYear splits.
                
    randomSeed:
        The seed for the random number generator. Used for random and
        bySite splits.
    
    Parameters
    ----------
    model_params : ModelParams
        The dictionary of model parameters, used as described above.
    samples : DataFrame
        A data frame containing the samples.
    X : dict
        The predictor data. The keys are the name of each source. Each
        value is an array of the predictors for that source. The first
        dimension is assumed to correspond to the rows of the samples
        data frame in length and order.
    y : Array
        The labels. A 1-dimensional array containing the label for each
        sample.

    Returns
    -------
    data : dict
        The split data with an item for each dataset. Keys are ``test``,
        ``val(idation)``, or ``train(ing)``. Values are dictionaries
        with two entries - ``X`` and ``y``. The ``X`` values are again
        dictionaries, with entries for each source. The lowest level
        values are numpy arrays containg the relevant data. ``val``
        entries are included even when no validation set is required.
        In this case, all lowest level values are ``None``.
    """
    if model_params['splitMethod'] == 'random':
        train_index, val_index, test_index = random_split(
                num_samples=samples.shape[0],
                split_sizes=model_params['splitSizes'],
                val_data=model_params['validationSet'],
                random_seed=model_params['randomSeed'])
    elif model_params['splitMethod'] == 'bySite':
        train_index, val_index, test_index = split_by_site(
                data=samples,
                split_sizes=model_params['splitSizes'],
                split_col=model_params['siteColumn'],
                stratify=model_params['splitStratify'],
                val_data=model_params['validationSet'],
                random_seed=model_params['randomSeed'])
    else:
        sample_years = np.array(samples[model_params['yearColumn']])
        test_index = sample_years > model_params['splitYear']
        if model_params['validationSet']:
            train_index = sample_years < model_params['splitYear']
            val_index = sample_years == model_params['splitYear']
        else:
            train_index = sample_years <= model_params['splitYear']
            val_index = None

    y_train, y_test, y_val = split_data(y, train_index, test_index, val_index)
    data = {'train': {'X': {}, 'y': y_train},
            'val':   {'X': {}, 'y': y_val},
            'test':  {'X': {}, 'y': y_test}}
    for source, xdata in X.items():
        train, test, val = split_data(xdata, train_index, test_index, val_index)
        data['train']['X'][source] = train
        data['val']['X'][source] = val
        data['test']['X'][source] = test
    return data


def kfold_split(model_params, samples, X, y):
    """Splits data into folds

    A generator that splits data into folds containing test, training
    and optionally validation sets. The samples are split into ``K``
    roughly equal parts (where ``K`` is the number of folds), then
    folds are formed by using each part in turn as the test set and the
    remaining parts as the training set. The validation set (if used)
    is randomly selected from the training set. The generator yields
    each fold in turn.

    The following model parameters determine how to split the data. 

    validationSet:
        Indicates if a validation set should be created.
    
    splitMethod:
      - random: samples are split randomly
      - bySite: sites are split into test and training sites, then all
        samples from the site are allocated to the respective set
      - NOTE: byYear cannot be used with k-fold splits
    
    splitFolds:
        The number of folds (``K``) required.
        
    splitSizes:
        The proportion or number of samples to allocate to validation
        sets. This must contain two values (consistent with standard
        splitting) but the first value is ignored.
    
    siteColumn:
        The column in the samples containing the site. Used for bySite
        splits.
    
    splitStratify:
        Indicates the column to use for stratified splitting, if this
        is needed. Stratified splitting groups the sites by value and
        ensures each split contains a roughly equal proportion of each
        group. If None, stratified splittig is not used. Used for
        bySite splits.
    
    randomSeed:
        The seed for the random number generator.
    
    Parameters
    ----------
    model_params : ModelParams
        The dictionary of model parameters, used as described above.
    samples : DataFrame
        A data frame containing the samples.
    X : dict
        The predictor data. The keys are the name of each source. Each
        value is an array of the predictors for that source. The first
        dimension is assumed to correspond to the rows of the samples
        data frame in length and order.
    y : Array
        The labels. A 1-dimensional array containing the label for each
        sample.

    Yields
    ------
    data : dict
        The split data with an item for each dataset. Keys are ``test``,
        ``val(idation)``, or ``train(ing)``. Values are dictionaries
        with two entries - ``X`` and ``y``. The ``X`` values are again
        dictionaries, with entries for each source. The lowest level
        values are numpy arrays containg the relevant data. ``val``
        entries are included even when no validation set is required.
        In this case, all lowest level values are ``None``.
    """
    if model_params['splitMethod'] == 'random':
        train_index, val_index, test_index = kfold_by_site(
                data=samples.reset_index(),
                nFolds=model_params['splitFolds'],
                split_col=samples.index.name or 'index',
                stratify=False,
                val_size=model_params['splitSizes'][1] if model_params['validationSet'] else 0,
                random_seed=model_params['randomSeed'])
    elif model_params['splitMethod'] == 'bySite':
        train_index, val_index, test_index = kfold_by_site(
                data=samples,
                nFolds=model_params['splitFolds'],
                split_col=model_params['siteColumn'],
                stratify=model_params['splitStratify'],
                val_size=model_params['splitSizes'][1] if model_params['validationSet'] else 0,
                random_seed=model_params['randomSeed'])
    else: # Combination of byYear and k-folds is not valid!
        raise Exception(f'Invalid Model Parameters - cannot use "{model_params["splitMethod"]}" '\
                        'with "splitFolds" > 1')

    for fold in range(model_params['splitFolds']):
        y_train, y_test, y_val = split_data(
            y, train_index[fold], test_index[fold], val_index[fold])
        data = {'train': {'X': {}, 'y': y_train},
                'val':   {'X': {}, 'y': y_val},
                'test':  {'X': {}, 'y': y_test}}
        for source, xdata in X.items():
            train, test, val = split_data(
                xdata, train_index[fold], test_index[fold], val_index[fold])
            data['train']['X'][source] = train
            data['val']['X'][source] = val
            data['test']['X'][source] = test
        yield data


def train_test_model(model_params, train, val, test):
    """Trains and tests a model.
    
    Builds, trains and evaluates an LFMC model. After training the
    model, several derived models are created from the fully-trained
    (base) model:
      - best - a model using the checkpoint with the best training/
        validation loss
      - merge10 - a model created by merging the last 10 checkpoints.
        The checkpoints are merged by averaging the corresponding
        weights from each model.
      - ensemble10 - an ensembled model of the last 10 checkpoints.
        This model averages the predictions made by each model in the
        ensemble to make the final prediction.
      - merge_best10 - similar to the merge10 model, but uses the 10
        checkpoints with the lowest training/validation losses.
          
    All 5 models are evaluated by calculating statistics based on the
    test predictions. Both the predictions and statistics are saved.
    
    To facilitate overfitting analysis, predictions and statistics
    using the training data are generated using the merge10 model.
    
    Parameters
    ----------
    model_params : ModelParams
        The model parameters.
    train : dict
        The training dataset.
    val : dict
        The valiadtion dataset.
    test : dict
        The test dataset.
        
    The train, val and test dictionaries are all the same format. They
    have two items, ``X`` and ``y`` for the predictors and labels
    respectively. ``X`` is also a dictionary with an item for each source.

    Returns
    -------
    model : LfmcTempCnn
        The built model and results.
    """
    #--------------------------
    # Build and train the model
    #--------------------------
    model_dir = os.path.join(model_params['modelDir'], '')
    model = LfmcTempCnn(model_params, inputs=train['X'])
    results = model.train(train['X'], train['y'], val['X'], val['y'])
    print(f"Training results: minLoss: {results['minLoss']}, runTime: {results['runTime']}")

    # Plot the training history
    model.plot_train_hist()
    if model_params['validationSet']:
        model.plotTrainHist(metric='loss')

    # Create the derived models
    best_num = model.best_model()               # Extract the best checkpoint model
    model.merge_models('merge10', 10)           # Merge the last 10 checkpoints into a new model
    model.ensemble_models('ensemble10', 10)     # Create an ensembled model of last 10 checkpoints
    best10 = model.best_model(n=10, merge=True) # Merge the best 10 checkpoint models

    #-------------------------------
    # Evaluate and plot test results
    #-------------------------------
    test_results = model.evaluate(test['X'], test['y'], plot=False)
    best_results = model.evaluate(test['X'], test['y'], 'best', plot=False)
    merge_results = model.evaluate(test['X'], test['y'], 'merge10', plot=False)
    ensemble_results = model.evaluate(test['X'], test['y'], 'ensemble10', plot=False)
    best10_results = model.evaluate(test['X'], test['y'], 'merge_best10', plot=False)

    # Save results to a CSV file
    all_results = pd.DataFrame({'y': test['y'],
                               'base': test_results['predict'],
                               'best': best_results['predict'],
                               'merge_best10': best10_results['predict'],
                               'merge10': merge_results['predict'],
                               'ensemble10': ensemble_results['predict']})
    all_results.to_csv(model_dir + 'predictions.csv')
    model.all_results = all_results

    # Save statistics to a CSV file
    all_stats = pd.DataFrame([test_results['stats'], best_results['stats'],
                              best10_results['stats'], merge_results['stats'],
                              ensemble_results['stats']],
                             index = ['base', 'best', 'merge_best10', 'merge10', 'ensemble10'])
    all_stats['runTime'] = [test_results['runTime'], best_results['runTime'],
                           best10_results['runTime'], merge_results['runTime'],
                           ensemble_results['runTime']]
    all_stats.to_csv(model_dir + 'predict_stats.csv')
    model.all_stats = all_stats

    #--------------------------------------------
    # Overfitting check - predict training labels
    #--------------------------------------------
    train_eval = model.evaluate(train['X'], train['y'], 'merge10', plot=False)
    fig = plot_results('trainData Results', train['y'], train_eval['predict'],
                       train_eval['stats'], vmax=50)
    fig.savefig(model_dir + 'trainData.png', dpi=300)
    plt.close(fig)
    # Save training predictions to a CSV file
    train_predicts = pd.DataFrame({'y': train['y'], 'train': train_eval['predict']})
    train_predicts.to_csv(model_dir + 'train_predicts.csv')
    # Save training prediction statistics to a CSV file
    train_stats = pd.DataFrame([train_eval['stats']], index = ['train'])
    train_stats['runTime'] = [train_eval['runTime']]
    train_stats.to_csv(model_dir + 'train_stats.csv')
    
    return model


def run_kfold_model(model_params, samples, X, y):
    """Runs a K-fold model.
    
    Splits the data into folds, and builds and evaluates a model for
    each fold.

    Parameters
    ----------
    model_params : ModelParams
        The dictionary of model parameters.
    samples : DataFrame
        A data frame containing the samples.
    X : dict
        The predictor data. The keys are the name of each source. Each
        value is an array of the predictors for that source. The first
        dimension is assumed to correspond to the rows of the samples
        data frame in length and order.
    y : Array
        The labels. A 1-dimensional array containing the label for each
        sample.

    Returns
    -------
    models : list
        A list (length k) of the k-fold models. To save memory, the
        Keras components are removed from the models before they are
        returned.
    """
    model_name = model_params['modelName']
    model_dir = os.path.join(model_params['modelDir'], '')
    models = ModelList(model_name, model_dir)

    for fold, data in enumerate(kfold_split(model_params, samples, X, y)):
        fold_params = ModelParams(model_params.copy())
        fold_params['modelName'] = f"{model_name}_fold{fold}"
        fold_params['modelDir'] = os.path.join(model_dir, f'fold{fold}')
        if not os.path.exists(fold_params['modelDir']):
            os.makedirs(fold_params['modelDir'])
        model = train_test_model(fold_params, **data)
        model.clear_model()
        models.append(model)
        
    all_results = models[0].all_results
    for model in models[1:]:
        all_results = all_results.append(model.all_results)
    all_results.to_csv(model_dir + 'predictions.csv')
    models.all_results = all_results

    stats = {}
    for y in all_results.columns[1:]:
        stats[y] = calc_statistics(all_results.y, all_results[y])
    all_stats = pd.DataFrame.from_dict(stats, orient='index')
    all_stats.to_csv(model_dir + 'predict_stats.csv')
    models.all_stats = all_stats

    plot_all_results(all_results, all_stats, model_dir)
        
    return models

def run_test(model_params, samples, X, y):
    """Runs LFMC model test.
    
    Runs a set of tests of LFMC models. The number of runs made is
    controlled by the modelRuns model parameter.
    
    If only a single run is requested, the model parameters are passed
    unchanged to either ``train_test_model`` (after splitting the data)
    or ``run_kfold_model``.
    
    If multiple runs are requested, the model parameters are modified
    to set a unique output directory for each run and to set the random
    seed (using the seedList). If required, the data is re-split
    between each run. After all runs are completed, aggregate
    evaluation statistics are created. If the same splits were used for
    each run, ensembles of equivalent models from all runs are created
    and evaluated.

    Parameters
    ----------
    model_params : ModelParams
        The dictionary of model parameters.
    samples : DataFrame
        A data frame containing the samples.
    X : dict
        The predictor data. The keys are the name of each source. Each
        value is an array of the predictors for that source. The first
        dimension is assumed to correspond to the rows of the samples
        data frame in length and order.
    y : Array
        The labels. A 1-dimensional array containing the label for each
        sample.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    #Build and run several models
    model_dir = os.path.join(model_params['modelDir'], '')
    if model_params['modelRuns'] > 1:
        model_name = model_params['modelName']
        model_seeds = model_params['seedList']
        if model_seeds is None or len(model_seeds) == 0:
            model_seeds = [model_params['randomSeed']]
        models = ModelList(model_name, model_dir)
        data = None
        for run in range(model_params['modelRuns']):
            run_params = ModelParams(model_params.copy())
            run_params['modelName'] = f"{model_name}_run{run}"
            run_params['modelDir'] = os.path.join(model_dir, f"run{run}")
            run_params['randomSeed'] = model_seeds[run] if run < len(model_seeds) else None
            if not os.path.exists(run_params['modelDir']):
                os.makedirs(run_params['modelDir'])
            if model_params['splitFolds'] <= 1:
                if data is None or model_params['resplit']:
                    data = train_test_split(run_params, samples, X, y)
                model = train_test_model(run_params, **data)
                model.clear_model()
                plot_all_results(model.all_results, model.all_stats, run_params['modelDir'])
                models.append(model)
            else:
                models.append(run_kfold_model(run_params, samples, X, y))

        # Calculate mean and variances of the run prediction statistics
        stack_stats = pd.concat([model.all_stats.stack() for model in models], axis=1)
        stack_stats.to_csv(model_dir + 'stats_all.csv')
        models.run_stats = stack_stats
        means = stack_stats.mean(axis=1).unstack()
        means.to_csv(model_dir + 'stats_means.csv')
        models.means = means
        variances = stack_stats.var(axis=1).unstack()
        variances.to_csv(model_dir + 'stats_vars.csv')
        models.variances = variances
        print("\nMeans:\n", means, "\n\nVariances:\n", variances)

        # Ensemble the run predictions if each run used the same splits
        if not model_params['resplit']:
            stack_results = pd.concat([model.all_results.stack() for model in models], axis=1)
            ensembles = stack_results.mean(axis=1).unstack()
            stats = {}
            for y in ensembles.columns[1:]:
                stats[y] = calc_statistics(ensembles.y, ensembles[y])
            all_stats = pd.DataFrame.from_dict(stats, orient='index')
            all_stats.to_csv(model_dir + 'ensemble_stats.csv')
            models.all_stats = all_stats
            print("\nEnsembled Results:\n", all_stats)
            plot_all_results(ensembles, all_stats, model_dir)
            
        return models

    # Build and run a single model
    elif model_params['splitFolds'] <= 1:
        data = train_test_split(model_params, samples, X, y)
        model = train_test_model(model_params, **data)
        model.clear_model()
        plot_all_results(model.all_results, model.all_stats, model_dir)
        return model

    # Build and run a k-fold model
    else:    # model_params['splitFolds'] > 1:
        return run_kfold_model(model_params, samples, X, y)