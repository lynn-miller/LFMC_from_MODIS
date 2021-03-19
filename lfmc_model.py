"""TempCNN model used for LFMC estimation."""

import os
import gc
import glob
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import tensorflow as tf
import tensorflow.keras as keras

from model_parameters import ModelParams
from model_utils import calc_statistics, plot_results


def eval_param(param, starts_with='keras'):
    """Evaluates a string as code

    If ``param`` is prefixed with ``starts_with``, it is assumed to be
    code and evaluated. Other strings are returned unchanged.
    
    If model parameters are set as keras classes or functions, they
    cannot be converted to JSON and stored. Setting the model parameter
    to a string of the function call allows the full set of model
    parameters to be stored in as text, while allowing specification of
    any valid parameter.
    
    Note
    ----
    Currently only implemented for optimizer and kernel_regularizer
    keras parameters (regulariser and optimiser) model parameters.

    Parameters
    ----------
    param : str or object
        The string to evaluate. If not a string it is not evaluated.
    startsWith : str, optional
        Only evaluate strings prefixed by this string. The default is
        'keras'.

    Returns
    -------
    object
        The results of evaluating ``param``. If ``param`` is not a
        string or does not start with ``starts_with``, ``param`` is
        returned unchanged.
    """
    if type(param) is not str:
        return param
    if param.startswith(starts_with):
        return eval(param)
    else:
        return param


def conv1d_block(name, model_params, conv_params):
    """Generates layers for a 1d-convolution block

    Layers generated are: Conv1D, BatchNormalization, Activation,
    Dropout and AveragePooling1D. Conv1d and Activation layers are
    always generated. Other layers are optional and only included if
    the relevant model_params and conv_params are set as below.
    
    Parameters
    ----------
    name : str
        The name of the block. Used to generate unique names for each
        layer.
    model_params : ModelParams
        Uses the following ModelParams values:
          - initialiser  
          - regulariser  
          - activation  
          - dropoutRate (Dropout layer included if > 0)
    conv_params : dict
        The convolutional parameters for this block. Required keys are:
          - filters
          - kernel
          - stride
          - bnorm (BatchNormalization layer included if True)
          - poolSize (AveragePooling1D layer included if > 0)

    Returns
    -------
    block : list of keras layers
        The keras layers that comprise the convolution block.
    """
    block = []
    block.append(keras.layers.Conv1D(name=name + '_conv1d',
                                     filters=conv_params['filters'],
                                     kernel_size=conv_params['kernel'],
                                     strides=conv_params['stride'],
                                     padding="same",
                                     kernel_initializer=model_params['initialiser'],
                                     kernel_regularizer=eval_param(model_params['regulariser'])))
    if conv_params['bnorm']:
        block.append(keras.layers.BatchNormalization(name=name + '_bnorm', axis=-1))
    block.append(keras.layers.Activation(model_params['activation'], name=name + '_act'))
    if model_params['dropoutRate'] > 0:
        block.append(keras.layers.Dropout(model_params['dropoutRate'], name=name + '_dropout'))
    if conv_params['poolSize'] > 0:
        block.append(keras.layers.AveragePooling1D(pool_size=conv_params['poolSize'],
                                                   name=name + '_pool'))
    return block

	
def dense_block(name, model_params, fc_params):
    """Generates layers for a fully-connected (dense) block
 
    Layers generated are: Dense, BatchNormalization, Activation,
    and Dropout. Dense and Activation layers are always generated. The
    other layers are optional and only included if the relevant
    model_params and conv_params are set as below.

    Parameters
    ----------
    name : str
        The name of the block. Used to generate unique names for each
        layer.
    model_params : ModelParams
        Uses the following ModelParams values:
            initialiser
            regulariser
            activation
            dropoutRate (Dropout layer included if > 0)            
    fc_params : dict
        The fully-connected parameters for this block. Required keys
        are:
            units
            bnorm (BatchNormalization layer included if True)

    Returns
    -------
    block : list of keras layers
        The keras layers that comprise the fully-connected block.
    """
    block = []
    block.append(keras.layers.Dense(fc_params['units'],
                                    name=name + '_dense',
                                    kernel_initializer=model_params['initialiser'],
                                    kernel_regularizer=eval_param(model_params['regulariser'])))
    if fc_params['bnorm']:
        block.append(keras.layers.BatchNormalization(name=name + '_bnorm', axis=-1))
    block.append(keras.layers.Activation(model_params['activation'], name=name + '_act'))
    if model_params['dropoutRate'] > 0:
        block.append(keras.layers.Dropout(model_params['dropoutRate'], name=name + '_dropout'))
    return block

class ModelList(list):
    """A list of models or ModelLists
    
    Adds a name and model_dir attribute to a list and allows related
    models to be grouped hierarchically - e.g. if the represent a set
    of tests, and each test is run several times. 
    """
    
    def __init__(self, name, model_dir):
        self.name = name
        self.model_dir = model_dir
        super(ModelList, self).__init__()

class LfmcTempCnn():
    """A TempCNN for LFMC estimation
    
    Includes methods to build, compile, train and evaluate a keras
    1D-CNN model, and to save the models and other outputs.
    
    Parameters
    ----------
    params: ModelParams, optional
        All parameters needed for building the Keras model. Default is
        None. The default can be used when creating an object for a
        previously built and trained model. The params will then be
        loaded when the load method is called.
    
    inputs: dict, optional
        If specified, the init method will build and compile the model.
        The parameter should contain a key for each input and the
        values an array of the correct shape. Default is None, meaning
        the model is not built or compiled.
        
    Attributes
    ----------
    params: ModelParams
        All parameters needed for building the Keras model
        
    monitor: str
        Indicates if the callbacks should monitor the training loss
        (``loss``) or validation loss (``val_loss``). Set to ``val_loss``
        if a validation set is used (i.e. ``params['validationSet']``
        is ``True``), else set to ``loss``.
    
    model_dir: str
        The directory where model outputs will be stored. Identical to
        ``params['modelDir']``
    
    callback_list: list
        A list of keras callbacks. Always includes the ModelCheckpoint
        callback and optionally includes the EarlyStopping callback.
    
    model: keras.Model
        The keras model
        
    derived_models: dict
        Dictionary of models derived from the trained model using the
        ``best_model``, ``merge_models``, or ``ensemble_models``
        methods. Keys are the model names, values are type keras.Model,
        or a list of keras.Models if an ensemble model.
    
    history: list
        The monitored loss and other metrics at each checkpoint.
    """
    
    input_list = ['modis', 'aux']
    
    def __init__(self, params=None, inputs=None):
        if params is not None:
            self.params = params
            self.monitor = 'val_loss' if self.params['validationSet'] else 'loss'
            self.model_dir = os.path.join(params['modelDir'], '')  # add separator if necessary
            np.random.seed(params['randomSeed'])
            tf.random.set_seed(params['randomSeed'])
        if inputs is not None:
            self.build(inputs)
            self.compile()
            self.set_callbacks()
        self.derived_models = {}

    def _inputsToList(self, inputs, build=False):
        if build:
            return [keras.Input(inputs[source].shape[1:], name=source)
                    for source in self.input_list]
        else:
            return [inputs[source] for source in self.input_list]
    
    def _addBlock(self, block, input):
        x = input
        for layer in block:
            x = layer(x)
        return x
    
    def build(self, inputs):
        """Builds the model.
        
        Build the model with layers and hyper-parameters as specified
        in params. The shape of the inputs are used to define the keras
        input layers. 
        
        Parameters
        ----------
        inputs : dict
            Should contain a key for each input and values an array of
            the correct shape. The dimensions of the array are used to
            build the model layers.

        Returns
        -------
        None.
        """
        inputs = self._inputsToList(inputs, build=True)
        x = inputs[0] # modis
        for i, conv_params in enumerate(self.params['conv']):
            x = self._addBlock(conv1d_block(f'conv{i}', self.params, conv_params), x)
        x = keras.layers.Flatten(name='flatten')(x)
        x = keras.layers.Concatenate(name='concat')([x, inputs[1]]) # aux
        for i, fc_params in enumerate(self.params['fc']):
            x = self._addBlock(dense_block(f'dense{i}', self.params, fc_params), x)
        x = keras.layers.Dense(1, kernel_initializer=self.params['initialiser'],
                               activation='linear', name='final')(x)
        self.model = keras.Model(inputs=inputs, outputs=x, name=self.params['modelName'])

    def compile(self):
        """Compiles the model
        
        Compiles the model using the optimizer, loss function and
        metrics from params.

        Returns
        -------
        None.
        """
        self.model.compile(optimizer=eval_param(self.params['optimiser']),
                           loss=self.params['loss'],
                           metrics=self.params['metrics'])

    def set_callbacks(self):
        """Creates a list of the model callbacks.
        
        The callback list contains a ModelCheckpoint callback, so a
        checkpoint is taken after each epoch. If earlyStopping is set
        in params, an EarlyStopping checkpoint is also created. The
        callbacks are saved to the callback_list attribute.

        Returns
        -------
        None.
        """
        checkpoint_path = os.path.join(self.params['tempDir'], '_{epoch:04d}_temp.h5')
        checkpoint = keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor=self.monitor,
                verbose=self.params['verbose'],
                save_best_only=False,
                save_weights_only=False,
                mode='min',
                save_freq='epoch')
        self.callback_list = [checkpoint]
        if self.params['earlyStopping'] > 1:
            early_stop = keras.callbacks.EarlyStopping(
                    monitor=self.monitor,
                    min_delta=0,
                    patience=self.params['earlyStopping'],
                    verbose=self.params['verbose'],
                    mode='auto')
            self.callback_list.append(early_stop)

    def train(self, Xtrain, ytrain, Xval=None, yval=None):
        """Trains the model
        

        Parameters
        ----------
        Xtrain : dict
            The training features. The dict keys are the sources and
            the values are arrays of the data for each source.
        ytrain : array
            The training labels.
        Xval : dict, optional
            The validation features in the same format as Xtrain. The
            default is None. Ignored if the model params has
            validationSet = False.
        yval : TYPE, optional
            The validation labels. The default is None. Ignored if the
            model params has validationSet = False.

        Returns
        -------
        dict
            Dictionary of the training results, with keys:
            'minLoss' - the minimum loss from the training checkpoints
            'history' - the loss and metrics at all checkpoints
            'runTime' - the training time in seconds
        """
        Xtrain = self._inputsToList(Xtrain)
        Xval = self._inputsToList(Xval)
        start_train_time = time.time()
        if self.params['validationSet']:
            hist = self.model.fit(x=Xtrain, y=ytrain, epochs=self.params['epochs'],
                    batch_size=self.params['batchSize'], shuffle=self.params['shuffle'],
                    validation_data=(Xval, yval),
                    verbose=self.params['verbose'], callbacks=self.callback_list)
        else:
            hist = self.model.fit(x=Xtrain, y=ytrain, epochs=self.params['epochs'],
                    batch_size=self.params['batchSize'], shuffle=self.params['shuffle'],
                    verbose=self.params['verbose'], callbacks=self.callback_list)
        trainTime = round(time.time() - start_train_time, 2)
        self.history = pd.DataFrame(hist.history)
        self.history.set_index(self.history.index + 1, inplace=True)
        self.save()
        return {'minLoss': np.min(hist.history[self.monitor]),
                'history': hist.history,
                'runTime': trainTime}

    def predict(self, X, model_name=None, ensemble=np.mean, batch_size=1024, verbose=0):
        """Predicts labels from a model
        
        Parameters
        ----------
        X : dict
            The features to predict from.
        model_name : str, optional
            The name of the model to use for prediction. The 'base'
            model (i.e. fully-trained model) is used if no model
            specified. The default is None.
        ensemble : function, optional
            If the model is an ensemble, the function to use to combine
            the individual predictions. The default is np.mean.
        batch_size : int, optional
            keras.Model.predict batch_size parameter. The default is 1024.
        verbose : int, optional
            keras.Model.predict verbose parameter. The default is 0.

        Returns
        -------
        yhat : array
            The predicted labels.
        """
        X = self._inputsToList(X)
        model = self.model if model_name is None else self.derived_models[model_name]
        if type(model) is list:
            yhat = [m.predict(X, batch_size=batch_size, verbose=verbose) for m in model]
            yhat = ensemble(np.hstack(yhat), axis=1)
        else:
            yhat = model.predict(X, batch_size=batch_size, verbose=verbose).flatten()
        return yhat

    def evaluate(self, X, y, model_name=None, ensemble=np.mean, plot=True):
        """Evaluates the model
        
        Evaluate the model using X as the test data and y as the labels
        for the test data.

        Parameters
        ----------
        X : dict
            The test data.
        y : array
            The test labels.
        model_name : str, optional
            Name of the model to use for the predictions. The default
            is None.
        ensemble : function, optional
            The function to use if the model is an ensemble. The
            default is np.mean.
        plot : bool, optional
            Flag indicating if a scatter plot of the results should be
            created. The default is True.

        Returns
        -------
        dict
            Dictionary of the evaluation results, with keys:
              - 'predict' - the predicted values
              - 'stats' - the evaluation statistics (bias, R, R2, RMSE,
                ubRMSE)
              - 'runTime' - the prediction time in seconds
        """
        start_test_time = time.time()
        yhat = self.predict(X, model_name=model_name, ensemble=ensemble)
        test_time = round(time.time() - start_test_time, 2)
        stats = calc_statistics(y, yhat)
        fig_name = 'base' if model_name is None else model_name
        if plot:
            fig = plot_results(f'{fig_name} Results', y, yhat, stats)
            fig.savefig(self.model_dir + fig_name + '.png', dpi=300)
        return {'predict': yhat, 'stats': stats, 'runTime': test_time}

    def summary(self):
        """Prints the model summary
        
        Returns
        -------
        None.
        """
        self.model.summary()
    
    def plot(self, file_name):
        """Saves model plot
        
        Calls the Keras plot_model utility to create an image of the
        model network.

        Parameters
        ----------
        file_name : str
            Name of the plot file (relative to self.model_dir).

        Returns
        -------
        None.
        """
        outFile = self.model_dir + file_name
        keras.utils.plot_model(self.model, to_file=outFile, show_shapes=True,
                               show_layer_names=True)

    def load(self, model_dir):
        """Loads the model
        
        Loads a saved model from disk. Run this method after creating
        the instance with no parameters.

        Parameters
        ----------
        model_dir : str
            The full path name of the directory storing the model.

        Returns
        -------
        None.
        """
        self.model = keras.models.load_model(model_dir + 'base.h5')
        self.history = pd.read_csv(model_dir + 'train_history.csv')
        self.history.set_index(self.history.index + 1, inplace=True)
        with open(model_dir + 'model_params.json', 'r') as f:
            self.params = ModelParams(source = f)
        self.monitor = 'val_loss' if self.params['validationSet'] else 'loss'
        self.model_dir = model_dir

    def save(self):
        """Saves the model
        
        Saves the Keras model, model plot, model parameters and
        training history

        Returns
        -------
        None.
        """
        self.model.save(self.model_dir + 'base.h5')
        self.plot('model_plot.png')
        with open(self.model_dir + 'model_params.json', 'w') as f:
            self.params.save(f)
        self.history.to_csv(self.model_dir + 'train_history.csv', index=False)
        
    def load_model(self, model_name):
        """Loads a derived model
        
        Parameters
        ----------
        model_name : str
            The name of the derived model. This is assumed to be the
            filename (relative to the model directory), excluding the
            suffix, for a single model; or a directory containing the
            individual files for an ensemble.

        Returns
        -------
        None.
        """
        try:
            saved_model = keras.models.load_model(f"{self.model_dir}{model_name}.h5")
        except:
            model_files = glob.glob(os.path.join(self.model_dir, model_name, "*.h5"))
            saved_model = [m for m in model_files]
        self.derived_models[model_name] = saved_model
        
    def _save_model(self, new_model, model_name):
        if type(new_model) is list:
            save_dir = os.path.join(self.model_dir, model_name)
            if os.path.exists(save_dir):
                shutil.rmtree(save_dir)
            os.makedirs(save_dir)
            for i, m in enumerate(new_model):
                m.save(os.path.join(save_dir, f"m{i:03d}.h5"))
        else:
            new_model.save(f"{self.model_dir}{model_name}.h5")
        self.derived_models[model_name] = new_model
        
    def _get_model_list(self, sort=True):
        model_list = glob.glob(os.path.join(self.params['tempDir'], '*_temp.h5'), recursive=True)
        if sort:
            model_list.sort()
        return model_list
    
    def clear_model(self):
        """Clears the model
        
        Removes as much of the model as possible, including:
          - model and derivedModel attributes
          - keras backend session
          - runs garbage collection
          - delete all checkpoint files
            
        After running clear_model, the model instance still exists but
        no keras components will exist. As the checkpoint files are
        removed, no new derived models can be created. All other
        components are saved to disk and can be reloaded using the load
        and load_model methods.

        Returns
        -------
        None.
        """
        self.model = None
        self.derived_models = None
        keras.backend.clear_session()
        gc.collect()
        for file in self._get_model_list(sort=False):
            os.remove(file)
        
    def get_models(self, models=None, last_n=False, load=False):
        """Gets a subset of checkpoint models
        

        Parameters
        ----------
        models : int or list of int, optional
          - If a list of int, then interpreted as a list of checkpoint
            numbers.
          - If an int and ``last_n`` is True, then interpreted as
            requesting the last N models, where N is ``abs(models)``.
          - If an int and ``last_n`` is False, then interpreted as the
            required checkpoint number, or if negative, the Nth last
            checkpoint.
          - If None, all checkpoints are returned. The default is None.
        last_n : bool, optional
            If True and ``models`` is type int, interpret models as a
            request for the last N models. Ignored if ``models`` is
            type str. The default is False.
        load : bool, optional
            If True, load the set of models from the saved checkpoints
            and return a list of the models. If False, return a list of
            model file names. The default is False.

        Returns
        -------
        str, keras.Model, or list
            If a single model requested then either the filename of the
            checkpoint (``load=False``) or the requested checkpoint as
            a keras Model (``load=True``). If more than one model
            requested, then the checkpoint filenames (``load=False``)
            or the checkpoint models (``load=True``) as a list.
        """
        model_list = self._get_model_list()
        model_series = pd.Series(model_list, index=self.history.index)
        if models is None:   # Return the entire series of models
            return model_series
        # Select the requested model(s)
        if type(models) is int and last_n:
            modelx = model_series[-abs(models):]
        else:
            if type(models) is int and models < 0:
                modelx = model_list[models]   # Use the list as np.series[-n] fails
            else:
                modelx = model_series[models]
        if load:    # Load the requested model(s)
            if type(modelx) is str:  # Only one model requested
                return keras.models.load_model(modelx)
            else:
                return [keras.models.load_model(m) for m in modelx]
        else:
            return modelx

    def merge_models(self, model_name, models=None):
        """Creates a model by merging a set of models
        
        Create a model from a set of model checkpoints by averaging the
        equivalent weights from each model. A merged model is similar
        to an ensemble but usually slightly less accurate than an
        ensemble of the same set. The advantage is that it is much
        faster to predict from the merged model.

        Parameters
        ----------
        model_name : str
            A name for the merged model.
        models : list, optional
            The models to merge. See ``get_models model`` parameter for
            details of how this parameter is used. The default is None,
            meaning all checkpoints are merged.

        Returns
        -------
        None.
        """
        weights = [m.get_weights() for m in self.get_models(models, last_n=True, load=True)]
        # average weights
        new_weights = [np.array(w).mean(axis=0) for w in zip(*weights)]
        model_conf = self.model.get_config()
        new_model = keras.models.Model.from_config(model_conf)
        new_model.set_weights(new_weights)
        self._save_model(new_model, model_name)

    def ensemble_models(self, model_name, models=None):
        """Creates a model by ensembling a set of models
        
        Create a model as an ensemble of checkpoints.

        Parameters
        ----------
        model_name : str
            A name for the ensemble model.
        models : list, optional
            The models to ensemble. See ``get_models model`` parameter
            for details of how this parameter is used. The default is
            None, meaning all checkpoints are ensembled.

        Returns
        -------
        None.
        """
        ensemble = self.get_models(models, last_n=True, load=True)
        self._save_model(ensemble, model_name)

    def best_model(self, n=1, merge=True):
        """Creates a model using the best ``n`` checkpoints
        
        Create a model using the checkpoint(s) with the lowest loss.

        Parameters
        ----------
        n : int
            The number of checkpoints to use. If 1, then a model is
            created using a single checkpoint with the lowest loss. If
            > 1, create a model using the ``n`` checkpoints with the
            lowest loss. The default is 1.
        merge : bool, optional
            If n > 1, indicates whether to merge (True) or ensemble
            (False) the best ``n`` checkpoints. The default is True.

        Returns
        -------
        None.
        """
        best = list(self.history[self.monitor].nsmallest(n).index)
        if len(best) == 1:
            best = best[0]
            best_model = self.get_models(best, load=True)
            self._save_model(best_model, 'best')
        elif merge:
            self.merge_models(f'merge_best{n:02d}', best)
        else:
            self.ensemble_models(f'ensemble_best{n:02d}', best)
        return best
    
    def plot_train_hist(self, file_name=None, metric=None, rolling=10):
        """Creates a plot of the training result at each epoch.

        Parameters
        ----------
        file_name : str, optional
            The file name for the plot, relative to the model directory.
            The `.png` extension is appended to the file name. If
            ``None`` the name ``training_{metric}.png`` is used. The
            default is None.
        metric : str, optional
            The name of the metric to plot. If ``None``, the checkpoint
            monitored metric is used, otherwise can be any metric
            included in the ``metrics`` model parameter. The default is
            None.
        rolling : int, optional
            The number of epochs to use to plot the rolling (moving)
            average. If 1 (or less), no rolling average is plotted. The
            default is 10.

        Returns
        -------
        None.
        """
        if metric is None:
            metric = self.monitor
        if file_name is None:
            file_name = f'training_{metric}'
        plt.figure(figsize=(5, 5))
        plt.plot(self.history[metric], label='Epoch values')
        if rolling > 1:
            plt.plot(self.history[metric].rolling(rolling, center=True).mean(),
                     label=f'Moving average({rolling})')
        plt.ylabel(metric, fontsize=12)
        plt.xlabel('epoch', fontsize=12)
        minY = round(self.history[metric].min()*2-50, -2)/2
        if self.params['epochs'] >= 10:
            maxY = round(self.history[metric][5:].max()*2+50, -2)/2
        else:
            maxY = round(self.history[metric].max()*2+50, -2)/2
        plt.axis([0, self.params['epochs'], minY, maxY])
        plt.legend()
        plt.title(f'Training Results - {metric}')
        plt.savefig(self.model_dir + file_name + '.png', dpi=300)
        plt.close()
