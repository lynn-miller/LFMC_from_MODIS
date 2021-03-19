"""The Model Parameters dictionary"""

import pprint
import json

class ModelParams(dict):
    """A dictionary for LFMC model parameters
    
    Extends the dictionary class by adding a help function. By default, the dictionary is
    created with keys for all parameters needed to build a model for LFMC estimation and
    initialised to the default values.
    
    Parameters
    ----------
        source : None, dict, str, or file object, optional
            If dict: A dictionary containing all model parameters
            If str: A string representation of the model parameters in JSON format.
            If file object: An open JSON file containing all model parameters.
            If None: The object is initialised with defaults for all model parameters.
            The default is None.
        model_name : str, optional
            The name of the model. It should be a valid Keras model name. The default is
            'default_model'.
        conv_layers : int, optional
            The number of convolutional layers to create. See the ``set_layers`` method for more
            details. The default is 0.
        fc_layers : int, optional
            The number of fully connected or dense layers to create. See the ``set_layers`` method
            for more details. The default is 0.
        
    Attributes
    ----------
    _param_help: dict
        A dictionary containing the ``help`` text for each model parameter. The ``general``
        key contains the ``help`` text for the object.
    """
    
    def __init__(self, source=None, model_name='default_model', conv_layers=0, fc_layers=0):
        if source is None:    # New model - set all parameters to defaults
            model_params = self.set_defaults(model_name)
            super(ModelParams, self).__init__(model_params)
            self.set_layers(conv_layers, fc_layers)
        else:
            if type(source) is dict:   # Set parameters from a dictionary
                model_params = source
            elif type(source) is str:  # Set parameters from a JSON string
                model_params = json.loads(source)
            else:                      # Set parameters from a JSON input file
                model_params = json.load(source)
            super(ModelParams, self).__init__(model_params)
        
    def set_defaults(self, model_name='default_model'):
        """Returns the default model parameters dictionary
        
        Parameters
        ----------
        model_name : str, optional
            The name of the model. It should be a valid Keras model name. The default is
            'default_model'.

        Returns
        -------
        model_params : dict
            The dictionary of model parameters.
        """
        model_params = {
            'modelName': model_name,
            'description': '',
            'modelDir': '',
            'tempDir': '',

            'randomSeed': 1234,
            'modelRuns': 1,
            'resplit': False,
            'seedList': [],

            # MODIS data parameters
            'modisFilename': None,
            'modisChannels': 7,
            'modisNormalise': {'method': 'minMax', 'percentiles': 2},
            'modisDays': 365,

            # Auxiliary data parameters
            'auxFilename': None,
            'auxColumns': 9,
            'auxAugment': False,
            'targetColumn': 'LFMC value',

            # Data splitting parameters
            'splitMethod': 'byYear',          # 'random', 'byYear', 'bySite'
            'splitSizes': (0.33, 0.086) ,     # for random/bySite methods - test/val proportions
            'siteColumn': 'Site',
            'splitStratify': 'Land Cover',
            'splitYear': 2013,                # for byYear method - last training year
            'yearColumn': 'Sampling year',
            'splitFolds': 0,                  # If > 1, create k-Fold models

            # Overfitting controls
            'batchNormalise': True,
            'dropoutRate': 0.5,
            'regulariser': 'keras.regularizers.l2(1.e-6)',
            'validationSet': False,
            'earlyStopping': False,           # int > 1 for early stopping (used as patience parameter

            # Fitting parameters
            'epochs': 100,
            'batchSize': 32,
            'shuffle': True,
            'verbose': 0,

            # Keras methods
            'optimiser': 'adam',              # Either the name of the optimiser or a string that evaluates to a Keras optimizer object
            'activation': 'relu',
            'initialiser': 'he_normal',
            'loss': 'mean_squared_error',
            'metrics': ['mean_absolute_error'],
        }
        return model_params

    def set_layers(self, conv_layers=None, fc_layers=None):
        """Sets model layer parameters
        
        Sets the parameters for the convolutional (``conv``) and fully connected (``fc``) layers.

        Parameters
        ----------
        conv_layers : int, optional
            The number of convolutional layers. The default is None.
        fc_layers : int, optional
            The number of fully connected layers. The default is None.

        Returns
        -------
        None.

        """
        conv_params = {
            'filters': 32,   # Convolution filters
            'kernel': 5,     # Convolution kernel size
            'stride': 1,     # Convolution stride
            'bnorm': self['batchNormalise'],
            'poolSize': 0,   # 0 to disable pooling
        }

        fc_params = {
            'units': 256,
            'bnorm': self['batchNormalise'],
        }

        # Set layer parameters to defaults
        if conv_layers is not None:
            self['conv'] = [conv_params.copy() for _ in range(conv_layers)]
        if fc_layers is not None:
            self['fc'] = [fc_params.copy() for _ in range(fc_layers)]

    def __str__(self):
        return pprint.pformat(self, width=100, sort_dicts=False)

    def save(self, stream=None):
        """Saves the model parameters
        
        Convert the model parameters dictionary to a JSON string and optionally save to a file.
        
        Parameters
        ----------
        stream : file handle, optional
            A file handle for the output file. If None, the converted JSON string is returned.
            The default is None.

        Returns
        -------
        str
            The JSON representation of the model parameters. Only returned if no file stream
            parameter specified.

        """
        if stream is None:  # No output file - return parameters as a JSON string
            return json.dumps(self, indent=2)
        else:               # Save parameters to the output file
            json.dump(self, stream, indent=2)

    def help(self, key=None):
        """Prints a help message
        
        Prints the general help message if no key provided, or the help message for the specified
        key.
        
        Parameters
        ----------
        key : str, optional
            The key for which help is requested. The general help message is displayed if no key is
            specified. The default is None.

        Returns
        -------
        None.
        """
        def pp(text, indent=0, quote=False):
            spaces = " " * indent
            if quote:
                out = pprint.pformat(text).replace("('", "'").replace("')", "'").replace(
                        "\n ", f"\n{spaces}").replace("\\n", "\n")
            else:
                out = pprint.pformat(text).replace("('", "").replace("')", "").replace(
                        "'", "").replace("\n ", f"\n{spaces}").replace("\\n", "\n")
            return out

        sep = '\n  '
        if key is None:
            text = pp(self._param_help['general']) + sep + sep.join(self)
        else:
            keyValue = pp(self.get(key, 'not defined'), indent=10, quote=True)
            keyHelp = pp(self._param_help.get(key, 'not available'), indent=8)
            text = f'{key}:\n  value: {keyValue}\n  help: {keyHelp}'
        print(text)

    _param_help = {
        'general':        'Dictionary of all parameters used to build an LFMC model. For more help '
                          'run model_params.help("parameter").\nAvailable parameters are:',
        'modelName':      'A name for the model; must be a valid Keras model name',
        'description':    'A free-format description of the model - only used for documentation',
        'modelDir':       'A directory for all model outputs',
        'tempDir':        'Directory used to store temporay files such as checkpoints',
        'randomSeed':     'Number used to set all random seeds (for random, numpy and tensorflow)',
        'modelRuns':      'Number of times to buid and run the model',
        'resplit':        'True: redo the test/train splits on each run; False: use the same ' \
                          'test/train split for each run',
        'seedList':       'A list of random seeds used to seed each run if modelRuns > 1. If the ' \
                          'list size (n) is less than the number of runs, then only the first n ' \
                          'runs will be seeded. If the list is empty (and modelRuns > 1) the ' \
                          'randomSeed will be used to seed the first run, all other runs will be ' \
                          'unseeded. Extra seeds (n > modelRuns) are ignored.',
        'modisFilename':  'Full path name of the file containing the MODIS data',
        'modisChannels':  'Number of channels in the MODIS dataset',
        'modisNormalise': 'A dictionary containing the method to use to normalise the MODIS data, '\
                          'plus any parameters required by this method',
        'modisDays':      'Number of days of MODIS data to use',
        'auxFilename':    'Full path name of the file containing the auxiliary data and target',
        'auxAugment':     'Indicates if the auxiliary data should be augmented with the last day '\
                          'of MODIS data',
        'targetColumn':   'Column name (in the auxiliary data) of the target column',
        'splitMethod':    '"random" for random train/test splits, "byYear" to split data by '\
                          'sample collection ' \
                          'year, "bySite" to split the data by sample collection site',
        'splitSizes':     'A tuple specifying the proportion of data or sites to use for test and '\
                          'validation sets for the "random" and "bySite" split methods. If no ' \
                          'validation set is used, only one value is needed but must be a tuple.',
        'siteColumn':     'Column name (in the auxiliary data) of the sample collection site.',
        'splitStratify':  'Specifies the column (in the auxiliary data) to use for stratified '\
                          'splits, if these are required. Set to False to disable stratified '\
                          'splits. Ignored for "byYear" splits.',
        'yearColumn':     'For "byYear" splits, specifies the column (in the auxiliary data) of ' \
                          'the sample collection year.',
        'splitFolds':     'If > 1, k-fold splitting will be used. If False, 0 or 1, a single ' \
                          'random split will be made. Ignored for "byYear" splits.',
        'batchNormalise': "Default setting for including a batch normalisation step in a block, "\
                          "can be overriden for a block using the block's bnorm setting",
        'dropoutRate':    "The dropout rate to use for all layers in the model",
        'regulariser':    'A string representation of the regulariser to use in all model layers. '\
                          'If the string starts with "keras", it will be interpreted as a call to '\
                          'a keras regularizer function.',
        'validationSet':  'Indicates if a validation set should be used when training the model',
        'earlyStopping':  'If False or 0, early stopping is not used. Otherwise early stopping '\
                          'is used and the value used as the patience setting.',
        'epochs':         'Number of training epochs.',
        'batchSize':      'Training batch size.',
        'shuffle':        'Indicates if the training data should be shuffled between epochs.',
        'verbose':        'Sets the verbosity level during training.',
        'optimiser':      'The Keras optimiser. If the value starts with "keras", it will be '\
                          'interpreted as code to create a Keras optimizer object.',
        'activation':     'The activation function to use for all model layers',
        'initialiser':    'The function used to initialise the model weights',
        'loss':           'The loss function to use when training the model.',
        'metrics':        'A list of metrics to be evaluated at each checkpoint.',
        'conv':           'A list of convolutional parameter sets, each entry in the list '\
                          'corresponds to a convolutional layer that will be added to the model',
        'fc':             'A list of fully connected parameter sets, each entry in the list '\
                          'corresponds to a fully connected layer that will be added to the model',
    }
        