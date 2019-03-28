import importlib
from typing import Callable
from keras.models import Model
from keras.layers import Dropout, Dense
from keras.optimizers import Adam
from keras.callbacks import History
from imageatm.handlers.data_generator import DataGenerator


class ImageClassifier:
    """Class that represents the classifier for images.

    The following pretrained CNNs from Keras can be used as base model:
    - Xception
    - VGG16
    - VGG19
    - ResNet50, ResNet101, ResNet152
    - ResNet50V2, ResNet101V2, ResNet152V2
    - ResNeXt50, ResNeXt101
    - InceptionV3
    - InceptionResNetV2
    - MobileNet
    - MobileNetV2
    - DenseNet121, DenseNet169, DenseNet201
    - NASNetLarge, NASNetMobile

    Attributes:
        base_model_name: Name of Keras base model.
        n_classes: Number of classes.
        learning_rate: Learning rate for training phase.
        dropout_rate: Fraction set randomly.
        loss: A loss function as one of two parameters to compile the model.
        weights: Pretrained weights the model architecture is loaded with (default imagenet).
    """

    def __init__(
        self,
        base_model_name: str,
        n_classes: int,
        learning_rate: float,
        dropout_rate: float,
        loss: str,
        weights: str = 'imagenet',
    ) -> None:
        """Inits ImageClassifier object.

        Loads Keras base_module specified by base_model_name.
        """
        self.n_classes = n_classes
        self.base_model_name = base_model_name
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.loss = loss
        self.weights = weights
        self._load_base_module()

    def _load_base_module(self):
        """Loads Keras base_module specified by base_model_name. """
        if self.base_model_name == 'InceptionV3':
            self.base_module = importlib.import_module('keras.applications.inception_v3')

        elif self.base_model_name == 'InceptionResNetV2':
            self.base_module = importlib.import_module('keras.applications.inception_resnet_v2')

        elif self.base_model_name in ['NASNetLarge', 'NASNetMobile']:
            self.base_module = importlib.import_module('keras.applications.nasnet')

        elif self.base_model_name in ['DenseNet121', 'DenseNet169', 'DenseNet201']:
            self.base_module = importlib.import_module('keras.applications.densenet')

        elif self.base_model_name in ['ResNet50', 'ResNet101', 'ResNet152']:
            self.base_module = importlib.import_module('keras.applications.resnet')

        elif self.base_model_name in ['ResNet50V2', 'ResNet101V2', 'ResNet152V2']:
            self.base_module = importlib.import_module('keras.applications.resnet_v2')

        elif self.base_model_name in ['ResNeXt50', 'ResNeXt101']:
            self.base_module = importlib.import_module('keras.applications.resnext')

        else:
            self.base_module = importlib.import_module(
                'keras.applications.' + self.base_model_name.lower()
            )

    def get_base_layers(self) -> list:
        """ Gets layers of classifiers' base model

        Returns:
            base_layers: list of layers
        """
        return self.base_model.layers

    def get_preprocess_input(self) -> Callable:
        """ Gets preprocess_input of classifiers' base_module

        Returns:
            preprocess_input: Callable
        """
        return self.base_module.preprocess_input

    def set_learning_rate(self, learning_rate: float):
        """
        sets classifiers' learning_rate

        Args:
            learning_rate: the learning_rate.
        """
        self.learning_rate = learning_rate

    def build(self):
        """
        Builds classifiers' model.

        The following steps will be performed in sequence:
            - Loads a pretrained base model.
            - Adds dropout and dense layer to the model.
            - Sets classifiers' model.
        """
        BaseCnn = getattr(self.base_module, self.base_model_name)
        self.base_model = BaseCnn(
            input_shape=(224, 224, 3), weights=self.weights, include_top=False, pooling='avg'
        )

        x = Dropout(self.dropout_rate)(self.base_model.output)
        x = Dense(units=self.n_classes, activation='softmax')(x)

        self.model = Model(self.base_model.inputs, x)

    def compile(self):
        """ Configures classifiers' model for training."""
        self.model.compile(
            optimizer=Adam(lr=self.learning_rate), loss=self.loss, metrics=['accuracy']
        )

    def fit_generator(self, **kwargs) -> History:
        """
        Trains classifiers' model on data generated by a Python generator.

        Args:
            generator: Input samples from a data generator on which to train the model.
            validation_data: Input samples from a data generator on which to evaluate the model.
            epochs: Number of epochs to train the model.
            initial_epoch: Epoch at which to start training.
            verbose: Verbosity mode.
            use_multiprocessing: Use process based threading.
            workers: Maximum number of processes.
            max_queue_size: Maximum size for the generator queue.
            callbacks: List of callbacks to apply during training.

        Returns:
            history: A `History` object.

        """
        return self.model.fit_generator(**kwargs)

    def predict_generator(self, data_generator: DataGenerator, **kwargs) -> History:
        """
        Generates predictions for the input samples from a data generator.

        Args:
            data_generator: Input samples from a data generator.
            workers: Maximum number of processes.
            use_multiprocessing: Use process based threading.
            verbose: Verbosity mode.

        Returns:
            history: A `History` object.
        """
        return self.model.predict_generator(data_generator, **kwargs)

    def summary(self):
        """ Summarizes classifiers' model."""
        self.model.summary()
