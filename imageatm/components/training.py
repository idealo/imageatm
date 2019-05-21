import typing
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from pathlib import Path
from imageatm.utils.io import load_json
from imageatm.utils.logger import get_logger
from imageatm.utils.tf_keras import use_multiprocessing, LoggingMetrics, LoggingModels
from imageatm.handlers.data_generator import TrainDataGenerator, ValDataGenerator
from imageatm.handlers.image_classifier import ImageClassifier


tf.logging.set_verbosity(tf.logging.ERROR)

BATCH_SIZE = 64
DROPOUT_RATE = 0.75
BASE_MODEL_NAME = 'MobileNet'
LOSS = 'categorical_crossentropy'
EPOCHS_TRAIN_DENSE = 100
EPOCHS_TRAIN_ALL = 100
LEARNING_RATE_DENSE = 0.001
LEARNING_RATE_ALL = 0.0003


class Training:
    """Builds model and runs training.

    The following pretrained CNNs from Keras can be used for transfer learning:

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

    Training is split into two phases, at first only the last dense layer gets
    trained, and then all layers are trained. The maximum number of epochs for
    each phase is set by *epochs_train_dense* (default: 100) and
    *epochs_train_all* (default: 100), respectively. Similarly,
    *learning_rate_dense* (default: 0.001) and *learning_rate_all*
    (default: 0.0003) can be set.

    For each phase the learning rate is reduced after a patience period if no
    improvement in validation accuracy has been observed. The patience period
    depends on the average number of samples per class:

    - if n_per_class < 200: patience = 5 epochs
    - if n_per_class >= 200 and < 500: patience = 4 epochs
    - if n_per_class >= 500: patience = 2 epochs

    The training is stopped early after a patience period that is three times
    the learning rate patience to allow for two learning rate adjustments
    with no validation accuracy improvement before stopping training.

    Attributes:
        image_dir: Directory with images used for training.
        job_dir: Directory with train_samples.json, val_samples.json,
                 and class_mapping.json.
        epochs_train_dense: Maximum number of epochs to train dense layers (default 100).
        epochs_train_all: Maximum number of epochs to train all layers (default 100).
        learning_rate_dense: Learning rate for dense training phase (default 0.001).
        learning_rate_all: Learning rate for all training phase (default 0.0003).
        batch_size: Number of images per batch (default 64).
        dropout_rate: Fraction of nodes before output layer set to random value (default 0.75).
        base_model_name: Name of pretrained CNN (default MobileNet).
    """

    def __init__(
        self,
        image_dir: str,
        job_dir: str,
        epochs_train_dense: typing.Union[int, str] = EPOCHS_TRAIN_DENSE,
        epochs_train_all: typing.Union[int, str] = EPOCHS_TRAIN_ALL,
        learning_rate_dense: typing.Union[float, str] = LEARNING_RATE_DENSE,
        learning_rate_all: typing.Union[float, str] = LEARNING_RATE_ALL,
        batch_size: typing.Union[int, str] = BATCH_SIZE,
        dropout_rate: typing.Union[float, str] = DROPOUT_RATE,
        base_model_name: str = BASE_MODEL_NAME,
        loss: str = LOSS,
        **kwargs,
    ) -> None:
        """Inits training component.

        Checks whether multiprocessing is available and sets number of workers for training.
        """
        self.image_dir = Path(image_dir).resolve()
        self.job_dir = Path(job_dir).resolve()

        self.logger = get_logger(__name__, self.job_dir)
        self.samples_train = load_json(self.job_dir / 'train_samples.json')
        self.samples_val = load_json(self.job_dir / 'val_samples.json')
        self.class_mapping = load_json(self.job_dir / 'class_mapping.json')
        self.n_classes = len(self.class_mapping)

        self.epochs_train_dense = int(epochs_train_dense)
        self.epochs_train_all = int(epochs_train_all)
        self.learning_rate_dense = float(learning_rate_dense)
        self.learning_rate_all = float(learning_rate_all)
        self.batch_size = int(batch_size)
        self.dropout_rate = float(dropout_rate)
        self.base_model_name = base_model_name
        self.loss = loss
        self.use_multiprocessing, self.workers = use_multiprocessing()

    def _set_patience(self):
        """Adjust patience for early stopping and learning rate schedule
        based on training set size.
        """
        n_per_class = int(len(self.samples_train) / self.n_classes)

        self.patience_learning_rate = 5
        if n_per_class >= 200:
            self.patience_learning_rate = 4

        if n_per_class >= 500:
            self.patience_learning_rate = 2

        self.patience_early_stopping = 3 * self.patience_learning_rate

        self.logger.info('Early stopping patience: {}'.format(self.patience_early_stopping))
        self.logger.info('Learning rate patience: {}'.format(self.patience_learning_rate))

    def _build_model(self):
        self.classifier = ImageClassifier(
            self.base_model_name,
            self.n_classes,
            self.learning_rate_dense,
            self.dropout_rate,
            self.loss,
        )
        self.classifier.build()

    def _fit_model(self):
        training_generator = TrainDataGenerator(
            self.samples_train,
            self.image_dir,
            self.batch_size,
            self.n_classes,
            self.classifier.get_preprocess_input(),
        )

        validation_generator = ValDataGenerator(
            self.samples_val,
            self.image_dir,
            self.batch_size,
            self.n_classes,
            self.classifier.get_preprocess_input(),
        )

        # TODO: initialize callbacks TensorBoardBatch
        # tensorboard = TensorBoardBatch(log_dir=Path(job_dir).resolve() / 'logs')

        model_save_name = (
            'model_' + self.base_model_name.lower() + '_{epoch:02d}_{val_acc:.3f}.hdf5'
        )
        model_dir = self.job_dir / 'models'
        model_dir.mkdir(parents=True, exist_ok=True)

        logging_metrics = LoggingMetrics(logger=self.logger)
        logging_models = LoggingModels(
            logger=self.logger,
            filepath=str(model_dir / model_save_name),
            monitor='val_acc',
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
        )

        def _train_dense_layers():
            if self.epochs_train_dense > 0:
                self.logger.info('\n****** Train dense layers ******\n')

                min_lr = self.learning_rate_dense / 10
                reduce_lr = ReduceLROnPlateau(
                    monitor='val_acc',
                    factor=0.3162,
                    patience=self.patience_learning_rate,
                    min_lr=min_lr,
                    verbose=1,
                )

                early_stopping = EarlyStopping(
                    monitor='val_acc',
                    min_delta=0.002,
                    patience=self.patience_early_stopping,
                    verbose=1,
                    mode='auto',
                    baseline=None,
                    restore_best_weights=True,
                )

                # freeze convolutional layers in base net
                for layer in self.classifier.get_base_layers():
                    layer.trainable = False

                self.classifier.compile()
                # self.classifier.summary()

                self.hist_dense = self.classifier.fit_generator(
                    generator=training_generator,
                    validation_data=validation_generator,
                    epochs=self.epochs_train_dense,
                    verbose=1,
                    use_multiprocessing=self.use_multiprocessing,
                    workers=self.workers,
                    max_queue_size=30,
                    callbacks=[logging_metrics, logging_models, reduce_lr, early_stopping],
                )

        def _train_all_layers():
            if self.epochs_train_all > 0:
                self.logger.info('\n****** Train all layers ******\n')

                min_lr = self.learning_rate_all / 10
                reduce_lr = ReduceLROnPlateau(
                    monitor='val_acc',
                    factor=0.3162,
                    patience=self.patience_learning_rate,
                    min_lr=min_lr,
                    verbose=1,
                )

                early_stopping = EarlyStopping(
                    monitor='val_acc',
                    min_delta=0.002,
                    patience=self.patience_early_stopping,
                    verbose=1,
                    mode='auto',
                    baseline=None,
                    restore_best_weights=False,
                )

                # unfreeze all layers
                for layer in self.classifier.get_base_layers():
                    layer.trainable = True

                self.classifier.set_learning_rate(self.learning_rate_all)

                self.classifier.compile()
                # self.classifier.summary()

                self.hist_all = self.classifier.fit_generator(
                    generator=training_generator,
                    validation_data=validation_generator,
                    epochs=self.epochs_train_dense + self.epochs_train_all,
                    initial_epoch=self.epochs_train_dense,
                    verbose=1,
                    use_multiprocessing=self.use_multiprocessing,
                    workers=self.workers,
                    max_queue_size=30,
                    callbacks=[logging_metrics, logging_models, reduce_lr, early_stopping],
                )

        self._set_patience()
        _train_dense_layers()
        _train_all_layers()

        K.clear_session()

    def run(self):
        """Builds the model and runs training.
        """
        self._build_model()
        self._fit_model()
