import pytest
from pathlib import Path
from imageatm.components.training import Training
from imageatm.handlers.data_generator import TrainDataGenerator, ValDataGenerator
from imageatm.handlers.image_classifier import ImageClassifier

p = Path(__file__)

TEST_IMAGE_DIR = p.resolve().parent / '../data/test_images'
TEST_JOB_DIR = p.resolve().parent / '../data/test_train_job'


@pytest.fixture(autouse=True)
def common_patches(mocker):
    mocker.patch('imageatm.handlers.image_classifier.ImageClassifier.__init__')
    mocker.patch('imageatm.handlers.image_classifier.ImageClassifier.build')
    mocker.patch('imageatm.handlers.image_classifier.ImageClassifier.compile')
    mocker.patch('imageatm.handlers.image_classifier.ImageClassifier.fit_generator')
    mocker.patch('imageatm.handlers.image_classifier.ImageClassifier.summary')
    mocker.patch('imageatm.handlers.image_classifier.ImageClassifier.get_preprocess_input')
    mocker.patch('imageatm.handlers.image_classifier.ImageClassifier.get_base_layers')
    ImageClassifier.__init__.return_value = None

    mocker.patch('imageatm.handlers.data_generator.TrainDataGenerator.__init__')
    mocker.patch('imageatm.handlers.data_generator.ValDataGenerator.__init__')
    TrainDataGenerator.__init__.return_value = None
    ValDataGenerator.__init__.return_value = None


class TestTraining(object):
    train = None

    def test__init(self):
        global train
        train = Training(image_dir=TEST_IMAGE_DIR, job_dir=TEST_JOB_DIR)

        assert train.n_classes == 2
        assert train.epochs_train_dense == 2

    def test__build_model(self):
        global train
        train._build_model()

        ImageClassifier.__init__.assert_called_once_with(
            'MobileNet', 2, 0.001, 0.75, 'categorical_crossentropy'
        )
        ImageClassifier.build.assert_called_once_with()

    def test__fit_model(self):
        global train
        train._fit_model()

        TrainDataGenerator.__init__.assert_called_once()
        ValDataGenerator.__init__.assert_called_once()
        ImageClassifier.get_preprocess_input.assert_called_with()

        # TODO: this tests are only rudimentary
        ImageClassifier.compile.assert_called()
        ImageClassifier.fit_generator.assert_called()
        ImageClassifier.get_base_layers.assert_called()
