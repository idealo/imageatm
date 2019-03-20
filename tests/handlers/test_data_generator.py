import pytest
import numpy as np
from pathlib import Path
from imageatm.handlers.data_generator import TrainDataGenerator, ValDataGenerator

TEST_CONFIG = {'batch_size': 3, 'n_classes': 2}

TEST_SAMPLES = [
    {'image_id': 'helmet_1.jpg', 'label': 0},
    {'image_id': 'helmet_2.jpg', 'label': 0},
    {'image_id': 'helmet_3.jpg', 'label': 0},
    {'image_id': 'helmet_4.jpg', 'label': 1},
    {'image_id': 'helmet_5.jpg', 'label': 1},
]


@pytest.fixture(autouse=True)
def common_patches(mocker):
    mocker.patch.object(TrainDataGenerator, 'data_generator')
    TrainDataGenerator.data_generator.return_value = 'X', 'y'

    mocker.patch.object(ValDataGenerator, 'data_generator')
    ValDataGenerator.data_generator.return_value = 'X', 'y'
    # Setting seed for np.random.shuffle in Train-mode
    np.random.seed(10247)


class TestTrainDataGenerator(object):
    generator = None

    def test__init(self):
        global generator
        generator = TrainDataGenerator(
            TEST_SAMPLES,
            'image_dir',
            TEST_CONFIG['batch_size'],
            TEST_CONFIG['n_classes'],
            'preprocess_input',
        )

        assert generator.samples == TEST_SAMPLES
        assert generator.image_dir == Path('image_dir')
        assert generator.batch_size == 3
        assert generator.n_classes == 2
        assert generator.basenet_preprocess == 'preprocess_input'
        assert generator.img_load_dims == (256, 256)
        assert generator.img_crop_dims == (224, 224)
        assert generator.train is True

    def test__len(self):
        global generator
        x = generator.__len__()

        assert x == 2

    def test__get_item(self):
        global generator
        generator.__getitem__(1)

        generator.data_generator.assert_called_with(
            [{'image_id': 'helmet_2.jpg', 'label': 0}, {'image_id': 'helmet_5.jpg', 'label': 1}]
        )


class TestValDataGenerator(object):
    generator = None

    def test__init(self):
        global generator
        generator = ValDataGenerator(
            TEST_SAMPLES,
            'image_dir',
            TEST_CONFIG['batch_size'],
            TEST_CONFIG['n_classes'],
            'preprocess_input',
        )

        assert generator.samples == TEST_SAMPLES
        assert generator.image_dir == Path('image_dir')
        assert generator.batch_size == 3
        assert generator.n_classes == 2
        assert generator.basenet_preprocess == 'preprocess_input'
        assert generator.img_load_dims == (224, 224)
        assert generator.train is False
        assert not hasattr(generator, 'img_crop_dims')

    def test__len(self):
        global generator
        assert generator.__len__() == 2

    def test__get_item(self):
        global generator
        generator.__getitem__(1)

        generator.data_generator.assert_called_with(
            [{'image_id': 'helmet_4.jpg', 'label': 1}, {'image_id': 'helmet_5.jpg', 'label': 1}]
        )
