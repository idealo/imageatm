import shutil
import pytest
import numpy.testing as npt
from collections import Counter
from pathlib import Path
from imageatm.components.data_prep import DataPrep
from imageatm.handlers.utils import load_json

p = Path(__file__)
"""Files for test_valid_images."""
INVALID_IMG_PATH = p.parent / '../data/test_images' / 'image_invalid.jpg'

VALID_IMG_PATH = p.parent / '../data/test_images' / 'image_960x540.jpg'

"""Files for sample validation."""
TEST_STR_FILE = p.parent / '../data/test_samples' / 'test_str_labels.json'

TEST_INT_FILE = p.parent / '../data/test_samples' / 'test_samples_int.json'

TEST_FILE_STR2INT = p.parent / '../data/test_samples' / 'test_int_labels.json'

TEST_IMG_DIR = p.parent / '../data/test_images'

TEST_SPLIT_FILE = p.parent / '../data/test_samples' / 'test_split.json'

TEST_STR_FILE_CORRUPTED = p.parent / '../data/test_samples' / 'test_str_labels_corrupted.json'

TEST_JOB_DIR = p.parent / 'test_job_dir'


@pytest.fixture(scope='session', autouse=True)
def tear_down(request):
    def remove_job_dir():
        shutil.rmtree(TEST_JOB_DIR)

    request.addfinalizer(remove_job_dir)


class TestDataPrep(object):
    dp = None

    def test__init(self, mocker):
        mocker.patch('imageatm.handlers.utils.load_json', return_value={})
        global dp
        dp = DataPrep(image_dir=TEST_IMG_DIR, job_dir=TEST_JOB_DIR, samples_file=TEST_STR_FILE)

        assert dp.image_dir == TEST_IMG_DIR
        assert dp.job_dir == TEST_JOB_DIR
        assert dp.samples_file == TEST_STR_FILE

    def test__validate_images(self):
        expected = [
            'helmet_1.jpg',
            'helmet_10.jpg',
            'helmet_2.jpg',
            'helmet_3.jpg',
            'helmet_4.jpg',
            'helmet_5.jpg',
            'helmet_6.jpg',
            'helmet_7.jpg',
            'helmet_8.jpg',
            'helmet_9.jpg',
            'image_960x540.jpg',
            'image_png.png',
        ]
        global dp
        dp._validate_images()

        assert sorted(dp.valid_image_ids) == expected

    def test__validate_sample(self):
        global dp

        valid_image_ids = ['helmet_1.jpg', 'helmet_2.jpg', 'image_png.png']

        sample = {'image_id': 'helmet_3.jpg', 'blabla': 'left'}
        result = dp._validate_sample(sample, valid_image_ids)
        assert result == False

        sample = {'blabla': 'helmet_1.jpg', 'label': 'right'}
        result = dp._validate_sample(sample, valid_image_ids)
        assert result == False

        sample = {'image_id': 'truncated.jpg', 'label': 'left'}
        result = dp._validate_sample(sample, valid_image_ids)
        assert result == False

        sample = {'image_id': 'helmet_1.jpg', 'label': 'left'}
        result = dp._validate_sample(sample, valid_image_ids)
        assert result == True

        sample = {'image_id': 'helmet_5.jpg', 'label': 'left'}
        result = dp._validate_sample(sample, valid_image_ids)
        assert result == False

    def test__validate_samples(self):
        global dp
        expected = load_json(TEST_STR_FILE)

        dp.min_class_size = 1
        dp.valid_image_ids = ['helmet_1.jpg', 'helmet_2.jpg', 'helmet_3.jpg', 'image_png.png']
        dp._validate_samples()

        assert dp.samples == expected
        assert dp.invalid_samples == []
        assert dp.samples_count == {'left': 3, 'right': 1}

        # exclude first 3 samples as they are corrupted
        expected = [
            {'image_id': 'helmet_1.jpg', 'label': 'left'},
            {'image_id': 'helmet_2.jpg', 'label': 'left'},
            {'image_id': 'image_png.png', 'label': 'right'},
        ]

        dp.valid_image_ids = ['helmet_1.jpg', 'helmet_2.jpg', 'image_png.png']
        dp.samples = load_json(TEST_STR_FILE_CORRUPTED)
        dp._validate_samples()

        assert dp.samples == expected

    def test__create_class_mapping(self):
        global dp
        dp.samples = load_json(TEST_STR_FILE)
        dp.samples_count = {
            'left': 3,
            'right': 1
        }
        dp._create_class_mapping()
        expected = {0: 'left', 1: 'right'}
        assert dp.class_mapping == expected

        dp.samples = dp.samples[::-1]
        dp._create_class_mapping()
        expected = {0: 'left', 1: 'right'}
        assert dp.class_mapping == expected

        dp.samples = load_json(TEST_INT_FILE)
        dp.samples_count = {
            1: 10,
            2: 20
        }
        dp._create_class_mapping()
        print(dp.class_mapping)
        expected = {0: 1, 1: 2}
        assert dp.class_mapping == expected

        dp.samples = dp.samples[::-1]
        dp._create_class_mapping()
        expected = {0: 1, 1: 2}
        assert dp.class_mapping == expected

    def test__apply_class_mapping(self):
        global dp
        dp.samples = load_json(TEST_STR_FILE)
        dp.class_mapping = {
            0: 'left',
            1: 'right'
        }
        dp._apply_class_mapping()
        expected = load_json(TEST_FILE_STR2INT)

        assert dp.samples == expected

        dp.samples = [
            {'image_id': 'helmet_2.jpg', 'label': 'left'},
            {'image_id': 'image_png.png', 'label': 'right', 'test': 'abc'},
        ]
        expected = [
            {'image_id': 'helmet_2.jpg', 'label': 0},
            {'image_id': 'image_png.png', 'label': 1},
        ]

        dp._apply_class_mapping()

        assert dp.samples == expected

        dp.samples = [
            {'image_id': 'helmet_2.jpg', 'label': 'left'},
            {'image_id': 'image_png.png', 'label': 'right'},
        ]
        expected = [
            {'image_id': 'helmet_2.jpg', 'label': 1},
            {'image_id': 'image_png.png', 'label': 0},
        ]

        assert dp.samples != expected

        dp.samples = [
            {'image_id': 'helmet_2.jpg', 'label': 1},
            {'image_id': 'image_png.png', 'label': 0},
        ]
        expected = [
            {'image_id': 'helmet_2.jpg', 'label': 1},
            {'image_id': 'image_png.png', 'label': 0},
        ]

        assert dp.samples == expected

    def test_split_samples_full(self):
        global dp
        dp.samples = load_json(TEST_SPLIT_FILE)  ##100
        dp.test_size = 0.2
        dp.val_size = 0.5
        dp.part_size = 1.0

        dp._split_samples()

        npt.assert_almost_equal(dp.train_size, 0.3)
        assert dp.test_size + dp.val_size + dp.train_size == 1.0

        assert len(dp.test_samples) == 40
        assert len(dp.val_samples) == 100
        assert len(dp.train_samples) == 60

        train_labels_count = Counter([i['label'] for i in dp.train_samples])
        val_labels_count = Counter([i['label'] for i in dp.val_samples])
        test_labels_count = Counter([i['label'] for i in dp.test_samples])

        assert test_labels_count[1] == 20
        assert test_labels_count[2] == 12
        assert test_labels_count[3] == 8

        assert val_labels_count[1] == 50
        assert val_labels_count[2] == 30
        assert val_labels_count[3] == 20

        assert train_labels_count[1] == 30
        assert train_labels_count[2] == 18
        assert train_labels_count[3] == 12

    def test_split_samples_full_2(self):
        global dp
        dp.samples = load_json(TEST_SPLIT_FILE)
        dp.test_size = 0.2
        dp.val_size = 0.1
        dp.part_size = 1.0

        dp._split_samples()

        npt.assert_almost_equal(dp.train_size, 0.7)
        assert dp.test_size + dp.val_size + dp.train_size == 1.0

        assert len(dp.test_samples) == 40
        assert len(dp.val_samples) == 20
        assert len(dp.train_samples) == 140

        train_labels_count = Counter([i['label'] for i in dp.train_samples])
        val_labels_count = Counter([i['label'] for i in dp.val_samples])
        test_labels_count = Counter([i['label'] for i in dp.test_samples])

        assert test_labels_count[1] == 20
        assert test_labels_count[2] == 12
        assert test_labels_count[3] == 8

        assert val_labels_count[1] == 10
        assert val_labels_count[2] == 6
        assert val_labels_count[3] == 4

        assert train_labels_count[1] == 70
        assert train_labels_count[2] == 42
        assert train_labels_count[3] == 28

    def test_split_samples_half(self):
        global dp
        dp.samples = load_json(TEST_SPLIT_FILE)
        dp.test_size = 0.2
        dp.val_size = 0.5
        dp.part_size = 1 / 2

        dp._split_samples()

        npt.assert_almost_equal(dp.train_size, 0.3)
        assert dp.test_size + dp.val_size + dp.train_size == 1.0

        assert len(dp.test_samples) == 20
        assert len(dp.val_samples) == 50
        assert len(dp.train_samples) == 30

        train_labels_count = Counter([i['label'] for i in dp.train_samples])
        val_labels_count = Counter([i['label'] for i in dp.val_samples])
        test_labels_count = Counter([i['label'] for i in dp.test_samples])

        assert test_labels_count[1] == 10
        assert test_labels_count[2] == 6
        assert test_labels_count[3] == 4

        assert val_labels_count[1] == 25
        assert val_labels_count[2] == 15
        assert val_labels_count[3] == 10

        assert train_labels_count[1] == 15
        assert train_labels_count[2] == 9
        assert train_labels_count[3] == 6

    def test_split_samples_half_2(self):
        global dp
        dp.samples = load_json(TEST_SPLIT_FILE)
        dp.test_size = 0.2
        dp.val_size = 0.4
        dp.part_size = 2 / 3

        dp._split_samples()

        npt.assert_almost_equal(dp.train_size, 0.4)
        assert dp.test_size + dp.val_size + dp.train_size == 1.0

        assert len(dp.test_samples) == 27
        assert len(dp.val_samples) == 53
        assert len(dp.train_samples) == 53

        train_labels_count = Counter([i['label'] for i in dp.train_samples])
        val_labels_count = Counter([i['label'] for i in dp.val_samples])
        test_labels_count = Counter([i['label'] for i in dp.test_samples])

        assert test_labels_count[1] == 14
        assert test_labels_count[2] == 8
        assert test_labels_count[3] == 5

        assert val_labels_count[1] == 26
        assert val_labels_count[2] == 16
        assert val_labels_count[3] == 11

        assert train_labels_count[1] == 27
        assert train_labels_count[2] == 16
        assert train_labels_count[3] == 10
