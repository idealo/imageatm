import shutil
import pytest
import numpy.testing as npt
from collections import Counter
from mock import call
from pathlib import Path
from imageatm.components.dataprep import DataPrep
from imageatm.utils.io import load_json
from imageatm.utils.images import resize_image_mp

p = Path(__file__)
"""Files for test_valid_images."""
INVALID_IMG_PATH = p.parent / '../data/test_images' / 'image_invalid.jpg'

VALID_IMG_PATH = p.parent / '../data/test_images' / 'image_960x540.jpg'

"""Files for sample validation."""
TEST_STR_FILE = p.parent / '../data/test_samples' / 'test_str_labels.json'

TEST_INT_FILE = p.parent / '../data/test_samples' / 'test_samples_int.json'

TEST_FILE_STR2INT = p.parent / '../data/test_samples' / 'test_int_labels.json'

TEST_IMG_DIR = p.parent / '../data/test_images'

TEST_NO_IMG_DIR = p.parent / '../data/test_no_images'

TEST_SPLIT_FILE = p.parent / '../data/test_samples' / 'test_split.json'

TEST_STR_FILE_CORRUPTED = p.parent / '../data/test_samples' / 'test_str_labels_corrupted.json'

TEST_JOB_DIR = p.parent / 'test_job_dir'


@pytest.fixture(scope='class', autouse=True)
def tear_down(request):
    def remove_job_dir():
        shutil.rmtree(TEST_JOB_DIR)
        shutil.rmtree(p.parent / '../data/test_images_resized')

    request.addfinalizer(remove_job_dir)


class TestDataPrep(object):
    dp = None

    def test__init(self, mocker):
        mocker.patch('imageatm.utils.io.load_json', return_value={})
        global dp
        dp = DataPrep(image_dir=TEST_IMG_DIR, job_dir=TEST_JOB_DIR, samples_file=TEST_STR_FILE)

        assert dp.image_dir == TEST_IMG_DIR
        assert dp.job_dir == TEST_JOB_DIR
        assert dp.samples_file == TEST_STR_FILE

    def test__validate_images_1(self):
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

    def test__validate_images_2(self, mocker):
        mp_save_json = mocker.patch('imageatm.components.dataprep.save_json')

        global dp
        dp.image_dir = TEST_NO_IMG_DIR

        invalid_image_files = []
        files = [str(i.absolute()) for i in dp.image_dir.glob('*')]
        files.sort()
        for file in files:
            inv_tuple = (
                str(file),
                str(OSError("cannot identify image file '{}'".format(str(file)))),
            )
            invalid_image_files.append(inv_tuple)

        dp._validate_images()

        mp_save_json.assert_called_with(
            invalid_image_files, dp.job_dir / 'invalid_image_files.json'
        )

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

    def test__validate_samples_1(self):
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

    def test__validate_samples_2(self, mocker):
        mp_logger_info = mocker.patch('logging.Logger.info')

        global dp
        dp.valid_image_ids = ['1.jpg', '2.jpg', '3.jpg', '4.jpg']
        dp.samples = load_json(TEST_INT_FILE)
        dp._validate_samples()
        calls = [
            call('\n****** Running samples validation ******\n'),
            call('The following samples were dropped:'),
            call("- {'image_id': '5.jpg', 'label': 1}"),
            call("- {'image_id': '6.jpg', 'label': 1}"),
            call("- {'image_id': '7.jpg', 'label': 2}"),
            call("- {'image_id': '8.jpg', 'label': 1}"),
            call("- {'image_id': '9.jpg', 'label': 1}"),
            call("- {'image_id': '10.jpg', 'label': 2}"),
            call("- {'image_id': '11.jpg', 'label': 1}"),
            call("- {'image_id': '12.jpg', 'label': 1}"),
            call("- {'image_id': '13.jpg', 'label': 1}"),
            call("- {'image_id': '14.jpg', 'label': 2}"),
            call(
                'NOTE: 26 samples were identified as invalid.\n'
                'The full list of invalid samples will be saved in job dir.\n'
            ),
            call('Class distribution after validation:'),
            call('1: 2 (50.0%)'),
            call('2: 2 (50.0%)'),
        ]

        mp_logger_info.assert_has_calls(calls)

    def test__validate_samples_3(self):
        global dp

        dp.min_class_size = 1000
        dp.valid_image_ids = ['1.jpg', '2.jpg', '3.jpg', '4.jpg']

        with pytest.raises(AssertionError) as excinfo:
            dp._validate_samples()

        assert 'Program ended. Collect more samples.' in str(excinfo)

        dp.min_class_size = 1
        dp.valid_image_ids = ['2.jpg', '3.jpg']

        with pytest.raises(AssertionError) as excinfo:
            dp._validate_samples()

        assert 'Program ended. Only one label in the dataset.' in str(excinfo)

    def test__create_class_mapping(self):
        global dp
        dp.samples = load_json(TEST_STR_FILE)
        dp.samples_count = {'left': 3, 'right': 1}
        dp._create_class_mapping()
        expected = {0: 'left', 1: 'right'}
        assert dp.class_mapping == expected

        dp.samples = dp.samples[::-1]
        dp._create_class_mapping()
        expected = {0: 'left', 1: 'right'}
        assert dp.class_mapping == expected

        dp.samples = load_json(TEST_INT_FILE)
        dp.samples_count = {1: 10, 2: 20}
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
        dp.class_mapping = {0: 'left', 1: 'right'}
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

    def test_run_1(self, mocker):
        mp_validate_images = mocker.patch('imageatm.components.dataprep.DataPrep._validate_images')
        mp_validate_samples = mocker.patch(
            'imageatm.components.dataprep.DataPrep._validate_samples'
        )
        mp_create_class_mapping = mocker.patch(
            'imageatm.components.dataprep.DataPrep._create_class_mapping'
        )
        mp_apply_class_mapping = mocker.patch(
            'imageatm.components.dataprep.DataPrep._apply_class_mapping'
        )
        mp_split_samples = mocker.patch('imageatm.components.dataprep.DataPrep._split_samples')
        mp_resize_images = mocker.patch('imageatm.components.dataprep.DataPrep._resize_images')
        mp_save_files = mocker.patch('imageatm.components.dataprep.DataPrep._save_files')

        global dp
        dp.image_dir == TEST_IMG_DIR
        dp.job_dir == TEST_JOB_DIR
        dp.samples_file == TEST_STR_FILE

        dp.run(resize=False)

        mp_validate_images.assert_called_once()
        mp_validate_samples.assert_called_once()
        mp_create_class_mapping.assert_called_once()
        mp_apply_class_mapping.assert_called_once()
        mp_split_samples.assert_called_once()
        mp_resize_images.assert_not_called()
        mp_save_files.assert_called_once()

    def test_run_2(self, mocker):
        mp_validate_images = mocker.patch('imageatm.components.dataprep.DataPrep._validate_images')
        mp_validate_samples = mocker.patch(
            'imageatm.components.dataprep.DataPrep._validate_samples'
        )
        mp_create_class_mapping = mocker.patch(
            'imageatm.components.dataprep.DataPrep._create_class_mapping'
        )
        mp_apply_class_mapping = mocker.patch(
            'imageatm.components.dataprep.DataPrep._apply_class_mapping'
        )
        mp_split_samples = mocker.patch('imageatm.components.dataprep.DataPrep._split_samples')
        mp_resize_images = mocker.patch('imageatm.components.dataprep.DataPrep._resize_images')
        mp_save_files = mocker.patch('imageatm.components.dataprep.DataPrep._save_files')

        global dp
        dp.run(resize=True)

        mp_validate_images.assert_called_once()
        mp_validate_samples.assert_called_once()
        mp_create_class_mapping.assert_called_once()
        mp_apply_class_mapping.assert_called_once()
        mp_split_samples.assert_called_once()
        mp_resize_images.assert_called_once()
        mp_save_files.assert_called_once()

    def test__save_files_1(self, mocker):
        mp_save_json = mocker.patch('imageatm.components.dataprep.save_json')

        global dp
        dp.job_dir = TEST_JOB_DIR

        dp._save_files()

        calls = [
            call(dp.train_samples, dp.job_dir / 'train_samples.json'),
            call(dp.val_samples, dp.job_dir / 'val_samples.json'),
            call(dp.test_samples, dp.job_dir / 'test_samples.json'),
            call(dp.class_mapping, dp.job_dir / 'class_mapping.json'),
        ]

        mp_save_json.assert_has_calls(calls)

    def test__save_files_2(self, mocker):
        mp_save_json = mocker.patch('imageatm.components.dataprep.save_json')
        mp_logger_info = mocker.patch('logging.Logger.info')

        global dp
        dp.job_dir = TEST_JOB_DIR
        dp.invalid_samples = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

        dp._save_files()

        calls = [
            call(dp.train_samples, dp.job_dir / 'train_samples.json'),
            call(dp.val_samples, dp.job_dir / 'val_samples.json'),
            call(dp.test_samples, dp.job_dir / 'test_samples.json'),
            call(dp.class_mapping, dp.job_dir / 'class_mapping.json'),
            call(dp.invalid_samples, dp.job_dir / 'invalid_samples.json'),
        ]

        mp_save_json.assert_has_calls(calls)
        mp_logger_info.assert_called_with(
            'NOTE: More than 10 samples were identified as invalid.\n'
            'The full list of invalid samples has been saved here:\n{}'.format(
                dp.job_dir / 'invalid_samples.json'
            )
        )

    def test__resize_images(self, mocker):
        mp_parallelise = mocker.patch('imageatm.components.dataprep.parallelise')

        global dp
        dp.image_dir = TEST_IMG_DIR
        dp.job_dir = TEST_JOB_DIR
        dp.samples_file = TEST_STR_FILE
        print(dp.image_dir)
        new_image_dir = '_'.join([str(dp.image_dir), 'resized'])
        args = [(dp.image_dir, new_image_dir, i['image_id']) for i in dp.samples]

        dp._resize_images()

        mp_parallelise.assert_called_with(resize_image_mp, args)
        assert str(dp.image_dir) == new_image_dir
