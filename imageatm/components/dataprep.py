from typing import Counter as Counter_type
from typing import Optional, List, Callable
from collections import Counter
from pathlib import Path
from imageatm.utils.images import resize_image_mp, validate_image
from imageatm.utils.io import load_json, save_json
from imageatm.utils.process import parallelise
from imageatm.utils.logger import get_logger
from sklearn.model_selection import train_test_split


MIN_CLASS_SIZE = 2
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.2
PART_SIZE = 1.0


class DataPrep:
    """Prepares data for training.

    Based on a *samples file* and an *image directory* a train, validation, and test set will be created.
    Before data preparation starts samples file and image directory will be validated.

    DataPrep creates a job directory with the following set of files

    - class_mapping.json
    - train_samples.json
    - val_samples.json
    - test_samples.json

    These files are required for subsequent components (training and evaluation).


    The samples file must be in JSON format with the following keys

    [
    {
        "image_id": "image_1.jpg",
        "label": "Class 1"
    },
    {
        "image_id": "image_2.jpg",
        "label": "Class 1"
    },
    ...
    ]



    Attributes:
        image_dir: path of image directory.
        job_dir: path to job directory with samples.
        image_dir: path of image directory.
        samples_file: path to samples file.
        min_class_size: minimal number of samples per label (default 2).
        test_size: represent the proportion of the dataset to include in the test set (default 0.2).
        val_size: represent the proportion of the dataset to include in the val set (default 0.1).
        part_size: represent the proportion of the dataset to include in all sets (default 1).
    """

    def __init__(
        self,
        job_dir: str,
        image_dir: str,
        samples_file: str,
        min_class_size: int = MIN_CLASS_SIZE,
        test_size: float = TEST_SIZE,
        val_size: float = VALIDATION_SIZE,
        part_size: float = PART_SIZE,
        **kwargs
    ) -> None:
        """Inits data preparation component.

        Loads samples file. Initializes variables for further operations:
        *valid_image_ids*, *class_mapping*, *train_samples*, *val_samples*, *test_samples*.
        """
        self.job_dir = Path(job_dir).resolve()
        self.job_dir.mkdir(parents=True, exist_ok=True)

        self.image_dir = Path(image_dir)
        self.samples_file = Path(samples_file)
        self.samples_file = Path(samples_file)
        self.min_class_size = min_class_size
        self.test_size = test_size
        self.val_size = val_size
        self.part_size = part_size

        self.class_mapping: Optional[dict] = None
        self.valid_image_ids: Optional[List[str]] = None
        self.train_samples = None
        self.val_samples = None
        self.test_samples = None

        self.logger = get_logger(__name__, self.job_dir)
        self.samples = load_json(self.samples_file)

    def _get_counter(self, list_to_count: list, print_count: bool = False) -> Counter_type:
        """Retrieves samples count for each label from loaded samples and prints distribution when parameter is set."""
        counter = Counter(list_to_count)
        total = len(list_to_count)

        if print_count:
            for key, val in sorted(counter.items()):
                self.logger.info('{}: {} ({}%)'.format(key, val, round(val * 100 / total, 1)))

        return counter

    def _validate_images(self):
        """Checks if all files in *image_dir* are valid images.

        Checks if files are images with extention 'JPEG' or 'PNG' and if they are not truncated.
        self.logger.infos filenames that didn't pass the validation.

        Sets:
            self.valid_image_ids: a list of valid image.
        """
        self.logger.info('\n****** Running image validation ******\n')

        # validate images, use multiprocessing
        files = [str(i.absolute()) for i in self.image_dir.glob('*')]
        files.sort()

        results = parallelise(validate_image, files)

        valid_image_files = [j for i, j in enumerate(files) if results[i][0]]
        self.valid_image_ids = [Path(i).name for i in valid_image_files]

        # return list of invalid images to user and save them if there are more than 10
        invalid_image_files = [
            (j, str(results[i][1])) for i, j in enumerate(files) if not results[i][0]
        ]

        if invalid_image_files:
            self.logger.info('The following files are not valid image files:')
            for file_name, error_msg in invalid_image_files[:10]:
                self.logger.info('- {} ({})'.format(file_name, error_msg))
            if len(invalid_image_files) > 10:
                save_json(invalid_image_files, self.job_dir / 'invalid_image_files.json')
                self.logger.info(
                    (
                        'NOTE: More than 10 files were identified as invalid image files.\n'
                        'The full list of those files has been saved here:\n{}'.format(
                            self.job_dir / 'invalid_image_files.json'
                        )
                    )
                )

    @staticmethod
    def _validate_sample(sample: dict, valid_image_ids: list) -> bool:
        return all(
            [
                ('label' in sample),
                ('image_id' in sample),
                (sample.get('image_id') in valid_image_ids),
            ]
        )

    def _validate_samples(self):
        """Validates the samples.

        Compares samples with valid image ids. Checks if:

            - samples have 'label' and 'image_id' keys
            - there is more than 1 class
            - there are enough samples in each class

        Reassigns self.samples to valid samples.
        Sets counter of each class in samples.

        self.logger.infos:
            - samples that didn't pass the validation
            - distribution of valid samples for each label
        """
        self.logger.info('\n****** Running samples validation ******\n')

        assert self.valid_image_ids, 'Images have to be validated before samples validation.'
        valid_image_ids = set(self.valid_image_ids)  # convert to set to optimise lookup

        # exclude samples with invalid image or invalid keys
        valid_samples = []
        self.invalid_samples = []
        for sample in self.samples:
            if self._validate_sample(sample, valid_image_ids):
                valid_samples.append(sample)
            else:
                self.invalid_samples.append(sample)

        # replace self.samples with valid samples
        assert valid_samples, 'Program ended. No valid samples found.'
        self.samples = valid_samples

        # self.logger.infos invalid images to user and saves them in json if there are more than 10
        if self.invalid_samples:
            self.logger.info('The following samples were dropped:')
            for sample in self.invalid_samples[:10]:
                self.logger.info('- {}'.format(sample))

            if len(self.invalid_samples) > 10:
                self.logger.info(
                    (
                        'NOTE: {} samples were identified as invalid.\n'
                        'The full list of invalid samples will be saved in job dir.\n'.format(
                            len(self.invalid_samples)
                        )
                    )
                )

        self.logger.info('Class distribution after validation:')
        self.samples_count = self._get_counter([i['label'] for i in self.samples], print_count=True)

        # check if each class has sufficient samples
        warnings_count = 0
        for key, val in self.samples_count.items():
            if val < self.min_class_size:
                warnings_count += 1
                self.logger.info('Not enough samples for label {}'.format(key))

        assert warnings_count == 0, 'Program ended. Collect more samples.'
        assert len(self.samples_count) > 1, 'Program ended. Only one label in the dataset.'

    def _create_class_mapping(self, print_mapping: bool = True):
        """Produces a class-mapping."""
        self.class_mapping = {}
        labels = list(self.samples_count.keys())
        labels.sort()

        for i, j in enumerate(labels):
            self.class_mapping[i] = j

        if print_mapping:
            self.logger.info('Class mapping:\n{}'.format(self.class_mapping))

    def _apply_class_mapping(self):
        """Applies the class-mapping."""
        class_mapping_inv = {v: k for k, v in self.class_mapping.items()}

        samples_int = [
            {'label': class_mapping_inv[sample['label']], 'image_id': sample['image_id']}
            for sample in self.samples
        ]

        # replace self.samples with samples int
        self.samples = samples_int

    def _split_samples(self):
        """Produces stratified train, val, test sets.

        Sets:
            self.train_samples
            self.val_samples
            self.test_samples
        """
        self.logger.info('\n****** Creating train/val/test sets ******\n')

        self.train_size = 1 - (self.test_size + self.val_size)

        self.logger.info(
            'Split distribution: train: {:.2f}, val: {}, test: {:}\n'.format(
                self.train_size, self.val_size, self.test_size
            )
        )

        self.logger.info(
            'Partial split distribution: train: {:.2f}, val: {:.2f}, test: {:.2f}\n'.format(
                self.train_size * self.part_size,
                self.val_size * self.part_size,
                self.test_size * self.part_size,
            )
        )

        labels = [i['label'] for i in self.samples]  # need label list for stratification

        split_test_size = self.test_size * self.part_size
        split_train_size = (1 - self.test_size) * self.part_size

        train_samples, self.test_samples, train_labels, _ = train_test_split(
            self.samples,
            labels,
            test_size=split_test_size,
            train_size=split_train_size,
            shuffle=True,
            random_state=10207,
            stratify=labels,
        )

        split_test_size = self.val_size / (1 - self.test_size)
        self.train_samples, self.val_samples, _, _ = train_test_split(
            train_samples,
            train_labels,
            test_size=split_test_size,
            shuffle=True,
            random_state=10207,
            stratify=train_labels,
        )

        self.logger.info('Train set:')
        self._get_counter([i['label'] for i in self.train_samples], print_count=True)

        self.logger.info('Val set:')
        self._get_counter([i['label'] for i in self.val_samples], print_count=True)

        self.logger.info('Test set:')
        self._get_counter([i['label'] for i in self.test_samples], print_count=True)

    def _resize_images(self, resize_image_mp: Callable = resize_image_mp):
        self.logger.info('\n****** Resizing images ******\n')
        new_image_dir = '_'.join([str(self.image_dir), 'resized'])
        Path(new_image_dir).resolve().mkdir(parents=True, exist_ok=True)

        args = [(self.image_dir, new_image_dir, i['image_id']) for i in self.samples]
        parallelise(resize_image_mp, args)
        self.logger.info(
            'Stored {} resized images under {}'.format(len(self.samples), new_image_dir)
        )

        self.image_dir = Path(new_image_dir)
        self.logger.info('Changed image directory to {}'.format(self.image_dir))

    def _save_files(self):
        save_json(self.train_samples, self.job_dir / 'train_samples.json')
        save_json(self.val_samples, self.job_dir / 'val_samples.json')
        save_json(self.test_samples, self.job_dir / 'test_samples.json')
        save_json(self.class_mapping, self.job_dir / 'class_mapping.json')

        if len(self.invalid_samples) > 10:
            save_json(self.invalid_samples, self.job_dir / 'invalid_samples.json')
            self.logger.info(
                (
                    'NOTE: More than 10 samples were identified as invalid.\n'
                    'The full list of invalid samples has been saved here:\n{}'.format(
                        self.job_dir / 'invalid_samples.json'
                    )
                )
            )

    def run(self, resize: bool = False):
        """Executes all steps of data preparation.

            - Validates samples and images
            - Creates class-mapping (string to integer)
            - Applies class-mapping on samples
            - Splits sample into train, validation and test sets
            - Resizes images
            - Saves files (class-mapping, train-, validation- and test-set)

        Args:
            resize: boolean (creates a subfolder of resized images, default False).
        """
        self._validate_images()
        self._validate_samples()

        self._create_class_mapping()
        self._apply_class_mapping()
        self._split_samples()

        if resize:
            self._resize_images()

        self._save_files()
