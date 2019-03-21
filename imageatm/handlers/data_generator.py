import numpy as np
from typing import List, Callable, Tuple
from pathlib import Path
from imageatm.utils.images import load_image, random_crop
from keras.utils import Sequence, to_categorical


class DataGenerator(Sequence):
    """Class inherits from Keras Sequence base object, allows to use multiprocessing in .fit_generator.

    DataGenerator is extended by these classes:
        - TrainDataGenerator
        - ValDataGenerator.

    Attributes:
        samples: Dictionary of samples to generate data from.
        image_dir: Path of image directory.
        batch_size: Number of images per batch.
        n_classes: Number of classes in dataset.
        basenet_preprocess: Basenet specific preprocessing function.
        img_load_dims: Dimensions that images get resized into when loaded.
        train: If set to True samples are shuffled before each epoch and images are cropped once.
    """

    def __init__(
        self,
        samples: List[dict],
        image_dir: str,
        batch_size: int,
        n_classes: int,
        basenet_preprocess: Callable,
        img_load_dims: Tuple[int, int],
        train: bool,
    ) -> None:
        """Inits DataGenerator object.

        If *train* set *True* then samples are shuffled on init.
        """
        self.samples = samples
        self.image_dir = Path(image_dir)
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.basenet_preprocess = basenet_preprocess
        self.img_load_dims = img_load_dims
        self.train = train
        self.on_epoch_end()

    def on_epoch_end(self):
        """Method called at the end of every epoch.

        If *train* set *True* then samples are shuffled.
        """
        self.indexes = np.arange(len(self.samples))
        if self.train is True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        """Number of batches in the Sequence."""
        return int(np.ceil(len(self.samples) / self.batch_size))

    def __getitem__(self, index: int) -> Tuple[np.array, np.array]:
        """Gets batch at position `index`.

        If *train* set *True* then images will be cropped by *img_crop_dims*.
        """
        batch_indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        batch_samples = [self.samples[i] for i in batch_indexes]
        X, y = self._data_generator(batch_samples)
        return X, y

    def _data_generator(self, batch_samples: List[dict]) -> Tuple[np.array, np.array]:
        """Generates data from samples in specified batch."""
        #  initialize images and labels tensors for faster processing
        dims = self.img_crop_dims if self.train == True else self.img_load_dims
        X = np.empty((len(batch_samples), *dims, 3))
        y = np.empty((len(batch_samples), self.n_classes))

        for i, sample in enumerate(batch_samples):
            # load and randomly augment image
            img_file = self.image_dir / sample['image_id']
            img = np.asarray(load_image(img_file, self.img_load_dims))
            if self.train == True:
                img = random_crop(img, self.img_crop_dims)
            X[i,] = img

            # TODO: more efficient by preprocessing
            y[i,] = to_categorical([sample['label']], num_classes=self.n_classes)

        # apply basenet specific preprocessing
        # input is 4D numpy array of RGB values within [0, 255]
        X = self.basenet_preprocess(X)

        return X, y


class TrainDataGenerator(DataGenerator):
    """Class inherits from DataGenerator.

    Per default images will be cropped and samples are shuffled before each epoch.

    Attributes:
        samples: Dictionary of samples to generate data from.
        image_dir: Path of image directory.
        batch_size: Number of images per batch.
        n_classes: Number of classes in dataset.
        basenet_preprocess: Basenet specific preprocessing function.
        img_load_dims: Dimensions that images get resized into when loaded (default (256, 256)).
        img_crop_dims: Dimensions that images get resized into when loaded (default (224, 224)).
        train: If set to True samples are shuffled before each epoch and images are cropped once (default True).
    """

    def __init__(
        self,
        samples: List[dict],
        image_dir: str,
        batch_size: int,
        n_classes: int,
        basenet_preprocess: Callable,
        img_load_dims: Tuple[int, int] = (256, 256),
        img_crop_dims: Tuple[int, int] = (224, 224),
        train: bool = True,
    ) -> None:
        """Inits TrainDataGenerator object.

        Per default samples are shuffled on init."""
        super(TrainDataGenerator, self).__init__(
            samples, image_dir, batch_size, n_classes, basenet_preprocess, img_load_dims, train
        )
        self.img_crop_dims = img_crop_dims  # dimensions that images get randomly cropped to


class ValDataGenerator(DataGenerator):
    """Class inherits from DataGenerator.

    Per default neither images are cropped nor samples are shuffled.

    Attributes:
        samples: Dictionary of samples to generate data from.
        image_dir: Path of image directory.
        batch_size: Number of images per batch.
        n_classes: Number of classes in dataset.
        basenet_preprocess: Basenet specific preprocessing function.
        img_load_dims: Dimensions that images get resized into when loaded (default (224, 224)).
        train: If set to True samples are shuffled before each epoch and images are cropped once (default False).
    """

    def __init__(
        self,
        samples: List[dict],
        image_dir: str,
        batch_size: int,
        n_classes: int,
        basenet_preprocess: Callable,
        img_load_dims: Tuple[int, int] = (224, 224),
        train: bool = False,
    ) -> None:
        """Inits TrainDataGenerator object."""
        super(ValDataGenerator, self).__init__(
            samples, image_dir, batch_size, n_classes, basenet_preprocess, img_load_dims, train
        )
