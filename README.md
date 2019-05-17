# Image ATM (Automated Tagging Machine)

[![Build Status](https://travis-ci.org/idealo/imageatm.svg?branch=master)](https://travis-ci.org/idealo/imageatm)
[![License](https://img.shields.io/badge/License-Apache%202.0-orange.svg)](https://github.com/idealo/imageatm/blob/master/LICENSE)

Image ATM is a one-click tool that automates the workflow of a typical image classification pipeline in an opinionated way, this includes:

- Preprocessing and validating input images and labels
- Starting/terminating cloud instance with GPU support
- Training
- Model evaluation

Read the documentation at: [https://idealo.github.io/imageatm/](https://idealo.github.io/imageatm/)

Image ATM is compatible with Python 3.6 and is distributed under the Apache 2.0 license.

## Installation
There are two ways to install Image ATM:

* Install Image ATM from PyPI (recommended):
```
pip install imageatm
```

* Install Image ATM from the GitHub source:
```
git clone https://github.com/idealo/imageatm.git
cd imageatm
python setup.py install
```

## Usage

#### Train with CLI
Run this in your terminal
```
imageatm pipeline config/config_file.yml
```

#### Train without CLI
Run the data preparation:
``` python
from imageatm.components import DataPrep

dp = DataPrep(
    samples_file = 'sample_configfile.json',
    image_dir = 'sample_dataset/',
    job_dir='sample_jobdir/'
)
dp.run(resize=True)
```

Run the training:
``` python
from imageatm.components import Training

trainer = Training(image_dir=dp.image_dir, job_dir=dp.job_dir)
trainer.run()
```

Run the evaluation:
``` python
from imageatm.components import Evaluation

evaluator = Evaluation(image_dir=dp.image_dir, job_dir=dp.job_dir)
evaluator.run()
```

## Transfer learning
The following pretrained CNNs from Keras can be used for transfer learning in Image-ATM:

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
trained, and then all layers are trained.

For each phase the **learning rate is reduced** after a patience period if no
improvement in validation accuracy has been observed. The patience period
depends on the average number of samples per class (*n_per_class*):

- if *n_per_class* < 200: patience = 5 epochs
- if *n_per_class* >= 200 and < 500: patience = 4 epochs
- if *n_per_class* >= 500: patience = 2 epochs

**Training is stopped early** after a patience period that is three times
the learning rate patience to allow for two learning rate adjustments
before stopping training.

## Cite this work
Please cite Image ATM in your publications if this is useful for your research. Here is an example BibTeX entry:
```
@misc{idealods2019imageatm,
  title={Image ATM},
  author={Christopher Lennan and Malgorzata Adamczyk and Gunar Maiwald and Dat Tran},
  year={2019},
  howpublished={\url{https://github.com/idealo/imageatm}},
}
```

## Maintainers
* Christopher Lennan, github: [clennan](https://github.com/clennan)
* Malgorzata Adamczyk, github: [gosia-malgosia](https://github.com/gosia-malgosia)
* Gunar Maiwald: github: [gunarmaiwald](https://github.com/gunarmaiwald)
* Dat Tran, github: [datitran](https://github.com/datitran)

## Copyright

See [LICENSE](LICENSE) for details.

## TO-DOs:

- We are currently using Keras 2.2. The plan is to use tf.keras once TF 2.0 is out. Currently tf.keras is buggy,
  especially with model saving/loading (https://github.com/tensorflow/tensorflow/issues/22697)
