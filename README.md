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
    image_dir = 'sample_dataset/',
    samples_file = 'sample_configfile.json',
    job_dir='sample_jobdir/'
)
dp.run(resize=True)
```

Run the training:
``` python
from imageatm.components import Training

trainer = Training(dp.image_dir, dp.job_dir)
trainer.run()
```

Run the evaluation:
``` python
from imageatm.components import Evaluation

evaluater = Evaluation(image_dir=dp.image_dir, job_dir=dp.job_dir)
evaluater.run()
```

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