from setuptools import setup, find_packages

long_description = '''
Image ATM is a one-click tool that automates the workflow of a typical
image classification pipeline in an opinionated way, this includes:

- Preprocessing and validating input images and labels
- Starting/terminating cloud instance with GPU support
- Training
- Model evaluation

Read the documentation at: https://idealo.github.io/imageatm/

Image ATM is compatible with Python 3.6 and is distributed under the Apache 2.0 license.
'''

setup(
    name='imageatm',
    version='0.1.0',
    author='Christopher Lennan, Malgorzata Adamczyk, Gunar Maiwald, Dat Tran',
    author_email='christopherlennan@gmail.com, m.adamczyk.berlin@gmail.com, gunar.maiwald@web.de, datitran@gmail.com',
    description='Image classification for everyone',
    long_description=long_description,
    license='Apache 2.0',
    install_requires=[
        'Keras>=2.2.4',
        'keras-vis>=0.4.1',
        'tensorflow==1.13.1',
        'awscli',
        'Click',
        'h5py',
        'matplotlib',
        'Pillow',
        'scikit-learn',
        'scipy==1.1.*',
        'tqdm',
        'yarl',
    ],
    extras_require={
        'tests': ['pytest==4.3.0', 'pytest-cov==2.6.1', 'pytest-mock', 'mock', 'mypy'],
        'docs': ['mkdocs==1.0.4', 'mkdocs-material==4.0.2'],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    packages=find_packages(exclude=('tests',)),
    entry_points={'console_scripts': ['imageatm=imageatm.client.client:cli']},
)
