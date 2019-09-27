from pathlib import Path
from click.testing import CliRunner
from imageatm.client.client import cli

p = Path(__file__)

TEST_CONFIG_FILE = p.resolve().parent / 'test_configs' / 'config_1.yml'

TEST_SAMPLES_FILE = p.resolve().parent / 'test_samples' / 'test_int_labels.json'

TEST_IMG_DIR = p.resolve().parent / 'test_images' / 'test_image_dir'


def test_help():
    runner = CliRunner()

    result = runner.invoke(cli, ['--help'])
    assert result.exit_code == 0

    result = runner.invoke(cli, ['pipeline', '--help'])
    assert result.exit_code == 0

    result = runner.invoke(cli, ['dataprep', '--help'])
    assert result.exit_code == 0

    result = runner.invoke(cli, ['train', '--help'])
    assert result.exit_code == 0

    result = runner.invoke(cli, ['evaluate', '--help'])
    assert result.exit_code == 0

    result = runner.invoke(cli, ['cloud', '--help'])
    assert result.exit_code == 0


def test_options_available():
    runner = CliRunner()

    expected = 'Options:\n  --help  Show this message and exit.\n\nCommands'
    result = runner.invoke(cli, ['--help'])
    assert expected in result.stdout

    expected = (
        'Options:\n'
        '  --image-dir PATH              Directory with image files.\n'
        '  --samples-file PATH           JSON file with samples.\n'
        '  --job-dir PATH                Directory with train, val, and test samples\n'
        '                                files and class_mapping file.\n'
        '  --provider TEXT               Cloud provider, currently supported: [aws].\n'
        '  --instance-type TEXT          Cloud instance_type [aws].\n'
        '  --region TEXT                 Cloud region [aws].\n'
        '  --vpc-id TEXT                 Cloud VPC id [aws].\n'
        '  --bucket TEXT                 Cloud bucket used for persistence [aws].\n'
        '  --tf-dir TEXT                 Directory with Terraform configs [aws].\n'
        '  --train-cloud                 Run training in cloud [aws].\n'
        '  --destroy                     Destroys cloud.\n'
        '  --resize                      Resizes images in dataprep.\n'
        '  --batch-size INTEGER          Batch size.\n'
        '  --epochs-train-dense INTEGER  Number of epochs train only dense layer.\n'
        '  --epochs-train-all INTEGER    Number of epochs train all layers.\n'
        '  --learning-rate-dense FLOAT   Learning rate dense layers.\n'
        '  --learning-rate-all FLOAT     Learning rate all layers.\n'
        '  --base-model-name TEXT        Pretrained CNN to be used for transfer learning.\n'
        '  --create-report               Create evaluation report via jupyter notebook.\n'
        '  --kernel-name TEXT            Kernel-name for juypter notebook.\n'
        '  --export-html                 Export evaluation report to html.\n'
        '  --export-pdf                  Export evaluation report to pdf.\n'
        '  --cloud-tag TEXT              Tag under which all cloud resources are created.\n'
        '  --help                        Show this message and exit.\n'
    )
    result = runner.invoke(cli, ['pipeline', '--help'])
    print(result.stdout)
    assert expected in result.stdout

    expected = (
        'Options:\n'
        '  --config-file PATH   Central configuration file.\n'
        '  --image-dir PATH     Directory with image files.\n'
        '  --samples-file PATH  JSON file with samples.\n'
        '  --job-dir PATH       Directory with train, val, and test samples files and\n'
        '                       class_mapping file.\n'
        '  --resize             Resizes images and stores them in _resized subfolder.\n'
        '  --help               Show this message and exit.\n'
    )
    result = runner.invoke(cli, ['dataprep', '--help'])
    assert expected in result.stdout

    expected = (
        'Options:\n'
        '  --config-file PATH            Central configuration file.\n'
        '  --image-dir PATH              Directory with image files.\n'
        '  --job-dir PATH                Directory with train, val, and test samples\n'
        '                                files and class_mapping file.\n'
        '  --provider TEXT               Cloud provider, currently supported: [aws].\n'
        '  --instance-type TEXT          Cloud instance_type [aws].\n'
        '  --region TEXT                 Cloud region [aws].\n'
        '  --vpc-id TEXT                 Cloud VPC id [aws].\n'
        '  --bucket TEXT                 Cloud bucket used for persistence [aws].\n'
        '  --tf-dir TEXT                 Directory with Terraform configs [aws].\n'
        '  --train-cloud                 Run training in cloud [aws].\n'
        '  --destroy                     Destroys cloud.\n'
        '  --batch-size INTEGER          Batch size.\n'
        '  --epochs-train-dense INTEGER  Number of epochs train only dense layer.\n'
        '  --epochs-train-all INTEGER    Number of epochs train all layers.\n'
        '  --learning-rate-dense FLOAT   Learning rate dense layers.\n'
        '  --learning-rate-all FLOAT     Learning rate all layers.\n'
        '  --base-model-name TEXT        Pretrained CNN to be used for transfer learning.\n'
        '  --cloud-tag TEXT              Tag under which all cloud resources are created.\n'
        '  --help                        Show this message and exit.\n'
    )
    result = runner.invoke(cli, ['train', '--help'])
    assert expected in result.stdout

    expected = (
        'Options:\n'
        '  --config-file PATH  Central configuration file.\n'
        '  --image-dir PATH    Directory with image files.\n'
        '  --job-dir PATH      Directory with test samples files and trained model.\n'
        '  --create-report     Create evaluation report via jupyter notebook.\n'
        '  --kernel-name TEXT  Kernel-name for juypter notebook.\n'
        '  --export-html       Export evaluation report to html.\n'
        '  --export-pdf        Export evaluation report to pdf.\n'
        '  --help              Show this message and exit.\n'
    )
    result = runner.invoke(cli, ['evaluate', '--help'])
    assert expected in result.stdout

    expected = (
        'Options:\n'
        '  --config-file PATH    Central configuration file.\n'
        '  --job-dir PATH        Directory with test samples files and trained model.\n'
        '  --provider TEXT       Cloud provider, currently supported: [aws].\n'
        '  --instance-type TEXT  Cloud instance_type [aws].\n'
        '  --region TEXT         Cloud region [aws].\n'
        '  --vpc-id TEXT         Cloud VPC id [aws].\n'
        '  --bucket TEXT         Cloud bucket used for persistence [aws].\n'
        '  --tf-dir TEXT         Directory with Terraform configs [aws].\n'
        '  --train-cloud         Run training in cloud [aws].\n'
        '  --destroy             Destroys cloud.\n'
        '  --no-destroy          Keeps cloud.\n'
        '  --cloud-tag TEXT      Tag under which all cloud resources are created.\n'
        '  --help                Show this message and exit.\n'
    )
    result = runner.invoke(cli, ['cloud', '--help'])
    assert expected in result.stdout
