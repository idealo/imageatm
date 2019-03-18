import pytest
import shutil
from pathlib import Path
from imageatm.components.evaluation import Evaluation

TEST_IMAGE_DIR = Path('./tests/data/test_images').resolve()
TEST_JOB_DIR = Path('./tests/data/test_train_job').resolve()
TEST_BATCH_SIZE = 16
TEST_BASE_MODEL_NAME = 'MobileNet'


@pytest.fixture(scope='session', autouse=True)
def tear_down(request):
    def remove_evaluation_dir():
        shutil.rmtree(TEST_JOB_DIR / 'evaluation_model_mobilenet_15_0.375')

    def remove_logs():
        (TEST_JOB_DIR / 'logs').unlink()

    request.addfinalizer(remove_evaluation_dir)
    request.addfinalizer(remove_logs)


class TestEvaluation(object):
    eval = None


    def test__init__(self, mocker):
        mocker.patch('imageatm.components.evaluation.load_model', return_value={})
        global eval
        eval = Evaluation(
            image_dir=TEST_IMAGE_DIR,
            job_dir=TEST_JOB_DIR,
            batch_size=TEST_BATCH_SIZE,
            base_model_name=TEST_BASE_MODEL_NAME,
        )

        eval.show_plots = True

        assert eval.image_dir == TEST_IMAGE_DIR
        assert eval.job_dir == TEST_JOB_DIR

        assert eval.batch_size == 16
        assert eval.base_model_name == 'MobileNet'

        assert len(eval.samples_test) == 4
        assert len(eval.class_mapping) == 2
        assert eval.n_classes == 2
        assert eval.classes == ['0', '1']
        assert eval.y_true[0] == 0
        assert eval.y_true[1] == 0
        assert eval.y_true[2] == 0
        assert eval.y_true[3] == 1

    def test_run(self, mocker):
        mp_plot_dist = mocker.patch(
            'imageatm.components.evaluation.Evaluation._plot_test_set_distribution'
        )
        mp_make_pred = mocker.patch(
            'imageatm.components.evaluation.Evaluation._make_prediction_on_test_set'
        )
        mp_calc_cr = mocker.patch(
            'imageatm.components.evaluation.Evaluation._calc_classification_report'
        )
        mp_plot_cm = mocker.patch(
            'imageatm.components.evaluation.Evaluation._plot_confusion_matrix'
        )

        global eval
        eval.run()

        mp_plot_dist.assert_called()
        mp_make_pred.assert_called()
        mp_calc_cr.assert_called()
        mp_plot_cm.assert_called()

    def test__load_best_model(self, mocker):
        mp = mocker.patch('imageatm.components.evaluation.load_model')

        global eval
        eval._load_best_model()

        mp.assert_called_with(str(TEST_JOB_DIR / 'models/model_mobilenet_15_0.375.hdf5'))

    def test__plot_test_set_distribution(self, mocker):
        mock_plt_bar = mocker.patch('matplotlib.pyplot.bar')
        mock_plt_xlabel = mocker.patch('matplotlib.pyplot.xlabel')
        mock_plt_ylabel = mocker.patch('matplotlib.pyplot.ylabel')
        mock_plt_xticks = mocker.patch('matplotlib.pyplot.xticks')
        mock_plt_title = mocker.patch('matplotlib.pyplot.title')
        mock_plt_show = mocker.patch('matplotlib.pyplot.show')

        global eval
        eval._plot_test_set_distribution()

        mock_plt_bar.assert_called()
        mock_plt_xlabel.assert_called_with('Label', fontsize=12)
        mock_plt_ylabel.assert_called_with('Number of images', fontsize=12)
        mock_plt_xticks.assert_called()
        mock_plt_title.assert_called_with('Number of images in test set: 4', fontsize=16)
        mock_plt_show.assert_called()

    def test__plot_confusion_matrix(self, mocker):
        mock_plt_tight_layout = mocker.patch('matplotlib.pyplot.tight_layout')
        mock_plt_xlabel = mocker.patch('matplotlib.pyplot.xlabel')
        mock_plt_ylabel = mocker.patch('matplotlib.pyplot.ylabel')
        mock_plt_title = mocker.patch('matplotlib.pyplot.title')
        mock_plt_figure = mocker.patch('matplotlib.pyplot.figure')
        mock_plt_imshow = mocker.patch('matplotlib.pyplot.imshow')
        mock_plt_colorbar = mocker.patch('matplotlib.pyplot.colorbar')
        mock_plt_xticks = mocker.patch('matplotlib.pyplot.xticks')
        mock_plt_yticks = mocker.patch('matplotlib.pyplot.yticks')
        mock_plt_show = mocker.patch('matplotlib.pyplot.show')

        global eval
        eval.y_pred = [1, 0, 0, 0]
        eval._plot_confusion_matrix()

        mock_plt_figure.assert_called()
        mock_plt_imshow.assert_called()
        mock_plt_title.assert_called_with('Confusion matrix', fontsize=16)
        mock_plt_imshow.assert_called()
        mock_plt_colorbar.assert_called()
        mock_plt_xticks.assert_called()
        mock_plt_yticks.assert_called()
        mock_plt_xlabel.assert_called_with('Predicted label', fontsize=12)
        mock_plt_ylabel.assert_called_with('True label', fontsize=12)
        mock_plt_tight_layout.assert_called()
        mock_plt_show.assert_called()

    def test_get_correct_wrong_examples(self):
        def paramized_test_correct_wrong_function(y_pred, label, num_correct, num_wrong):
            global eval
            eval.y_pred = y_pred
            correct, wrong = eval.get_correct_wrong_examples(label)
            assert len(correct) == num_correct
            assert len(wrong) == num_wrong

        paramized_test_correct_wrong_function([0, 0, 0, 1], 0, 3, 0)
        paramized_test_correct_wrong_function([0, 0, 0, 1], 1, 1, 0)
        paramized_test_correct_wrong_function([1, 0, 0, 0], 0, 2, 1)
        paramized_test_correct_wrong_function([1, 0, 0, 0], 1, 0, 1)

    def test_visualize_images_empty_image_list(self):
        global eval
        assert eval.visualize_images(image_list=[]) is None

    def test__get_probabilities_prediction(self):
        global eval
        eval.y_pred_prob = [[0.4, 0.6], [0.2, 0.8], [0.7, 0.3]]
        assert eval._get_probabilities_prediction(predictions_dist=eval.y_pred_prob) == [
            0.6,
            0.8,
            0.7,
        ]
