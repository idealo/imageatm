import itertools
import numpy as np
import matplotlib.pyplot as plt
import nbformat
import papermill as pm
from nbconvert import HTMLExporter, PDFExporter
from os.path import dirname

plt.style.use('ggplot')
from typing import List, Union, Tuple, Any
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from vis.visualization import visualize_cam
from imageatm.handlers.image_classifier import ImageClassifier
from imageatm.handlers.data_generator import ValDataGenerator
from imageatm.utils.io import load_json
from imageatm.utils.images import load_image
from imageatm.utils.logger import get_logger
from imageatm.utils.tf_keras import use_multiprocessing, load_model
import imageatm.notebooks


BATCH_SIZE = 16
BASE_MODEL_NAME = 'MobileNet'
MAX_N_CLASSES = 20

USE_MULTIPROCESSING, WORKERS = use_multiprocessing()

TYPE_IMAGE_LIST = List[List[Tuple[int, np.array, dict]]]  # used for type hinting


class Evaluation:
    """Calculates performance metrics for trained models.

    Loads the best model (validation accuracy) from *models* directory in job directory.
    All metrics and graphs are based on *test_samples.json* in job directory.
    Plots will only be shown if number of classes 20 or less.

    Attributes:
        image_dir: Path of image directory.
        job_dir: Path to job directory with samples.
        batch_size: Number of images per batch (default 64).
        base_model_name: Name of pretrained CNN (default MobileNet).
    """

    def __init__(
        self,
        image_dir: str,
        job_dir: str,
        batch_size: int = BATCH_SIZE,
        base_model_name: str = BASE_MODEL_NAME,
        **kwargs
    ) -> None:
        """Inits evaluation component.

        Loads the best model from job directory.
        Creates evaluation directory if app was started from commandline.
        """
        self.image_dir = Path(image_dir).resolve()
        self.job_dir = Path(job_dir).resolve()
        self.batch_size = batch_size
        self.base_model_name = base_model_name

        self.logger = get_logger(__name__, self.job_dir)
        self.samples_test: list = load_json(self.job_dir / 'test_samples.json')  # type: ignore
        self.class_mapping: dict = load_json(self.job_dir / 'class_mapping.json')  # type: ignore
        self.n_classes = len(self.class_mapping)
        self.classes = [str(self.class_mapping[str(i)]) for i in range(self.n_classes)]
        self.y_true = np.array([i['label'] for i in self.samples_test])
        self.figures = []

        self._determine_plot_params()
        self._load_best_model()
        self._create_evaluation_dir()

    def _determine_plot_params(self):
        """Determines fontsizes and checks whether ipython kernel is present.

        Plots will only be shown if in ipython, otherwise saved as files.
        """
        self.fontsize_title = 18 if self.n_classes < 4 else 18
        self.fontsize_label = 14 if self.n_classes < 4 else 14
        self.fontsize_ticks = 12 if self.n_classes < 4 else 12
        self.mode_ipython = True if self._is_in_ipython_mode() else False

    def _is_in_ipython_mode(self):
        try:
            __IPYTHON__
            return True

        except NameError:
            ## TODO: Is this obsolete? Please remove!
            # Suppress figure window in terminal
            # https://matplotlib.org/faq/howto_faq.html#generate-images-without-having-a-window-appear
            import matplotlib
            matplotlib.use('Agg')

            return False

    def _load_best_model(self):
        """Loads best performing model from job_dir."""
        self.logger.info('\n****** Load model ******\n')
        best_model_file = self._determine_best_modelfile()
        self.best_model_file = Path(best_model_file).resolve()
        self.best_model = load_model(self.best_model_file)

        self.logger.info('loaded {}'.format(self.best_model_file))

    def _determine_best_modelfile(self):
        """Determines best performing model from job_dir."""
        job_path = self.job_dir / 'models'
        model_files = list(job_path.glob('**/*.hdf5'))
        max_acc_idx = np.argmax([m.name.split('_')[3][:5] for m in model_files])
        return model_files[max_acc_idx]

    def _create_evaluation_dir(self):
        """Creates evaluation dir for reporting."""
        if not self.mode_ipython:
            evaluation_dir_name = self.best_model_file.name.split('.hdf5')[0]
            self.evaluation_dir = self.job_dir / 'evaluation_{}'.format(evaluation_dir_name)

            self.evaluation_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _get_probabilities_prediction(predictions_dist: List[List[float]]) -> List[float]:
        index = np.argmax(predictions_dist, axis=1)
        prob = [pred[index] for pred, index in zip(predictions_dist, index)]
        return prob

    def _make_prediction_on_test_set(self):
        """Makes prediction on test set."""

        self.classifier = ImageClassifier(
            base_model_name=self.base_model_name,
            n_classes=self.n_classes,
            weights=None,
            dropout_rate=None,
            learning_rate=None,
            loss=None,
        )
        self.classifier.model = self.best_model

        self.data_generator = ValDataGenerator(
            samples=self.samples_test,
            image_dir=self.image_dir,
            batch_size=self.batch_size,
            n_classes=self.n_classes,
            basenet_preprocess=self.classifier.get_preprocess_input(),
        )

        predictions_dist = self.classifier.predict_generator(
            data_generator=self.data_generator,
            workers=WORKERS,
            use_multiprocessing=USE_MULTIPROCESSING,
            verbose=1,
        )

        self.y_pred = np.argmax(predictions_dist, axis=1)
        self.y_pred_prob = self._get_probabilities_prediction(predictions_dist=predictions_dist)

    def _plot_test_set_distribution(self, figsize: (float, float) = [8, 5]):
        """Plots bars with number of samples for each label in test set."""
        assert self.mode_ipython, 'Plotting is only possible when in ipython-mode'

        if self.n_classes > MAX_N_CLASSES:
            self.logger.info('\nPlotting only for max {} classes\n'.format(MAX_N_CLASSES))
            return

        x_tick_marks = np.arange(self.n_classes)
        y_values = np.bincount(self.y_true)
        title = 'Number of images in test set: {}'.format(len(self.samples_test))

        fig = plt.figure(figsize=figsize)
        plt.rcParams["axes.grid"] = True
        plt.bar(x_tick_marks, y_values)
        plt.title(title, fontsize=self.fontsize_title)
        plt.xlabel('Label', fontsize=self.fontsize_label)
        plt.ylabel('Number of images', fontsize=self.fontsize_label)
        plt.xticks(x_tick_marks, self.classes, fontsize=self.fontsize_ticks, rotation=30)

        plt.tight_layout()
        plt.show()

    def _print_test_set_distribution(self):
        """Prints distribution for labels in test set."""
        assert not self.mode_ipython, 'Printing is recommended when not in ipython-mode'

        max_length = len(max(self.classes, key=len))
        y_values = np.bincount(self.y_true)
        for i, c in enumerate(self.classes):
            label = c + ' ' * (max_length - len(c))
            self.logger.info("{}\t{}". format(label, y_values[i]))

    def _print_classification_report(self):
        """Prints classification for labels in test set."""
        cr = classification_report(
            y_true=self.y_true, y_pred=self.y_pred, target_names=self.classes, output_dict=True
        )

        metrics = ['precision', 'recall', 'f1-score', 'support']
        categories = self.classes.copy()
        categories.extend(['macro avg', 'weighted avg'])

        max_length = len(max(self.classes, key=len))
        self.logger.info("{}\t{}\t{}\t{}\t{}". format(' ' * max_length, 'prec', "rec", "f1", "support"))
        for c in categories:
            label = c + ' ' * (max_length - len(c))
            line_output = "{}\t".format(label)
            for m in metrics:
                if m == 'support':
                    line_output += "{}\t".format(cr[c][m])
                else:
                    line_output += "{0:.2f}\t".format(cr[c][m])
            self.logger.info(line_output)

    def _plot_confusion_matrix(self, figsize: (float, float) = [9, 9], precision: bool = False):
        """Plots normalized confusion matrix."""
        assert self.mode_ipython, 'Plotting is only possible when in ipython-mode'

        if self.n_classes > MAX_N_CLASSES:
            self.logger.info('\nPlotting only for max {} classes\n'.format(MAX_N_CLASSES))
            return

        (title, xlabel, ylabel) = \
            ('Confusion matrix (precision)', 'True label', 'Predicted label') if precision \
            else ('Confusion matrix (recall)', 'Predicted label', 'True label')

        cm = confusion_matrix(y_true=self.y_pred, y_pred=self.y_true) if precision \
            else confusion_matrix(y_true=self.y_true, y_pred=self.y_pred)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        tick_marks = np.arange(self.n_classes)

        fig = plt.figure(figsize=figsize)
        plt.rcParams["axes.grid"] = False
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
        plt.colorbar()
        plt.title(title, fontsize=self.fontsize_title)
        plt.xlabel(xlabel, fontsize=self.fontsize_label)
        plt.ylabel(ylabel, fontsize=self.fontsize_label)
        plt.xticks(tick_marks, self.classes, rotation=45, fontsize=self.fontsize_ticks, ha='right')
        plt.yticks(tick_marks, self.classes, fontsize=self.fontsize_ticks)

        thresh = cm.max() / 2.0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(
                j,
                i,
                '{:.2f}'.format(cm[i, j]),
                horizontalalignment='center',
                color='white' if cm[i, j] > thresh else 'black',
                fontsize=self.fontsize_ticks,
            )

        plt.tight_layout()
        plt.show()

    def _print_confusion_matrix(self, precision: bool = False):
        """Prints normalized confusion matrix."""
        assert not self.mode_ipython, 'Printing is recommended when not in ipython-mode'

        cm = confusion_matrix(y_true=self.y_pred, y_pred=self.y_true) if precision \
            else confusion_matrix(y_true=self.y_true, y_pred=self.y_pred)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        max_length = len(max(self.classes, key=len))
        for i, c in enumerate(self.classes):
            label = c + ' ' * (max_length - len(c))
            line_output = "{}\t".format(label)
            for x in cm[i].tolist():
                line_output += "{0:.2f}\t".format(x)
            self.logger.info(line_output)

    def _plot_correct_wrong_examples(self):
        """Plots correct and wrong examples for each label in test set."""
        assert self.mode_ipython, 'Plotting is only possible when in ipython-mode'

        if self.n_classes > MAX_N_CLASSES:
            self.logger.info('\nPlotting only for max {} classes\n'.format(MAX_N_CLASSES))
            return

        for i in range(len(self.classes)):
            c, w = self.get_correct_wrong_examples(label=i)
            self.visualize_images(c, title='Label: "{}" (correct predicted)'.format(self.classes[i]), show_heatmap=True, n_plot=3)
            self.visualize_images(w, title='Label: "{}" (wrong predicted)'.format(self.classes[i]), show_heatmap=True, n_plot=3)

    def _create_report(self, report_kernel_name:str, report_export_html:bool, report_export_pdf:bool):
        """Creates report from notebook-template and stores it in different formats all figures.

            - Jupyter Notebook
            - HTML
            - PDF
        """
        assert not self.mode_ipython, 'Create report is only possible when not in ipython mode'

        filepath_template = dirname(imageatm.notebooks.__file__) + '/evaluation_template.ipynb'
        filepath_notebook = self.evaluation_dir / 'evaluation_report.ipynb'
        filepath_html = self.evaluation_dir / 'evaluation_report.html'
        filepath_pdf = self.evaluation_dir / 'evaluation_report.pdf'

        pm.execute_notebook(
            str(filepath_template),
            str(filepath_notebook),
            parameters=dict(
                image_dir=str(self.image_dir),
                job_dir=str(self.job_dir)
            ),
            kernel_name=report_kernel_name
        )

        with open(filepath_notebook) as f:
            nb = nbformat.read(f, as_version=4)

        if report_export_html:
            self.logger.info('\n****** Create HTML ******\n')
            with open(filepath_notebook) as f:
                nb = nbformat.read(f, as_version=4)

            html_exporter = HTMLExporter()
            html_data, resources = html_exporter.from_notebook_node(nb)

            with open(filepath_html, 'w') as f:
                f.write(html_data)
                f.close()

        if report_export_pdf:
            self.logger.info('\n****** Create PDF ******\n')

            pdf_exporter = PDFExporter()
            pdf_exporter.template_file = dirname(imageatm.notebooks.__file__) + '/tex_templates/evaluation_report.tplx'
            pdf_data, resources = pdf_exporter.from_notebook_node(nb, resources={
                'metadata': {'name': 'Evaluation Report'}
            })

            with open(filepath_pdf, 'wb') as f:
                f.write(pdf_data)
                f.close()

    # TO-DO: Enforce string or integer but not both at the same time
    def get_correct_wrong_examples(
        self, label: Union[int, str]
    ) -> Tuple[TYPE_IMAGE_LIST, TYPE_IMAGE_LIST]:
        """Gets correctly and wrongly predicted samples for a given label.

        Args:
            label: int or str (label for which the predictions should be considered).

        Returns:
            (correct, wrong): Tuple of two image lists.
        """
        correct = []
        wrong = []

        if type(label) == str:
            class_mapping_inv = {v: k for k, v in self.class_mapping.items()}
            label = int(class_mapping_inv[label])

        for i, sample in enumerate(self.samples_test):
            if self.y_true[i] == label:
                image_file = self.image_dir / sample['image_id']
                if self.y_true[i] == self.y_pred[i]:
                    correct.append([i, load_image(image_file, target_size=(224, 224)), sample])
                else:
                    wrong.append([i, load_image(image_file, target_size=(224, 224)), sample])

        return correct, wrong

    def visualize_images(
        self, image_list: TYPE_IMAGE_LIST, title: str = 'Images for visualisation', show_heatmap: bool = False, n_plot: int = 20
    ):
        """Visualizes images in a sample list.

        Args:
            image_list: sample list.
            show_heatmap: boolean (generates a gradient based class activation map (grad-CAM), default False).
            n_plot: maximum number of plots to be shown (default 20).
        """
        assert self.mode_ipython, 'Plotting is only possible when in ipython-mode'

        if len(image_list) == 0:
            print('Empty list.')
            return
        else:
            n_rows = min(n_plot, len(image_list))
            n_cols = 2 if show_heatmap else 1

            figsize = [5 * n_cols, 4 * n_rows]
            fig = plt.figure(figsize=figsize)
            fig.suptitle(title, fontsize=self.fontsize_title)

            plot_count = 1
            for (i, img, sample) in image_list[:n_rows]:
                plt.subplot(n_rows, n_cols, plot_count)
                plt.imshow(img)
                plt.axis('off')
                plt.title(
                    'true: {}, predicted: {} ({})'.format(
                        self.class_mapping[str(self.y_true[i])],
                        self.class_mapping[str(self.y_pred[i])],
                        str(round(self.y_pred_prob[i], 2)),
                    )
                )
                plot_count += 1

                if show_heatmap is True:
                    heatmap = visualize_cam(
                        model=self.classifier.model,
                        layer_idx=89,
                        filter_indices=[self.y_pred[i]],
                        seed_input=self.classifier.get_preprocess_input()(
                            np.array(img).astype(np.float32)
                        ),
                    )
                    plt.subplot(n_rows, n_cols, plot_count)
                    plt.imshow(img)
                    plt.imshow(heatmap, alpha=0.7)
                    plt.axis('off')
                    plot_count += 1

            plt.show()

    def run(self,
            report_create: bool = False,
            report_kernel_name: str = 'imageatm',
            report_export_html: bool = False,
            report_export_pdf: bool = False
        ):
        """Runs evaluation pipeline on the best model found in job directory for the specific test set:

            - Makes prediction on test set
            - Plots test set distribution
            - Plots classification report (accuracy, precision, recall)
            - Plots confusion matrix (on precsion and on recall)
            - Plots correct and wrong examples

           If not in ipython mode an evaluation report is created.

        Args:
        	report_create: boolean (create ipython kernel)
            report_kernel_name: str (name of ipython kernel)
            report_export_html: boolean (exports report to html).
            report_export_pdf: boolean (exports report to pdf).
        """
        if self.mode_ipython:
            self.logger.info('\n****** Make prediction on test set ******\n')
            self._make_prediction_on_test_set()

            self.logger.info('\n****** Plot distribution on test set ******\n')
            self._plot_test_set_distribution(figsize=[8, 5])

            self.logger.info('\n****** Plot classification report ******\n')
            # self._plot_classification_report(figsize=[4 + self.n_classes*0.5, 4 + self.n_classes*0.5])
            self._print_classification_report()

            self.logger.info('\n****** Plot confusion matrix (recall) ******\n')
            self._plot_confusion_matrix(figsize=[4 + self.n_classes*0.5, 4 + self.n_classes*0.5])

            self.logger.info('\n****** Plot confusion matrix (precision) ******\n')
            self._plot_confusion_matrix(figsize=[4 + self.n_classes*0.5, 4 + self.n_classes*0.5], precision=True)

            self.logger.info('\n****** Plot correct and wrong examples ******\n')
            self._plot_correct_wrong_examples()

        elif report_create:
            self.logger.info('\n****** Create Jupyter Notebook (this may take a while) ******\n')
            self._create_report(report_kernel_name, report_export_html, report_export_pdf)

        else:
            self.logger.info('\n****** Make prediction on test set ******\n')
            self._make_prediction_on_test_set()

            self.logger.info('\n****** Print distribution on test set ******\n')
            self._print_test_set_distribution()

            self.logger.info('\n****** Print classification report ******\n')
            self._print_classification_report()

            self.logger.info('\n****** Print confusion matrix (recall) ******\n')
            self._print_confusion_matrix()

            self.logger.info('\n****** Print confusion matrix (precision) ******\n')
            self._print_confusion_matrix(precision=True)
