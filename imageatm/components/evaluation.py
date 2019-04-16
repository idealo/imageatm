import itertools
import numpy as np
import matplotlib.pyplot as plt

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


BATCH_SIZE = 16
BASE_MODEL_NAME = 'MobileNet'

USE_MULTIPROCESSING, WORKERS = use_multiprocessing()

TYPE_IMAGE_LIST = List[List[Tuple[int, np.array, dict]]]  # used for type hinting


class Evaluation:
    """Calculates performance metrics for trained models.

    Loads the best model (validation accuracy) from *models* directory in job directory.
    All metrics and graphs are based on *test_samples.json* in job directory.

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

        self._determine_plot_params()
        self._load_best_model()
        self._create_evaluation_dir()

    def _determine_plot_params(self):
        """Checks whether ipython kernel is present.

        Plots will only be shown if in ipython, otherwise saved as files.
        """
        try:
            __IPYTHON__
            self.show_plots = True
            self.save_plots = False
        except NameError:
            # Suppress figure window in terminal
            # https://matplotlib.org/faq/howto_faq.html#generate-images-without-having-a-window-appear
            import matplotlib

            matplotlib.use('Agg')
            self.show_plots = False
            self.save_plots = True

    def _load_best_model(self):
        """Loads best performing model from job_dir."""
        self.logger.info('\n****** Load model ******\n')

        job_path = self.job_dir / 'models'
        model_files = list(job_path.glob('**/*.hdf5'))
        max_acc_idx = np.argmax([m.name.split('_')[3][:5] for m in model_files])
        self.best_model_file = Path(model_files[max_acc_idx]).resolve()
        self.best_model = load_model(self.best_model_file)

        self.logger.info('loaded {}\n'.format(self.best_model_file))

    def _create_evaluation_dir(self):
        """Creates evaluation dir for reporting."""
        if self.save_plots:
            evaluation_dir_name = self.best_model_file.name.split('.hdf5')[0]
            self.evaluation_dir = self.job_dir / 'evaluation_{}'.format(evaluation_dir_name)

            self.evaluation_dir.mkdir(parents=True, exist_ok=True)

    def _plot_test_set_distribution(self):
        """Plots bars with number of samples for each label in test set."""
        self.logger.info('\n****** Calculate distribution on test set ******\n')

        counts = np.bincount(self.y_true)
        title = 'Number of images in test set: {}'.format(len(self.samples_test))
        index = np.arange(self.n_classes)
        title_fontsize = 16 if self.n_classes < 4 else 18
        text_fontsize = 12 if self.n_classes < 4 else 14

        plt.bar(index, counts)
        plt.xlabel('Label', fontsize=text_fontsize)
        plt.ylabel('Number of images', fontsize=text_fontsize)
        plt.xticks(index, self.classes, fontsize=text_fontsize, rotation=30)
        plt.title(title, fontsize=title_fontsize)

        # figsize = [min(15, self.n_classes * 2), 5]
        # plt.figure(figsize=figsize)
        plt.tight_layout()

        if self.save_plots:
            target_file = self.evaluation_dir / 'test_set_distribution.pdf'
            plt.savefig(target_file)
            self.logger.info('saved under {}'.format(target_file))

        if self.show_plots:
            plt.show()

    @staticmethod
    def _get_probabilities_prediction(predictions_dist: List[List[float]]) -> List[float]:
        index = np.argmax(predictions_dist, axis=1)
        prob = [pred[index] for pred, index in zip(predictions_dist, index)]
        return prob

    def _make_prediction_on_test_set(self):
        """Makes prediction on test set."""
        self.logger.info('\n****** Make prediction on test set ******\n')

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

    def _calc_classification_report(self):
        """Calculates classification report on prediction on test set."""
        self.logger.info('\n****** Calculate classification report ******\n')

        self.accuracy = accuracy_score(y_true=self.y_true, y_pred=self.y_pred)
        self.logger.info(
            '\nModel achieves {}% accuracy on test set\n'.format(round(self.accuracy * 100, 2))
        )

        cr = classification_report(
            y_true=self.y_true, y_pred=self.y_pred, target_names=self.classes
        )
        self.logger.info(cr)

    def _plot_confusion_matrix(self):
        """Plots normalized confusion matrix."""
        self.logger.info('\n****** Plot confusion matrix ******\n')

        cm = confusion_matrix(y_true=self.y_true, y_pred=self.y_pred)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        figsize = [min(15, self.n_classes * 3.5), min(15, self.n_classes * 3.5)]
        title_fontsize = 16 if self.n_classes < 4 else 18
        text_fontsize = 12 if self.n_classes < 4 else 14

        plt.figure(figsize=figsize)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion matrix', fontsize=title_fontsize)
        plt.colorbar()
        tick_marks = np.arange(self.n_classes)
        plt.xticks(tick_marks, self.classes, rotation=45, fontsize=text_fontsize)
        plt.yticks(tick_marks, self.classes, fontsize=text_fontsize)
        plt.ylabel('True label', fontsize=text_fontsize)
        plt.xlabel('Predicted label', fontsize=text_fontsize)

        thresh = cm.max() / 2.0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(
                j,
                i,
                '{:.2f}'.format(cm[i, j]),
                horizontalalignment='center',
                color='white' if cm[i, j] > thresh else 'black',
                fontsize=text_fontsize,
            )

        plt.tight_layout()

        if self.save_plots:
            target_file = self.evaluation_dir / 'confusion_matrix.pdf'
            plt.savefig(target_file)
            self.logger.info('saved under {}'.format(target_file))

        if self.show_plots:
            plt.show()

    def run(self):
        """Runs evaluation pipeline on the best model found in job directory for the specific test set:

            - Plots test set distribution
            - Makes prediction on test set
            - Calculates classification report (accuracy, precision, recall)
            - Plots confusion matrix
        """

        self._plot_test_set_distribution()
        self._make_prediction_on_test_set()
        self._calc_classification_report()
        self._plot_confusion_matrix()

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

    # visualize misclassified images:
    def visualize_images(
        self, image_list: TYPE_IMAGE_LIST, show_heatmap: bool = False, n_plot: int = 20
    ):
        """Visualizes images in a sample list.

        Args:
            image_list: sample list.
            show_heatmap: boolean (generates a gradient based class activation map (grad-CAM), default False).
            n_plot: maximum number of plots to be shown (default 20).
        """
        if len(image_list) == 0:
            print('Empty list.')
            return
        else:
            n_rows = min(n_plot, len(image_list))
            n_cols = 2 if show_heatmap else 1

            figsize = [5 * n_cols, 5 * n_rows]
            plt.figure(figsize=figsize)

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

        if self.save_plots:
            # TODO: pass name as argument
            target_file = self.evaluation_dir / 'misclassified_images.pdf'
            plt.savefig(target_file)
            self.logger.info('saved under {}'.format(target_file))

        if self.show_plots:
            plt.show()
