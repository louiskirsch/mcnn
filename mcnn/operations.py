from pathlib import Path

import matplotlib.colors
import tensorflow as tf
import numpy as np
import itertools

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from mcnn.model import McnnModel
from mcnn.samples import Dataset
from cnnvis.deconv import Deconvolutionizer
from cnnvis.visualization import ReconstructionImageVisualizer


def _evaluate_batch(session: tf.Session, model: McnnModel, input: np.ndarray, labels: np.ndarray):
    feed_dict = {
        model.input: input,
        model.labels: labels
    }
    global_step, loss, correct_count = model.step(session, feed_dict, loss=True, correct_count=True)
    accuracy = correct_count / model.batch_size
    print('[Test]  Step {} Loss {:.2f} Accuracy {:.2f}%'.format(global_step, loss, accuracy * 100))


def train(model: McnnModel, dataset: Dataset, step_count: int, checkpoint_dir: Path, log_dir: Path,
          steps_per_checkpoint: int, feature_name: str, save: bool = True):

    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)
    if not log_dir.exists():
        log_dir.mkdir(parents=True)

    with tf.Session() as session:

        model.restore_or_create(session, checkpoint_dir)
        model.setup_summary_writer(session, log_dir)

        train_sample_generator = dataset.batch_train_generator(feature_name, model.batch_size, model.sample_length)
        train_sample_iterator = iter(train_sample_generator)

        test_sample_generator = dataset.batch_test_generator(feature_name, model.batch_size, model.sample_length)
        test_sample_iterator = iter(test_sample_generator)

        for train_step in range(step_count):
            is_checkpoint_step = (train_step + 1) % steps_per_checkpoint == 0
            input, labels = next(train_sample_iterator)
            feed_dict = {
                model.input: input,
                model.labels: labels
            }
            if is_checkpoint_step:
                global_step, loss, _, correct_count = model.step(session, feed_dict, loss=True, train=True,
                                                                 correct_count=True, update_summary=True)
                accuracy = correct_count / model.batch_size
                print('[Train] Step {} Loss {:.2f} Accuracy {:.2f}%'.format(global_step, loss, accuracy * 100))
                _evaluate_batch(session, model, *next(test_sample_iterator))
                if save:
                    model.save(session, checkpoint_dir)
                    print('Model saved')
            else:
                model.step(session, feed_dict, train=True)


def deconv(model: McnnModel, dataset: Dataset, sample_count: int, checkpoint_dir: Path, feature_name: str):

    with tf.Session() as session:

        model.restore(session, checkpoint_dir)
        deconvolutionizer = Deconvolutionizer(session, model.input, model.batch_size)

        train_sample_generator = dataset.batch_train_generator(feature_name, model.batch_size, model.sample_length)

        samples = list(itertools.islice(train_sample_generator, sample_count // model.batch_size))

        analyze_layers = ['full/logits', 'relu']

        for idx, (input, label) in enumerate(samples):
            ids = list(range(idx * model.batch_size, (idx + 1) * model.batch_size))
            feed_dict = {
                model.input: input,
                deconvolutionizer.input_ids: ids
            }
            deconvolutionizer.track_samples(feed_dict, analyze_layers)

        def create_dict_by_ids(ids: np.ndarray):
            stacked = np.stack([samples[i // model.batch_size][0][i % model.batch_size] for i in ids], axis=0)
            return {
                model.input: stacked
            }

        print('Visualizing top activations')
        reconstructions = deconvolutionizer.visualize_top_activations(create_dict_by_ids, analyze_layers)

        color_map = matplotlib.colors.LinearSegmentedColormap.from_list('heat', [(0, 0, 0), (1, 0, 0)])

        for reconstruction in reconstructions:
            orig = reconstruction.orig_data
            heat = reconstruction.data
            heat = np.clip((heat - np.mean(heat)) / np.std(heat), 0, 1)
            length = orig.shape[0]
            x = range(length)
            y = orig
            plt.plot(y, c='black')
            plt.scatter(x, y, c=heat, cmap=color_map)
            plt.title(reconstruction.title)
            #plt.show()
            plt.savefig('plots/' + reconstruction.file_name)
            plt.close()
