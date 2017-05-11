from abc import abstractmethod
from pathlib import Path
from typing import Tuple, Iterable

import pandas as pd
import numpy as np

import random


def batch_samples(sample_generator, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    batch = [sample_generator() for _ in range(batch_size)]
    x = np.stack((x for x, y in batch), axis=0)
    y = np.stack((y for x, y in batch), axis=0)
    return x, y


class Dataset:

    @abstractmethod
    def batch_train_generator(self, feature_name: str, batch_size: int,
                              sample_length: int) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        raise NotImplementedError()

    @abstractmethod
    def batch_test_generator(self, feature_name: str, batch_size: int,
                             sample_length: int) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        raise NotImplementedError()


class HorizontalDataset(Dataset):

    def __init__(self, dataset_train_path: Path, dataset_test_path: Path):
        self.df_train = pd.read_csv(dataset_train_path, header=None)
        self.df_test = pd.read_csv(dataset_test_path, header=None)
        self.target_classes = self.df_train[0].unique()
        self.class_to_id = dict((c, id) for id, c in enumerate(self.target_classes))
        self.target_classes_count = len(self.target_classes)

    def _sample(self, dataset: pd.DataFrame, sample_length: int) -> Tuple[np.ndarray, int]:
        row_count, entry_length = dataset.shape
        ts_length = entry_length - 1
        row = np.random.randint(0, row_count)
        offset = np.random.randint(1, ts_length - sample_length + 1)
        x = dataset.iloc[row, offset:offset + sample_length].values
        y = self.class_to_id[dataset.iloc[row, 0]]
        return x, y

    def batch_train_generator(self, feature_name: str, batch_size: int,
                              sample_length: int) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        while True:
            yield batch_samples(lambda: self._sample(self.df_train, sample_length), batch_size)

    def batch_test_generator(self, feature_name: str, batch_size: int,
                             sample_length: int) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        while True:
            yield batch_samples(lambda: self._sample(self.df_test, sample_length), batch_size)


class PercentalSplitDataset(Dataset):

    def __init__(self, dataset_path: Path, target_name: str, split_threshold: float):
        self.df = pd.read_csv(dataset_path)
        self.target = self.df[target_name]
        self.split_threshold = split_threshold

        self.target_classes = self.target.unique()
        self.target_classes_count = len(self.target_classes)

        self.data_class_grouped = [self.df[self.target == cls] for cls in self.target_classes]

    def _random_training_index(self, row_count: int, sample_length: int) -> int:
        return random.randrange(int(self.split_threshold * (row_count - sample_length)))

    def _random_test_index(self, row_count: int, sample_length: int) -> int:
        return random.randrange(int(self.split_threshold * row_count), row_count - sample_length)

    def _row_count_for_class(self, cls_index: int) -> int:
        return self.data_class_grouped[cls_index].shape[0]

    def _z_normalize(self, data: np.ndarray) -> np.ndarray:
        return (data - np.mean(data)) / (np.std(data) + 1e-8)

    def _generate_sample(self, feature_name: str, random_index_fnc,
                         sample_length: int) -> Tuple[np.ndarray, np.ndarray]:
        cls_index = random.randrange(self.target_classes_count)
        row_count = self._row_count_for_class(cls_index)
        start_index = random_index_fnc(row_count, sample_length)
        x = self.data_class_grouped[cls_index][feature_name][start_index:start_index + sample_length]
        x = self._z_normalize(x)
        y = cls_index
        return x, y

    def _generate_batch(self, feature_name: str, batch_size: int, random_index_fnc,
                        sample_length: int) -> Tuple[np.ndarray, np.ndarray]:
        gen_sample = lambda: self._generate_sample(feature_name, random_index_fnc, sample_length)
        return batch_samples(gen_sample, batch_size)

    def batch_train_generator(self, feature_name: str, batch_size: int,
                              sample_length: int) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        while True:
            yield self._generate_batch(feature_name, batch_size, self._random_training_index, sample_length)

    def batch_test_generator(self, feature_name: str, batch_size: int,
                             sample_length: int) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        while True:
            yield self._generate_batch(feature_name, batch_size, self._random_test_index, sample_length)
