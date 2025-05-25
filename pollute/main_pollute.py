"""
Main module for dataset noise injection and label pollution.
"""

from tableshift.tableshift import get_dataset, get_iid_dataset
import pandas as pd
import numpy as np
import pickle
from typing import Callable, Optional, Union
from collections import Counter

from noise_classes import RandomPollution, BiasedPollution, CorrelatedPollution, ConcatenatedPollution, AgePollution
from dataset_noise_conditions import get_biased_condition_function, get_combo_condition_function


class DatasetNoiseInjector:
    """Main class for injecting noise into datasets."""

    def __init__(self, cache_dir: str = './tmp'):
        self.cache_dir = cache_dir
        self.noise_strategies = {
            'random': RandomPollution(),
            'biased': BiasedPollution(),
            'correlated': CorrelatedPollution(),
            'concatenated': ConcatenatedPollution(),
            'age_based': AgePollution()
        }

    def get_noisy_dataset(self,
                         dataset_name: str,
                         noise_type: str = 'random',
                         flip_prob: float = 0.0,
                         **kwargs):
        """
        Takes original dataset, pollutes the training labels, and returns noisy dataset.

        Args:
            dataset_name: Name of the dataset
            noise_type: Type of noise ('random', 'biased', 'correlated', 'concatenated', 'age_based')
            flip_prob: Probability of label flipping
            **kwargs: Additional arguments for specific noise strategies

        Returns:
            TabularDataset object with polluted labels
        """
        # Load dataset
        dset = self._load_dataset(dataset_name)

        # Extract training data
        X_tr, y_tr, _, _ = dset.get_pandas("train")
        if y_tr.dtypes == 'int64':
            y_tr = y_tr.astype(float)

        # Apply noise strategy
        if noise_type == 'random':
            y_tr_modified = self.noise_strategies['random'].apply_noise(
                X_tr, y_tr, flip_prob=flip_prob, **kwargs
            )
        elif noise_type == 'age_based' and dataset_name == 'diabetes_readmission':
            y_tr_modified = self.noise_strategies['age_based'].apply_noise(
                X_tr, y_tr, max_flip_prob=flip_prob, **kwargs
            )
        elif noise_type == 'biased':
            condition_func = get_biased_condition_function(dataset_name)
            flip_reverse = self._should_flip_reverse(dataset_name)

            y_tr_modified = self.noise_strategies['biased'].apply_noise(
                X_tr, y_tr,
                flip_prob=flip_prob,
                condition_func=condition_func,
                flip_reverse=flip_reverse,
                **kwargs
            )
        elif noise_type == 'correlated':
            condition_func = get_combo_condition_function(dataset_name)
            flip_reverse = self._should_flip_reverse(dataset_name)

            y_tr_modified = self.noise_strategies['correlated'].apply_noise(
                X_tr, y_tr,
                flip_prob=flip_prob,
                condition_func=condition_func,
                flip_reverse=flip_reverse,
                **kwargs
            )
        elif noise_type == 'concatenated':
            condition_func = get_combo_condition_function(dataset_name)
            flip_reverse = self._should_flip_reverse(dataset_name)

            y_tr_modified = self.noise_strategies['concatenated'].apply_noise(
                X_tr, y_tr,
                flip_prob=flip_prob,
                condition_func=condition_func,
                flip_reverse=flip_reverse,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported noise type: {noise_type} for dataset: {dataset_name}")

        # Update dataset with modified labels
        self._update_dataset_labels(dset, y_tr_modified)

        return dset

    def _load_dataset(self, dataset_name: str):
        """Load dataset from appropriate source."""
        return get_dataset(dataset_name, cache_dir=self.cache_dir)

    def _should_flip_reverse(self, dataset_name: str) -> bool:
        """Determine if labels should be flipped in reverse direction for specific datasets."""
        reverse_datasets = {
            'acsincome', 'acsfoodstamps', 'brfss_blood_pressure', 'anes'
        }
        return dataset_name in reverse_datasets

    def _update_dataset_labels(self, dset, y_tr_modified):
        """Update the dataset with modified labels."""
        train_idxs = dset._get_split_idxs("train")

        # Ensure index alignment
        if isinstance(y_tr_modified, pd.Series):
            y_tr_modified.index = dset._df.index[train_idxs]
        elif isinstance(y_tr_modified, pd.DataFrame):
            y_tr_modified.index = dset._df.index[train_idxs]

        # Update target column
        target_colname = dset._post_transform_target_name()
        dset._df.loc[train_idxs, target_colname] = y_tr_modified


def main():
    """Example usage of the DatasetNoiseInjector."""
    injector = DatasetNoiseInjector()

    # Example: Apply biased noise
    dataset_name = "diabetes_readmission"
    noisy_dset = injector.get_noisy_dataset(
        dataset_name=dataset_name,
        noise_type='biased',
        flip_prob=0.1
    )

    X_tr, y_tr, _, _ = noisy_dset.get_pandas("train")
    print(f"Dataset: {dataset_name}")
    print(f"Training samples: {len(y_tr)}")
    print(f"Label distribution: {Counter(y_tr.tolist())}")


if __name__ == "__main__":
    main()