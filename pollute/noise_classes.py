"""
Different noise injection strategies for dataset label pollution.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Callable, Optional
from collections import Counter


class NoiseStrategy(ABC):
    """Abstract base class for noise injection strategies."""

    @abstractmethod
    def apply_noise(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> pd.Series:
        """Apply noise to the labels."""
        pass


class RandomPollution(NoiseStrategy):
    """Random label flipping strategy."""

    def apply_noise(self,
                    X_train: pd.DataFrame,
                    y_train: pd.Series,
                    flip_prob: float,
                    seed: int = 42) -> pd.Series:
        """
        Pollute a binary pandas Series by flipping labels with a given probability.

        Args:
            X_train: Feature dataset (unused for random pollution)
            y_train: The binary pandas Series to pollute
            flip_prob: The probability with which to flip each label
            seed: Seed for the random number generator

        Returns:
            The polluted series
        """
        # Validate input
        if not all(y_train.isin([0.0, 1.0])):
            raise ValueError("Series should only contain float values of 0.0 or 1.0")

        np.random.seed(seed)

        # Generate flip mask
        probabilities = np.random.rand(len(y_train))
        flip_mask = probabilities < flip_prob

        # Apply flips
        polluted_series = y_train.copy()
        polluted_series[flip_mask] = 1.0 - polluted_series[flip_mask]


        return polluted_series


class BiasedPollution(NoiseStrategy):
    """Biased label flipping based on single feature conditions."""

    def apply_noise(self,
                    X_train: pd.DataFrame,
                    y_train: pd.Series,
                    flip_prob: float,
                    condition_func: Callable[[pd.DataFrame], pd.Series],
                    flip_reverse: bool = False,
                    seed: int = 42) -> pd.Series:
        """
        Apply biased noise to labels with different probabilities for 0->1 and 1->0 flips.

        Args:
            X_train: Feature dataset
            y_train: Label dataset
            flip_prob: Probability of flipping labels
            condition_func: Function to determine where to apply label flipping
            flip_reverse: If True, flip 1->0 instead of 0->1
            seed: Random seed

        Returns:
            Noised labels
        """
        np.random.seed(seed)

        # Set flip probabilities based on direction
        if flip_reverse:
            prob_0_to_1, prob_1_to_0 = 0.0, flip_prob
        else:
            prob_0_to_1, prob_1_to_0 = flip_prob, 0.0

        # Get condition mask
        condition_mask = condition_func(X_train)

        # Create copy for modification
        noised_y_train = y_train.copy()

        # Apply 0->1 flips
        if prob_0_to_1 > 0:
            zero_indices = noised_y_train[(noised_y_train == 0) & condition_mask].index
            flip_0_to_1 = np.random.rand(len(zero_indices)) < prob_0_to_1
            noised_y_train.loc[zero_indices[flip_0_to_1]] = 1
            print(f"Flipped {flip_0_to_1.sum()} labels from 0 to 1 out of {len(zero_indices)} eligible")

        # Apply 1->0 flips
        if prob_1_to_0 > 0:
            one_indices = noised_y_train[(noised_y_train == 1) & condition_mask].index
            flip_1_to_0 = np.random.rand(len(one_indices)) < prob_1_to_0
            noised_y_train.loc[one_indices[flip_1_to_0]] = 0


        return noised_y_train


class CorrelatedPollution(NoiseStrategy):
    """Correlated label flipping based on multiple conditions combined with AND."""

    def apply_noise(self,
                    X_train: pd.DataFrame,
                    y_train: pd.Series,
                    flip_prob: float,
                    condition_func: Callable[[pd.DataFrame], pd.Series],
                    flip_reverse: bool = False,
                    seed: int = 42) -> pd.Series:
        """
        Apply correlated noise using AND combination of conditions.

        Args:
            X_train: Feature dataset
            y_train: Label dataset
            flip_prob: Probability of flipping labels
            condition_func: Function that combines conditions with AND
            flip_reverse: If True, flip 1->0 instead of 0->1
            seed: Random seed

        Returns:
            Noised labels
        """
        np.random.seed(seed)

        # Set flip probabilities based on direction
        if flip_reverse:
            prob_0_to_1, prob_1_to_0 = 0.0, flip_prob
        else:
            prob_0_to_1, prob_1_to_0 = flip_prob, 0.0

        # Get condition mask (should use AND combination)
        condition_mask = condition_func(X_train, connector='and')

        # Create copy for modification
        noised_y_train = y_train.copy()

        # Apply 0->1 flips
        if prob_0_to_1 > 0:
            zero_indices = noised_y_train[(noised_y_train == 0) & condition_mask].index
            flip_0_to_1 = np.random.rand(len(zero_indices)) < prob_0_to_1
            noised_y_train.loc[zero_indices[flip_0_to_1]] = 1
            print(f"Flipped {flip_0_to_1.sum()} labels from 0 to 1 out of {len(zero_indices)} eligible")

        # Apply 1->0 flips
        if prob_1_to_0 > 0:
            one_indices = noised_y_train[(noised_y_train == 1) & condition_mask].index
            flip_1_to_0 = np.random.rand(len(one_indices)) < prob_1_to_0
            noised_y_train.loc[one_indices[flip_1_to_0]] = 0
            print(f"Flipped {flip_1_to_0.sum()} labels from 1 to 0 out of {len(one_indices)} eligible")


        return noised_y_train


class ConcatenatedPollution(NoiseStrategy):
    """Concatenated label flipping based on multiple conditions combined with OR."""

    def apply_noise(self,
                    X_train: pd.DataFrame,
                    y_train: pd.Series,
                    flip_prob: float,
                    condition_func: Callable[[pd.DataFrame], pd.Series],
                    flip_reverse: bool = False,
                    seed: int = 42) -> pd.Series:
        """
        Apply concatenated noise using OR combination of conditions.

        Args:
            X_train: Feature dataset
            y_train: Label dataset
            flip_prob: Probability of flipping labels
            condition_func: Function that combines conditions with OR
            flip_reverse: If True, flip 1->0 instead of 0->1
            seed: Random seed

        Returns:
            Noised labels
        """
        np.random.seed(seed)

        # Set flip probabilities based on direction
        if flip_reverse:
            prob_0_to_1, prob_1_to_0 = 0.0, flip_prob
        else:
            prob_0_to_1, prob_1_to_0 = flip_prob, 0.0

        # Get condition mask (should use OR combination)
        condition_mask = condition_func(X_train, connector='or')

        # Create copy for modification
        noised_y_train = y_train.copy()

        # Apply 0->1 flips
        if prob_0_to_1 > 0:
            zero_indices = noised_y_train[(noised_y_train == 0) & condition_mask].index
            flip_0_to_1 = np.random.rand(len(zero_indices)) < prob_0_to_1
            noised_y_train.loc[zero_indices[flip_0_to_1]] = 1
            print(f"Flipped {flip_0_to_1.sum()} labels from 0 to 1 out of {len(zero_indices)} eligible")

        # Apply 1->0 flips
        if prob_1_to_0 > 0:
            one_indices = noised_y_train[(noised_y_train == 1) & condition_mask].index
            flip_1_to_0 = np.random.rand(len(one_indices)) < prob_1_to_0
            noised_y_train.loc[one_indices[flip_1_to_0]] = 0
            print(f"Flipped {flip_1_to_0.sum()} labels from 1 to 0 out of {len(one_indices)} eligible")

        return noised_y_train


class AgePollution(NoiseStrategy):
    """Age-based label flipping with probabilities that scale with age."""

    def apply_noise(self,
                    X_train: pd.DataFrame,
                    y_train: pd.Series,
                    max_flip_prob: float = 0.4,
                    flip_direction: str = '0_to_1',
                    seed: int = 42) -> pd.Series:
        """
        Apply label flipping with probabilities that scale with age.

        Args:
            X_train: Feature dataset with age group columns
            y_train: Label dataset
            max_flip_prob: Maximum probability of flipping for the oldest age group
            flip_direction: Either '0_to_1' to flip 0 labels to 1, or '1_to_0'
            seed: Random seed

        Returns:
            Noised labels with age-dependent flipping probabilities
        """
        np.random.seed(seed)

        noised_y_train = y_train.copy()

        # Define age groups and scaling factors
        age_groups = [
            'age_20-30)', 'age_30-40)', 'age_40-50)', 'age_50-60)',
            'age_60-70)', 'age_70-80)', 'age_80-90)', 'age_90-100)'
        ]

        # Check which age group columns exist
        available_age_groups = [col for col in age_groups if col in X_train.columns]

        if not available_age_groups:
            raise ValueError("No age group columns found in the dataset")

        # Define scaling factors (youngest = 0, oldest = 1)
        scaling_factors = {
            'age_20-30)': 0.0, 'age_30-40)': 0.1, 'age_40-50)': 0.2,
            'age_50-60)': 0.4, 'age_60-70)': 0.6, 'age_70-80)': 0.8,
            'age_80-90)': 0.9, 'age_90-100)': 1.0
        }

        # Track statistics
        flips_per_group = {group: 0 for group in available_age_groups}
        total_in_group = {group: 0 for group in available_age_groups}

        # Process each age group
        for age_group in available_age_groups:
            age_mask = X_train[age_group] == 1

            if age_mask.sum() == 0:
                continue

            flip_prob = scaling_factors[age_group] * max_flip_prob

            if flip_direction == '0_to_1':
                target_indices = noised_y_train[(noised_y_train == 0) & age_mask].index
                total_in_group[age_group] = len(target_indices)

                if len(target_indices) > 0:
                    flip_mask = np.random.rand(len(target_indices)) < flip_prob
                    noised_y_train.loc[target_indices[flip_mask]] = 1
                    flips_per_group[age_group] = flip_mask.sum()

            elif flip_direction == '1_to_0':
                target_indices = noised_y_train[(noised_y_train == 1) & age_mask].index
                total_in_group[age_group] = len(target_indices)

                if len(target_indices) > 0:
                    flip_mask = np.random.rand(len(target_indices)) < flip_prob
                    noised_y_train.loc[target_indices[flip_mask]] = 0
                    flips_per_group[age_group] = flip_mask.sum()
            else:
                raise ValueError("flip_direction must be either '0_to_1' or '1_to_0'")

        # Print statistics
        self._print_age_statistics(flips_per_group, total_in_group, scaling_factors,
                                   max_flip_prob, flip_direction, y_train, noised_y_train)

        return noised_y_train

    def _print_age_statistics(self, flips_per_group, total_in_group, scaling_factors,
                              max_flip_prob, flip_direction, y_train, noised_y_train):
        """Print detailed statistics about age-based flipping."""
        print(f"\nAge-based pollution applied:")
        print(f"Flip direction: {flip_direction}, Max flip probability: {max_flip_prob}")
        print(f"Age group flip statistics:")

        for group in flips_per_group:
            if total_in_group[group] > 0:
                actual_flip_rate = flips_per_group[group] / total_in_group[group]
                target_flip_rate = scaling_factors[group] * max_flip_prob
                print(f"  {group}: {flips_per_group[group]}/{total_in_group[group]} flipped "
                      f"({actual_flip_rate:.2%}, target was {target_flip_rate:.2%})")

        total_flips = sum(flips_per_group.values())
        total_eligible = sum(total_in_group.values())

        if total_eligible > 0:
            print(f"\nTotal flips: {total_flips}/{total_eligible} ({total_flips / total_eligible:.2%})")



class SimplePollution(NoiseStrategy):
    """Simple conditional noise that flips a fraction of labels meeting a condition."""

    def apply_noise(self,
                    X_train: pd.DataFrame,
                    y_train: pd.Series,
                    flip_prob: float,
                    condition_func: Callable[[pd.DataFrame], pd.Series],
                    seed: int = 42) -> pd.Series:
        """
        Apply conditional noise by flipping a fraction of labels that meet the condition.

        Args:
            X_train: Feature dataset
            y_train: Label dataset
            flip_prob: Fraction of conditioned labels to flip
            condition_func: Function to determine which samples to consider
            seed: Random seed

        Returns:
            Noised labels
        """
        np.random.seed(seed)

        # Get condition mask
        condition_mask = condition_func(X_train)

        # Calculate number of labels to noise within the conditioned subset
        num_to_noise = int(condition_mask.sum() * flip_prob)

        # Randomly select indices to noise
        indices_to_noise = np.random.choice(
            X_train.index[condition_mask],
            size=num_to_noise,
            replace=False
        )

        # Apply noise by flipping labels
        noised_y_train = y_train.copy()
        noised_y_train.loc[indices_to_noise] = 1 - noised_y_train.loc[indices_to_noise]

        print(f"Simple conditional pollution applied:")
        print(f"Condition met by {condition_mask.sum()} samples")
        print(f"Flipped {num_to_noise} labels")

        return noised_y_train