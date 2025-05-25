"""
Usage:

python scripts/train_multi_model_optuna \
    --experiment adult \
    --model catboost \
    --use_gpu \
    --use_cached

python scripts/train_multi_model_optuna \
    --experiment adult \
    --model faireg \
    --constraint_type demographic_parity \
    --noise_type biased \
    --flip_prob 0.1 \
    --cleaning_method custom \
    --cleaning_params '{"custom_param": 0.3}'

"""
import argparse
import logging
import os
import time
from collections import Counter

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from fairlearn.reductions import ExponentiatedGradient, GridSearch
from add.ReweightingGridSearch import FairRS
from fairlearn.reductions import DemographicParity, EqualizedOdds, ErrorRateParity, TruePositiveRateParity, \
    FalsePositiveRateParity

from typing import Optional, Tuple, Dict

from tableshift.tableshift.core.utils import timestamp_as_int
from pollute.main_pollute import DatasetNoiseInjector
from add.cleaning_algo import DataCleaningMethod, CustomCleaningMethod
from add.ReweightingGridSearch import FairRS
from eval import evaluate, get_eval_subgroup_metrics_biased

def get_cleaning_method(cleaning_method: str, **cleaning_params) -> Optional[DataCleaningMethod]:
    """Get cleaning method instance based on method name."""
    if cleaning_method is None or cleaning_method == "none":
        return None

    cleaning_methods = {
        'custom': CustomCleaningMethod,
        # Add other cleaning methods here as you implement them
        # 'isolation_forest': IsolationForestCleaning,
        # 'local_outlier_factor': LOFCleaning,
        # etc.
    }

    if cleaning_method not in cleaning_methods:
        available_methods = list(cleaning_methods.keys()) + ["none"]
        raise ValueError(f"Unknown cleaning method: {cleaning_method}. Available: {available_methods}")

    # Initialize cleaning method with parameters
    cleaning_class = cleaning_methods[cleaning_method]
    return cleaning_class(**cleaning_params)


def apply_data_cleaning(X_tr: pd.DataFrame, y_tr: pd.Series,
                        sensitive_features_tr: Optional[pd.Series],
                        cleaning_method: Optional[DataCleaningMethod]) -> Tuple[
    pd.DataFrame, pd.Series, Optional[pd.Series], Dict]:
    """
    Apply data cleaning if specified.

    Returns:
        Cleaned X_tr, y_tr, sensitive_features_tr (if provided), and cleaning stats
    """
    cleaning_stats = {}

    if cleaning_method is None:
        cleaning_stats = {
            'cleaning_applied': False,
            'total_samples': len(y_tr),
            'samples_removed': 0,
            'samples_remaining': len(y_tr),
            'noise_rate': 0.0
        }
        return X_tr, y_tr, sensitive_features_tr, cleaning_stats

    print(f"Applying data cleaning with method: {cleaning_method.name}")
    print(f"Training samples before cleaning: {len(y_tr)}")

    # Apply cleaning
    X_tr_clean, y_tr_clean = cleaning_method.clean_data(X_tr, y_tr)

    # Also clean sensitive features if provided
    sensitive_features_tr_clean = None
    if sensitive_features_tr is not None:
        # Get the indices that were kept after cleaning
        clean_indices = X_tr_clean.index if hasattr(X_tr_clean, 'index') else range(len(X_tr_clean))
        if hasattr(sensitive_features_tr, 'iloc'):
            # If sensitive_features_tr is a pandas Series, use iloc
            original_indices = np.arange(len(sensitive_features_tr))
            keep_mask = ~np.isin(original_indices, cleaning_method.noise_indices)
            sensitive_features_tr_clean = sensitive_features_tr[keep_mask].reset_index(drop=True)
        else:
            # If it's a numpy array
            keep_mask = ~np.isin(np.arange(len(sensitive_features_tr)), cleaning_method.noise_indices)
            sensitive_features_tr_clean = sensitive_features_tr[keep_mask]

    # Update cleaning stats
    cleaning_stats = cleaning_method.cleaning_stats.copy()
    cleaning_stats['cleaning_applied'] = True
    cleaning_stats['method_name'] = cleaning_method.name

    print(f"Training samples after cleaning: {len(y_tr_clean)}")
    print(f"Samples removed: {cleaning_stats['samples_removed']}")
    print(f"Noise rate detected: {cleaning_stats['noise_rate']:.3f}")

    return X_tr_clean, y_tr_clean, sensitive_features_tr_clean, cleaning_stats
    """Get fairness constraint based on type."""
    constraints = {
        'demographic_parity': DemographicParity(),
        'equalized_odds': EqualizedOdds(),
        'error_rate_parity': ErrorRateParity(),
        'equal_opportunity': TruePositiveRateParity(),
        'predictive_equality': FalsePositiveRateParity()
    }

    if constraint_type not in constraints:
        raise ValueError(f"Unknown constraint type: {constraint_type}. Available: {list(constraints.keys())}")

    return constraints[constraint_type]


def create_model_and_params(model_name: str, trial: optuna.trial.Trial, use_gpu: bool,
                            random_seed: int, constraint_type: str = None):
    """Create model instance based on model name and trial parameters."""

    if model_name == "catboost":
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1., log=True),
            'depth': trial.suggest_int('depth', 3, 10),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 1e-6, 1., log=True),
            'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 1, 100, log=True),
            'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations', 1, 10),
            "use_best_model": True,
            "task_type": "GPU" if use_gpu else "CPU",
            'random_seed': random_seed,
        }
        return CatBoostClassifier(**params)

    elif model_name in ["faireg", "fairgs", "fairrs"]:
        # Base classifier for fairness models
        base_classifier = LGBMClassifier(
            objective='binary',
            n_estimators=trial.suggest_int('n_estimators', 50, 200),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
            max_depth=trial.suggest_int('max_depth', 3, 10),
            random_state=random_seed,
            metric='auc'
        )

        # Get constraint
        constraint = get_fairness_constraint(constraint_type)

        if model_name == "faireg":
            return ExponentiatedGradient(
                estimator=base_classifier,
                constraints=constraint,
                eps=trial.suggest_float('eps', 0.001, 0.1, log=True),
                max_iter=trial.suggest_int('max_iter', 20, 100)
            )

        elif model_name == "fairgs":
            return GridSearch(
                estimator=base_classifier,
                constraints=constraint,
                grid_size=trial.suggest_int('grid_size', 5, 20),
                grid_limit=trial.suggest_float('grid_limit', 1.0, 3.0),
                grid_offset=trial.suggest_float('grid_offset', 0.001, 0.1, log=True)
            )

        elif model_name == "fairrs":
            # Note: You'll need to implement FairRS or replace with another model
            # For now, using GridSearch as placeholder
            return GridSearch(
                estimator=base_classifier,
                constraints=constraint,
                grid_size=trial.suggest_int('grid_size', 5, 20),
                grid_limit=trial.suggest_float('grid_limit', 1.0, 3.0)
            )

    else:
        raise ValueError(f"Unknown model: {model_name}")


def fit_model(model, model_name: str, X_tr, y_tr, X_val, y_val, sensitive_features_tr=None,
              sensitive_features_val=None):
    """Fit model based on its type."""

    if model_name == "catboost":
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)

    elif model_name in ["faireg", "fairgs", "fairrs"]:
        if sensitive_features_tr is None:
            raise ValueError("Sensitive features required for fairness models")
        model.fit(X_tr, y_tr, sensitive_features=sensitive_features_tr)

    return model


def main(experiment: str, cache_dir: str, results_dir: str, num_samples: int,
         random_seed: int, use_gpu: bool, use_cached: bool, model: str,
         constraint_type: str = "demographic_parity", flip_prob: float = 0.0,
         noise_type: str = "random", cleaning_method: str = "none",
         cleaning_params: Dict = None):
    if cleaning_params is None:
        cleaning_params = {}

    start_time = timestamp_as_int()

    # Initialize the noise injector
    injector = DatasetNoiseInjector()

    # Get dataset with noise injection
    dset = injector.get_noisy_dataset(
        dataset_name=experiment,
        noise_type=noise_type,
        flip_prob=flip_prob
    )

    uid = dset.uid

    # Get data splits
    X_tr, y_tr, _, _ = dset.get_pandas("train")
    X_val, y_val, _, _ = dset.get_pandas("validation")

    # Get sensitive features for fairness models
    sensitive_features_tr = None
    sensitive_features_val = None
    sensitive_attr = None

    if model in ["faireg", "fairgs", "fairrs"]:
        try:
            sensitive_attr, min_val = get_eval_subgroup_metrics_biased(experiment)
            sensitive_features_tr = X_tr[sensitive_attr].copy()
            sensitive_features_val = X_val[sensitive_attr].copy()

            print("Y train", Counter(y_tr.tolist()))
            print(f"Sensitive attribute: {sensitive_attr}")
            print("Sensitive features train distribution:", Counter(sensitive_features_tr.tolist()))

        except NameError:
            print("Warning: get_eval_subgroup_metrics not available. You need to specify sensitive features manually.")
            # You'll need to specify the sensitive attribute for your dataset
            # sensitive_attr = "your_sensitive_column_name"  # e.g., "sex", "race", etc.
            raise ValueError("Sensitive attribute must be specified for fairness models")

    # Apply data cleaning if specified
    cleaner = get_cleaning_method(cleaning_method, **cleaning_params)
    X_tr, y_tr, sensitive_features_tr, cleaning_stats = apply_data_cleaning(
        X_tr, y_tr, sensitive_features_tr, cleaner
    )

    print(f"Final training set size: {len(y_tr)}")
    print("Cleaning stats:", cleaning_stats)

    def optimize_hp(trial: optuna.trial.Trial):
        model_instance = create_model_and_params(model, trial, use_gpu, random_seed, constraint_type)

        # Fit model
        model_instance = fit_model(
            model_instance, model, X_tr, y_tr, X_val, y_val,
            sensitive_features_tr, sensitive_features_val
        )

        # Predict and evaluate
        y_pred = model_instance.predict(X_val)
        return accuracy_score(y_val, y_pred)

    # Run optimization
    study = optuna.create_study(direction="maximize")
    study.optimize(optimize_hp, n_trials=num_samples)

    print('Trials:', len(study.trials))
    print('Best parameters:', study.best_trial.params)
    print('Best score:', study.best_value)
    print("Training completed! Retraining model with best params and evaluating it.")

    # Create final model with best parameters
    # Create a mock trial object for the best parameters
    class MockTrial:
        def __init__(self, params):
            self.params = params

        def suggest_float(self, name, low, high, log=False):
            return self.params[name]

        def suggest_int(self, name, low, high, log=False):
            return self.params[name]

    mock_trial = MockTrial(study.best_trial.params)
    clf_with_best_params = create_model_and_params(model, mock_trial, use_gpu, random_seed, constraint_type)

    # Fit final model
    clf_with_best_params = fit_model(
        clf_with_best_params, model, X_tr, y_tr, X_val, y_val,
        sensitive_features_tr, sensitive_features_val
    )

    # Setup results directory
    expt_results_dir = os.path.join(results_dir, experiment, str(start_time))

    # Prepare metrics
    metrics = {}
    metrics["estimator"] = model
    metrics["domain_split_varname"] = dset.domain_split_varname
    metrics["domain_split_ood_values"] = str(dset.get_domains("ood_test"))
    metrics["domain_split_id_values"] = str(dset.get_domains("id_test"))

    if model in ["faireg", "fairgs", "fairrs"]:
        metrics["constraint_type"] = constraint_type
        metrics["noise_rate"] = flip_prob
        metrics["noise_type"] = noise_type
        metrics["sensitive_attribute"] = sensitive_attr

    # Add cleaning statistics to metrics
    metrics.update({f"cleaning_{k}": v for k, v in cleaning_stats.items()})

    # Evaluate on different splits
    splits = (
        'id_test', 'ood_test', 'ood_validation', 'validation') if dset.is_domain_split else (
        'test', 'validation')

    for split in splits:
        X, y, _, _ = dset.get_pandas(split)

        # Get sensitive features for fairness models
        sensitive_features_split = None
        if model in ["faireg", "fairgs", "fairrs"] and sensitive_attr:
            sensitive_features_split = X[sensitive_attr].copy()

        # Evaluate model
        if model in ["faireg", "fairgs", "fairrs"]:
            _metrics = evaluate(clf_with_best_params, X, y, split, experiment,
                                sensitive_attr=sensitive_features_split)
        else:
            _metrics = evaluate(clf_with_best_params, X, y, split)

        print(_metrics)
        metrics.update(_metrics)

    # Save results
    noise_suffix = f"_noise_{noise_type}_{flip_prob}" if flip_prob > 0 else ""
    cleaning_suffix = f"_clean_{cleaning_method}" if cleaning_method != "none" else ""
    model_suffix = f"_{constraint_type}" if model in ["faireg", "fairgs", "fairrs"] else ""
    iter_fp = os.path.join(
        expt_results_dir,
        f"tune_results_{experiment}_{start_time}_{uid[:100]}_"
        f"{model}{model_suffix}{noise_suffix}{cleaning_suffix}.csv")

    if not os.path.exists(expt_results_dir):
        os.makedirs(expt_results_dir)

    logging.info(f"Writing results for {model} to {iter_fp}")
    pd.DataFrame(metrics, index=[1]).to_csv(iter_fp, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="tmp",
                        help="Directory to cache raw data files to.")
    parser.add_argument("--experiment", default="adult",
                        help="Experiment to run.")
    parser.add_argument("--model", default="catboost",
                        choices=["catboost", "faireg", "fairgs", "fairrs"],
                        help="Model to use for training")
    parser.add_argument("--constraint_type", default="demographic_parity",
                        choices=["demographic_parity", "equalized_odds", "error_rate_parity",
                                 "equal_opportunity", "predictive_equality"],
                        help="Fairness constraint type (only for fairness models)")
    parser.add_argument("--noise_type", default="random",
                        choices=["random", "biased", "correlated", "concatenated", "age_based"],
                        help="Type of noise injection")
    parser.add_argument("--flip_prob", type=float, default=0.0,
                        help="Label noise probability")
    parser.add_argument("--cleaning_method", default="none",
                        choices=["none", "custom"],  # Add more as you implement them
                        help="Data cleaning method to apply")
    parser.add_argument("--cleaning_params", type=str, default="{}",
                        help="JSON string of parameters for cleaning method (e.g., '{\"custom_param\": 0.3}')")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of hyperparameter samples to take in tuning sweep.")
    parser.add_argument("--results_dir", default="./optuna_results",
                        help="Where to write results.")
    parser.add_argument("--random_seed", default=42, type=int)
    parser.add_argument("--use_cached", default=False, action="store_true",
                        help="Whether to use cached data.")
    parser.add_argument("--use_gpu", action="store_true", default=False,
                        help="Whether to use GPU (if available)")

    args = parser.parse_args()

    # Parse cleaning parameters from JSON string
    import json

    try:
        cleaning_params = json.loads(args.cleaning_params)
    except json.JSONDecodeError:
        print(f"Warning: Invalid JSON for cleaning_params: {args.cleaning_params}")
        print("Using empty parameters...")
        cleaning_params = {}

    # Validate arguments
    if args.model in ["faireg", "fairgs", "fairrs"] and not args.constraint_type:
        parser.error("--constraint_type is required for fairness models")

    # Remove cleaning_params from args dict and pass separately
    args_dict = vars(args)
    args_dict.pop('cleaning_params')
    args_dict['cleaning_params'] = cleaning_params

    main(**args_dict)