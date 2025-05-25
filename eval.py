from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score, confusion_matrix
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
import numpy as np
from scipy.stats import entropy
import pandas as pd
from collections import Counter
from typing import Dict, Tuple, Union


def predictive_parity_difference(y_true, y_pred, sensitive_features):
    """
    Calculate the predictive parity difference between groups.

    Predictive parity is satisfied when the positive predictive value (precision)
    is the same across all groups.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        sensitive_features: Sensitive attribute values

    Returns:
        Difference in positive predictive values between groups
    """
    unique_groups = np.unique(sensitive_features)
    if len(unique_groups) != 2:
        raise ValueError("Predictive parity difference currently only supports binary sensitive attributes")

    ppvs = []
    for group in unique_groups:
        group_mask = sensitive_features == group
        group_y_true = y_true[group_mask]
        group_y_pred = y_pred[group_mask]

        # Calculate positive predictive value (precision)
        tp = np.sum((group_y_true == 1) & (group_y_pred == 1))
        fp = np.sum((group_y_true == 0) & (group_y_pred == 1))

        if tp + fp == 0:
            ppv = 0  # No positive predictions
        else:
            ppv = tp / (tp + fp)
        ppvs.append(ppv)

    return abs(ppvs[1] - ppvs[0])


def equality_of_opportunity_difference(y_true, y_pred, sensitive_features):
    """
    Calculate the equality of opportunity difference between groups.

    Equality of opportunity is satisfied when the true positive rate (sensitivity/recall)
    is the same across all groups.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        sensitive_features: Sensitive attribute values

    Returns:
        Difference in true positive rates between groups
    """
    unique_groups = np.unique(sensitive_features)
    if len(unique_groups) != 2:
        raise ValueError("Equality of opportunity difference currently only supports binary sensitive attributes")

    tprs = []
    for group in unique_groups:
        group_mask = sensitive_features == group
        group_y_true = y_true[group_mask]
        group_y_pred = y_pred[group_mask]

        # Calculate true positive rate (recall/sensitivity)
        tp = np.sum((group_y_true == 1) & (group_y_pred == 1))
        fn = np.sum((group_y_true == 1) & (group_y_pred == 0))

        if tp + fn == 0:
            tpr = 0  # No actual positives
        else:
            tpr = tp / (tp + fn)
        tprs.append(tpr)

    return abs(tprs[1] - tprs[0])


def get_eval_subgroup_metrics_biased(experiment: str) -> Tuple[str, Union[float, int]]:
    """Return the sensitive attribute and its minority value for biased noise evaluation."""
    if experiment == 'college_scorecard':
        return 'veteran', -1.0
    if experiment == 'diabetes_readmission':
        return 'race', 0
    if experiment == 'brfss_diabetes':
        return 'BMI5CAT_40', 1
    if experiment in ['acsincome', 'acsfoodstamps', 'acspubcov', 'acsunemployment']:
        return 'RAC1P', 0
    if experiment == 'nhanes_lead':
        return 'DMDMARTL_50', 1
    if experiment == 'heloc':
        return 'MaxDelqEver_-90', 1
    if experiment == 'anes':
        return 'VCF0302_no answer', 1
    if experiment == 'mimic_extract_mort_hosp':
        return 'age', 0.3391076864470278
    if experiment == 'physionet':
        return 'HR', -1.0
    if experiment == 'assistments':
        return 'attempt_count', 0
    if experiment == 'brfss_blood_pressure':
        return 'POVERTY', 1.732280272343581
    if experiment == 'mimic_extract_los_3':
        return 'age', 0.33453213776958474


def get_eval_subgroup_metrics_combo(experiment: str) -> Dict[str, Tuple[str, Union[float, int]]]:
    """Return the two sensitive attributes and their minority values for combo noise evaluation."""
    combo_attributes = {
        'diabetes_readmission': {
            'attr1': ('race', 0),
            'attr2': ('gender', 0)
        },
        'acsunemployment': {
            'attr1': ('RAC1P', 0),
            'attr2': ('MIG_01', 0)
        },
        'acspubcov': {
            'attr1': ('RAC1P', 0),
            'attr2': ('ESR_6', 1)
        },
        'acsincome': {
            'attr1': ('RAC1P', 0),
            'attr2': ('AGEP', 50)
        },
        'acsfoodstamps': {
            'attr1': ('RAC1P', 0),
            'attr2': ('POVPIP', 100)
        },
        'brfss_diabetes': {
            'attr1': ('BMI5CAT_40', 1),
            'attr2': ('SMOKDAY2_30', 1)
        },
        'brfss_blood_pressure': {
            'attr1': ('POVERTY', 1.732280272343581),
            'attr2': ('_AGE80', 65)
        },
        'assistments': {
            'attr1': ('attempt_count', 0),
            'attr2': ('hint_count', 0.1)
        },
        'physionet': {
            'attr1': ('HR', -1.0),
            'attr2': ('Temp', -1.0)
        },
        'mimic_extract_los_3': {
            'attr1': ('age', 0.7),
            'attr2': ('gender', 0)
        },
        'mimic_extract_mort_hosp': {
            'attr1': ('age', 0.7),
            'attr2': ('gender', 0)
        },
        'college_scorecard': {
            'attr1': ('veteran', -1.0),
            'attr2': ('agege24', 0.5)
        },
        'anes': {
            'attr1': ('VCF0302_no answer', 1),
            'attr2': ('no_answer_count', 10)
        },
        'heloc': {
            'attr1': ('MaxDelqEver_-90', 1),
            'attr2': ('NumTotalTrades', 1)
        },
        'nhanes_lead': {
            'attr1': ('DMDBORN4_MISSING', 1),
            'attr2': ('DMDMARTL_50', 1.0)
        }
    }

    if experiment not in combo_attributes:
        raise ValueError(f"No combo attributes defined for experiment: {experiment}")

    return combo_attributes[experiment]


def create_subgroup_masks(X: pd.DataFrame, experiment: str, noise_type: str) -> Dict[str, pd.Series]:
    """
    Create subgroup masks based on the noise type and experiment.

    Args:
        X: Feature dataframe
        experiment: Name of the experiment
        noise_type: Type of noise ('biased', 'correlated', 'concatenated')

    Returns:
        Dictionary of subgroup masks
    """
    if noise_type == 'biased':
        # Single attribute evaluation
        attr, min_val = get_eval_subgroup_metrics_biased(experiment)

        if experiment == 'mimic_extract_los_3' or experiment == 'mimic_extract_mort_hosp':
            # Special case for age quantile
            threshold = X[attr].quantile(0.7)
            minority_mask = X[attr] > threshold
        elif experiment in ['acsincome', 'acsfoodstamps'] and 'AGEP' in attr:
            minority_mask = X[attr] > min_val
        elif experiment == 'acsfoodstamps' and 'POVPIP' in attr:
            minority_mask = X[attr] < min_val
        elif experiment == 'college_scorecard':
            minority_mask = X[attr] != min_val
        else:
            minority_mask = X[attr] == min_val

        return {
            'minorities': minority_mask,
            'majorities': ~minority_mask
        }

    elif noise_type in ['correlated', 'concatenated']:
        # Dual attribute evaluation
        combo_attrs = get_eval_subgroup_metrics_combo(experiment)
        attr1_name, attr1_val = combo_attrs['attr1']
        attr2_name, attr2_val = combo_attrs['attr2']

        # Create individual condition masks
        mask1 = create_single_condition_mask(X, experiment, attr1_name, attr1_val)
        mask2 = create_single_condition_mask(X, experiment, attr2_name, attr2_val)

        if noise_type == 'correlated':
            # AND combination
            both_conditions = mask1 & mask2
            return {
                'both_conditions': both_conditions,
                'condition1_only': mask1 & ~mask2,
                'condition2_only': mask2 & ~mask1,
                'neither_condition': ~mask1 & ~mask2,
                'any_condition': mask1 | mask2,  # For comparison
                'all_conditions': both_conditions  # Alias
            }
        else:  # concatenated
            # OR combination
            any_condition = mask1 | mask2
            return {
                'any_condition': any_condition,
                'both_conditions': mask1 & mask2,
                'condition1_only': mask1 & ~mask2,
                'condition2_only': mask2 & ~mask1,
                'neither_condition': ~mask1 & ~mask2,
                'all_conditions': mask1 & mask2  # For comparison
            }

    else:
        raise ValueError(f"Unsupported noise type: {noise_type}")


def create_single_condition_mask(X: pd.DataFrame, experiment: str, attr_name: str,
                                 attr_val: Union[float, int]) -> pd.Series:
    """Create a single condition mask for a given attribute."""

    # Handle special cases
    if attr_name == 'no_answer_count':
        # Special case for ANES - count no answer columns
        no_answer_columns = [col for col in X.columns if 'no answer' in col]
        no_answer_counts = X[no_answer_columns].sum(axis=1)
        return no_answer_counts > attr_val

    elif experiment in ['mimic_extract_los_3', 'mimic_extract_mort_hosp'] and attr_name == 'age':
        # Special case for age quantile
        threshold = X[attr_name].quantile(attr_val)
        return X[attr_name] > threshold

    elif experiment == 'acsincome' and attr_name == 'AGEP':
        return X[attr_name] > attr_val

    elif experiment == 'acsfoodstamps' and attr_name == 'POVPIP':
        return X[attr_name] < attr_val

    elif experiment == 'brfss_blood_pressure' and attr_name == '_AGE80':
        return X[attr_name] > attr_val

    elif experiment == 'assistments' and attr_name in ['attempt_count', 'hint_count']:
        return X[attr_name] > attr_val

    elif experiment == 'college_scorecard' and attr_name in ['veteran', 'agege24']:
        if attr_name == 'veteran':
            return X[attr_name] != attr_val
        else:  # agege24
            return X[attr_name] > attr_val

    elif experiment == 'heloc' and attr_name == 'NumTotalTrades':
        return X[attr_name] > attr_val

    elif experiment == 'brfss_diabetes' and attr_name == 'SMOKDAY2_30':
        # Handle fallback for smoking variable
        if 'SMOKDAY2_30' in X.columns:
            return X['SMOKDAY2_30'] == attr_val
        elif 'SMOKDAY2_1' in X.columns:
            return X['SMOKDAY2_1'] == attr_val
        else:
            return pd.Series([False] * len(X), index=X.index)

    else:
        # Standard equality check
        return X[attr_name] == attr_val


def evaluate_subgroups(model, X: pd.DataFrame, y: pd.Series, split: str,
                       subgroup_masks: Dict[str, pd.Series]) -> Dict[str, float]:
    """
    Evaluate model performance on different subgroups.

    Args:
        model: The trained model
        X: Features dataframe
        y: Target series
        split: Data split name
        subgroup_masks: Dictionary of subgroup masks

    Returns:
        Dictionary of subgroup metrics
    """
    # Get predictions
    if hasattr(model, "predict_proba"):
        yhat_probs = model.predict_proba(X)
        yhat_soft = yhat_probs[:, 1]
        yhat_hard = model.predict(X)
    else:
        yhat_soft = model.predict(X)
        yhat_hard = (yhat_soft >= 0.5).astype(int)

    metrics = {}

    for group_name, mask in subgroup_masks.items():
        if mask.sum() == 0:  # Skip empty groups
            continue

        subgroup_y = y[mask]
        subgroup_yhat_hard = yhat_hard[mask]
        subgroup_yhat_soft = yhat_soft[mask]

        # Calculate metrics
        try:
            entropy_vals_subgroup = entropy(subgroup_yhat_soft, base=2)

            metrics[f"{split}_{group_name}_accuracy"] = accuracy_score(subgroup_y, subgroup_yhat_hard)
            metrics[f"{split}_{group_name}_f1"] = f1_score(subgroup_y, subgroup_yhat_hard,
                                                           average='binary', zero_division=0)
            metrics[f"{split}_{group_name}_auc"] = roc_auc_score(subgroup_y, subgroup_yhat_soft)

            # Confusion matrix metrics
            tn, fp, fn, tp = confusion_matrix(subgroup_y, subgroup_yhat_hard).ravel()

            selection_rate = (tp + fp) / len(subgroup_y) if len(subgroup_y) != 0 else 0

            metrics[f"{split}_{group_name}_TP"] = tp
            metrics[f"{split}_{group_name}_TN"] = tn
            metrics[f"{split}_{group_name}_FP"] = fp
            metrics[f"{split}_{group_name}_FN"] = fn
            metrics[f"{split}_{group_name}_Selection_Rate"] = selection_rate
            metrics[f"{split}_{group_name}_size"] = len(subgroup_y)
            metrics[f"{split}_{group_name}_mean_entropy"] = np.mean(entropy_vals_subgroup)

        except Exception as e:
            print(f"Warning: Could not calculate metrics for {group_name}: {e}")
            metrics[f"{split}_{group_name}_size"] = len(subgroup_y)

    return metrics


def evaluate(model, X: pd.DataFrame, y: pd.Series, split: str, experiment: str,
             noise_type: str = 'biased', sensitive_attr=None) -> dict:
    """
    Evaluate model performance with fairness metrics for different noise types.

    Args:
        model: The trained model to evaluate
        X: Features dataframe
        y: Target series
        split: Data split name (e.g., 'id_test', 'ood_test')
        experiment: Name of the experiment
        noise_type: Type of noise ('biased', 'correlated', 'concatenated')
        sensitive_attr: Sensitive attribute column used for fairness metrics (for biased only)

    Returns:
        Dictionary of metrics
    """
    # Get predictions
    if hasattr(model, "predict_proba"):
        yhat_probs = model.predict_proba(X)
        yhat_soft = yhat_probs[:, 1]
        yhat_hard = model.predict(X)
    else:
        yhat_soft = model.predict(X)
        yhat_hard = (yhat_soft >= 0.5).astype(int)

    entropy_vals = entropy(yhat_soft, base=2)

    # Overall metrics
    metrics = {}
    metrics[f"{split}_accuracy"] = accuracy_score(y, yhat_hard)
    metrics[f"{split}_auc"] = roc_auc_score(y, yhat_soft)
    metrics[f"{split}_map"] = average_precision_score(y, yhat_soft)
    metrics[f"{split}_f1"] = f1_score(y, yhat_hard)
    metrics[f"{split}_selection_rate"] = np.mean(yhat_hard == 1)
    metrics[f"{split}_num_samples"] = len(y)
    metrics[f"{split}_mean_entropy"] = np.mean(entropy_vals)

    # Add fairness metrics if we're evaluating on test sets
    if split in ['id_test', 'ood_test', 'test']:
        # Create subgroup masks based on noise type
        subgroup_masks = create_subgroup_masks(X, experiment, noise_type)

        # Calculate subgroup metrics
        subgroup_metrics = evaluate_subgroups(model, X, y, split, subgroup_masks)
        metrics.update(subgroup_metrics)

        # Add global fairness metrics for biased noise (backward compatibility)
        if noise_type == 'biased':
            minority_mask = subgroup_masks['minorities']

            if sensitive_attr is None:
                sensitive_attr = minority_mask

            # Add global fairness metrics
            if len(np.unique(sensitive_attr)) > 1:
                try:
                    metrics[f"{split}_demographic_parity_diff"] = demographic_parity_difference(
                        y_true=y,
                        y_pred=yhat_hard,
                        sensitive_features=sensitive_attr
                    )
                except:
                    metrics[f"{split}_demographic_parity_diff"] = np.nan

                try:
                    metrics[f"{split}_equalized_odds_diff"] = equalized_odds_difference(
                        y_true=y,
                        y_pred=yhat_hard,
                        sensitive_features=sensitive_attr
                    )
                except:
                    metrics[f"{split}_equalized_odds_diff"] = np.nan

                # Add predictive parity and equality of opportunity
                try:
                    metrics[f"{split}_predictive_parity_diff"] = predictive_parity_difference(
                        y_true=y,
                        y_pred=yhat_hard,
                        sensitive_features=sensitive_attr
                    )
                except:
                    metrics[f"{split}_predictive_parity_diff"] = np.nan

                try:
                    metrics[f"{split}_equality_of_opportunity_diff"] = equality_of_opportunity_difference(
                        y_true=y,
                        y_pred=yhat_hard,
                        sensitive_features=sensitive_attr
                    )
                except:
                    metrics[f"{split}_equality_of_opportunity_diff"] = np.nan

        # For correlated/concatenated noise, add fairness metrics for key comparisons
        elif noise_type in ['correlated', 'concatenated']:
            # Compare the target group (both_conditions or any_condition) vs rest
            if noise_type == 'correlated':
                target_mask = subgroup_masks['both_conditions']
            else:  # concatenated
                target_mask = subgroup_masks['any_condition']

            if len(np.unique(target_mask)) > 1:
                try:
                    metrics[f"{split}_demographic_parity_diff"] = demographic_parity_difference(
                        y_true=y,
                        y_pred=yhat_hard,
                        sensitive_features=target_mask
                    )
                except:
                    metrics[f"{split}_demographic_parity_diff"] = np.nan

                try:
                    metrics[f"{split}_equalized_odds_diff"] = equalized_odds_difference(
                        y_true=y,
                        y_pred=yhat_hard,
                        sensitive_features=target_mask
                    )
                except:
                    metrics[f"{split}_equalized_odds_diff"] = np.nan

                # Add predictive parity and equality of opportunity for correlated/concatenated
                try:
                    metrics[f"{split}_predictive_parity_diff"] = predictive_parity_difference(
                        y_true=y,
                        y_pred=yhat_hard,
                        sensitive_features=target_mask
                    )
                except:
                    metrics[f"{split}_predictive_parity_diff"] = np.nan

                try:
                    metrics[f"{split}_equality_of_opportunity_diff"] = equality_of_opportunity_difference(
                        y_true=y,
                        y_pred=yhat_hard,
                        sensitive_features=target_mask
                    )
                except:
                    metrics[f"{split}_equality_of_opportunity_diff"] = np.nan

    return metrics


def print_evaluation_summary(metrics: Dict[str, float], split: str, noise_type: str):
    """Print a summary of evaluation results."""
    print(f"\n=== Evaluation Summary ({split}, {noise_type} noise) ===")

    # Overall metrics
    print(f"Overall Accuracy: {metrics.get(f'{split}_accuracy', 'N/A'):.4f}")
    print(f"Overall AUC: {metrics.get(f'{split}_auc', 'N/A'):.4f}")
    print(f"Overall F1: {metrics.get(f'{split}_f1', 'N/A'):.4f}")

    # Subgroup metrics
    subgroup_keys = [k for k in metrics.keys() if '_accuracy' in k and split in k and k != f'{split}_accuracy']

    if subgroup_keys:
        print(f"\nSubgroup Performance:")
        for key in sorted(subgroup_keys):
            group_name = key.replace(f'{split}_', '').replace('_accuracy', '')
            accuracy = metrics[key]
            size_key = f'{split}_{group_name}_size'
            size = metrics.get(size_key, 'N/A')
            print(f"  {group_name}: Accuracy={accuracy:.4f}, Size={size}")

    # Fairness metrics
    dp_diff = metrics.get(f'{split}_demographic_parity_diff', None)
    eo_diff = metrics.get(f'{split}_equalized_odds_diff', None)
    pp_diff = metrics.get(f'{split}_predictive_parity_diff', None)
    eop_diff = metrics.get(f'{split}_equality_of_opportunity_diff', None)

    if any(metric is not None for metric in [dp_diff, eo_diff, pp_diff, eop_diff]):
        print(f"\nFairness Metrics:")
        if dp_diff is not None:
            print(f"  Demographic Parity Difference: {dp_diff:.4f}")
        if eo_diff is not None:
            print(f"  Equalized Odds Difference: {eo_diff:.4f}")
        if pp_diff is not None:
            print(f"  Predictive Parity Difference: {pp_diff:.4f}")
        if eop_diff is not None:
            print(f"  Equality of Opportunity Difference: {eop_diff:.4f}")