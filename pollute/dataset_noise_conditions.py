"""
Dataset-specific condition functions for noise injection.

This module contains two types of condition functions:
1. Biased conditions: Single condition functions for each dataset
2. Combo conditions: Multiple conditions combined with AND/OR for correlated/concatenated noise
"""

import pandas as pd
from collections import Counter
from typing import Callable


def get_biased_condition_function(dataset_name: str) -> Callable[[pd.DataFrame], pd.Series]:
    """
    Get the biased (single) condition function for a given dataset.

    Args:
        dataset_name: Name of the dataset

    Returns:
        Condition function that takes X_train and returns a boolean mask
    """
    biased_functions = {
        'diabetes_readmission': conditions_readmission_biased,
        'acsunemployment': conditions_acsincome_biased,
        'acspubcov': conditions_acsincome_biased,
        'acsincome': conditions_acsincome_biased,
        'acsfoodstamps': conditions_acsincome_biased,
        'brfss_diabetes': conditions_brfss_biased,
        'brfss_blood_pressure': conditions_brfss_blood_pressure_biased,
        'nhanes_lead': conditions_nhanes_lead_biased,
        'heloc': conditions_heloc_biased,
        'college_scorecard': conditions_college_scorecard_biased,
        'anes': conditions_anes_biased,
        'mimic_extract_los_3': conditions_mimic_los_biased,
        'mimic_extract_mort_hosp': conditions_mimic_los_biased,
        'physionet': conditions_physionet_biased,
        'assistments': conditions_assistments_biased,
    }

    if dataset_name not in biased_functions:
        raise ValueError(f"No biased condition function defined for dataset: {dataset_name}")

    return biased_functions[dataset_name]


def get_combo_condition_function(dataset_name: str) -> Callable[[pd.DataFrame, str], pd.Series]:
    """
    Get the combo (multiple) condition function for a given dataset.

    Args:
        dataset_name: Name of the dataset

    Returns:
        Condition function that takes X_train and connector ('and'/'or') and returns a boolean mask
    """
    combo_functions = {
        'diabetes_readmission': conditions_readmission_combo,
        'acsunemployment': conditions_acsunemployment_combo,
        'acspubcov': conditions_acspubcov_combo,
        'acsincome': conditions_acsincome_combo,
        'acsfoodstamps': conditions_acsfoodstamps_combo,
        'brfss_diabetes': conditions_brfss_combo,
        'brfss_blood_pressure': conditions_brfss_blood_pressure_combo,
        'nhanes_lead': conditions_nhanes_lead_combo,
        'heloc': conditions_heloc_combo,
        'college_scorecard': conditions_college_scorecard_combo,
        'anes': conditions_anes_combo,
        'mimic_extract_los_3': conditions_mimic_los_combo,
        'mimic_extract_mort_hosp': conditions_mimic_los_combo,
        'physionet': conditions_physionet_combo,
        'assistments': conditions_assistments_combo,
    }

    if dataset_name not in combo_functions:
        raise ValueError(f"No combo condition function defined for dataset: {dataset_name}")

    return combo_functions[dataset_name]


# ============================================================================
# BIASED CONDITIONS (Single conditions for each dataset)
# ============================================================================

def conditions_readmission_biased(X_train: pd.DataFrame) -> pd.Series:
    """Biased condition for diabetes readmission dataset - race only."""
    condition_demographic_risk = (X_train['race'] == 0)
    print('Readmission biased condition count:', Counter(condition_demographic_risk))
    return condition_demographic_risk


def conditions_mimic_los_biased(X_train: pd.DataFrame) -> pd.Series:
    """Biased condition for MIMIC length of stay dataset - older patients."""
    condition_age = (X_train['age'] > X_train['age'].quantile(0.7))
    print('MIMIC age threshold:', X_train['age'].quantile(0.7))
    return condition_age


def conditions_acsincome_biased(X_train: pd.DataFrame) -> pd.Series:
    """Biased condition for ACS income-related datasets - minority groups."""
    minority = X_train['RAC1P'] == 0
    return minority


def conditions_brfss_biased(X_train: pd.DataFrame) -> pd.Series:
    """Biased condition for BRFSS diabetes dataset - high BMI."""
    high_bmi = (X_train['BMI5CAT_40'] == 1)
    print('BRFSS high BMI count:', Counter(high_bmi.tolist()))
    return high_bmi


def conditions_brfss_blood_pressure_biased(X_train: pd.DataFrame) -> pd.Series:
    """Biased condition for BRFSS blood pressure dataset - poverty indicator."""
    condition_mask = X_train['POVERTY'] == 1.732280272343581
    print('BRFSS blood pressure poverty count:', Counter(condition_mask))
    return condition_mask


def conditions_nhanes_lead_biased(X_train: pd.DataFrame) -> pd.Series:
    """Biased condition for NHANES lead dataset - marital status indicator."""
    demographic_info_missing = X_train['DMDMARTL_50'] == 1.0
    return demographic_info_missing


def conditions_physionet_biased(X_train: pd.DataFrame) -> pd.Series:
    """Biased condition for PhysioNet dataset - missing/error HR values."""
    print('PhysioNet HR distribution:', Counter(X_train['HR'].tolist()))
    error_conditions = X_train['HR'] == -1.0
    print('PhysioNet error conditions:', Counter(error_conditions))
    return error_conditions


def conditions_heloc_biased(X_train: pd.DataFrame) -> pd.Series:
    """Biased condition for HELOC dataset - delinquency history."""
    combined_mask = X_train['MaxDelqEver_-90'] == 1
    return combined_mask


def conditions_college_scorecard_biased(X_train: pd.DataFrame) -> pd.Series:
    """Biased condition for College Scorecard dataset - veteran status."""
    condition_demographic = X_train['veteran'] != -1.0
    return condition_demographic


def conditions_anes_biased(X_train: pd.DataFrame) -> pd.Series:
    """Biased condition for ANES dataset - missing answers."""
    condition_mask = X_train['VCF0302_no answer'] == 1
    print('ANES condition count:', Counter(condition_mask))
    return condition_mask


def conditions_assistments_biased(X_train: pd.DataFrame) -> pd.Series:
    """Biased condition for ASSISTments dataset - high attempt count."""
    high_usage_mask = X_train['attempt_count'] > 0
    return high_usage_mask


# ============================================================================
# COMBO CONDITIONS (Multiple conditions with AND/OR)
# ============================================================================

def conditions_readmission_combo(X_train: pd.DataFrame, connector: str = 'and') -> pd.Series:
    """Combo condition for diabetes readmission dataset - race and gender."""
    if connector == 'and':
        condition_demographic_risk = (X_train['race'] == 0) & (X_train['gender'] == 0)
    else:  # connector == 'or'
        condition_demographic_risk = (X_train['race'] == 0) | (X_train['gender'] == 0)

    print(f'Readmission combo ({connector}) count:', Counter(condition_demographic_risk))
    return condition_demographic_risk


def conditions_acsunemployment_combo(X_train: pd.DataFrame, connector: str = 'and') -> pd.Series:
    """Combo condition for ACS unemployment dataset - race and migration."""
    if connector == 'and':
        minority = (X_train['RAC1P'] == 0) & (X_train['MIG_01'] == 0)
    else:  # connector == 'or'
        minority = (X_train['RAC1P'] == 0) | (X_train['MIG_01'] == 0)
    return minority


def conditions_acspubcov_combo(X_train: pd.DataFrame, connector: str = 'and') -> pd.Series:
    """Combo condition for ACS public coverage dataset - race and employment status."""
    if connector == 'and':
        condition = (X_train['RAC1P'] == 0) & (X_train['ESR_6'] == 1)  # Minority and not in labor force
    else:  # connector == 'or'
        condition = (X_train['RAC1P'] == 0) | (X_train['ESR_6'] == 1)
    return condition


def conditions_acsincome_combo(X_train: pd.DataFrame, connector: str = 'and') -> pd.Series:
    """Combo condition for ACS income dataset - race and age."""
    if connector == 'and':
        condition = (X_train['RAC1P'] == 0) & (X_train['AGEP'] > 50)  # Minority and older
    else:  # connector == 'or'
        condition = (X_train['RAC1P'] == 0) | (X_train['AGEP'] > 50)
    return condition


def conditions_acsfoodstamps_combo(X_train: pd.DataFrame, connector: str = 'and') -> pd.Series:
    """Combo condition for ACS food stamps dataset - race and poverty status."""
    if connector == 'and':
        condition = (X_train['RAC1P'] == 0) & (X_train['POVPIP'] < 100)  # Minority and below poverty
    else:  # connector == 'or'
        condition = (X_train['RAC1P'] == 0) | (X_train['POVPIP'] < 100)
    return condition


def conditions_brfss_combo(X_train: pd.DataFrame, connector: str = 'and') -> pd.Series:
    """Combo condition for BRFSS diabetes dataset - BMI and smoking."""
    high_bmi = (X_train['BMI5CAT_40'] == 1)
    smoker = (X_train['SMOKDAY2_30'] == 1) if 'SMOKDAY2_30' in X_train.columns else (X_train['SMOKDAY2_1'] == 1)

    if connector == 'and':
        condition = high_bmi & smoker
    else:  # connector == 'or'
        condition = high_bmi | smoker
    return condition


def conditions_brfss_blood_pressure_combo(X_train: pd.DataFrame, connector: str = 'and') -> pd.Series:
    """Combo condition for BRFSS blood pressure dataset - poverty and age."""
    poverty_condition = X_train['POVERTY'] == 1.732280272343581
    age_condition = X_train['_AGE80'] > 65
    if connector == 'and':
        condition = poverty_condition & age_condition
    else:  # connector == 'or'
        condition = poverty_condition | age_condition
    return condition


def conditions_assistments_combo(X_train: pd.DataFrame, connector: str = 'and') -> pd.Series:
    '''High attempt or hint usage condition'''
    high_usage_mask = X_train['attempt_count'] > 0
    hint_mask = X_train['hint_count'] > 0.1
    if connector == 'and':
        condition = high_usage_mask & hint_mask
    else:  # connector == 'or'
        condition = high_usage_mask | hint_mask
    return condition

def conditions_physionet_combo(X_train: pd.DataFrame, connector: str = 'and') -> pd.Series:
    '''heart rate or temp failure'''
    heart_rate = X_train['HR'] == -1.0
    temp = X_train['Temp'] == -1.0
    if connector == 'and':
        condition = heart_rate & temp
    else:  # connector == 'or'
        condition = heart_rate | temp
    return condition

def conditions_mimic_los_combo(X_train: pd.DataFrame, connector: str = 'and') -> pd.Series:
    condition_age = (X_train['age'] > X_train['age'].quantile(0.7))
    gender = X_train['gender'] == 0
    if connector == 'and':
        condition = condition_age & gender
    else:  # connector == 'or'
        condition = condition_age | gender
    return condition

def conditions_college_scorecard_combo(X_train: pd.DataFrame, connector: str = 'and') -> pd.Series:
    condition_demographic = X_train['veteran'] != -1.0
    condition_age = (X_train['agege24'] > 0.5)
    if connector == 'and':
        condition = condition_demographic & condition_age
    else:  # connector == 'or'
        condition = condition_age | condition_age
    return condition

def conditions_anes_combo(X_train: pd.DataFrame, connector: str = 'and') -> pd.Series:
    no_answer_columns = [col for col in X_train.columns if 'no answer' in col]
    condition_mask = X_train['VCF0302_no answer'] == 1
    no_answer_counts = X_train[no_answer_columns].sum(axis=1) > 10
    if connector == 'and':
        condition = condition_mask & no_answer_counts
    else:  # connector == 'or'
        condition = condition_mask | no_answer_counts
    return condition


def conditions_heloc_combo(X_train: pd.DataFrame, connector: str = 'and') -> pd.Series:
    """Biased condition for HELOC dataset - delinquency history."""
    deliquency = X_train['MaxDelqEver_-90'] == 1
    totaltrades = X_train['NumTotalTrades'] > 1
    if connector == 'and':
        condition = deliquency & totaltrades
    else:  # connector == 'or'
        condition = deliquency | totaltrades
    return condition


def conditions_nhanes_lead_combo(X_train: pd.DataFrame, connector: str = 'and') -> pd.Series:
    demographic_info_missing = X_train['DMDBORN4_MISSING'] == 1
    marital_statuls = X_train['DMDMARTL_50'] == 1.0

    if connector == 'and':
        condition = demographic_info_missing & marital_statuls
    else:  # connector == 'or'
        condition = demographic_info_missing | marital_statuls
    return condition
