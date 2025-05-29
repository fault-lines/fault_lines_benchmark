# Dataset-Specific Noise Conditions: Rationale and Documentation

This document explains the specific conditions chosen for label noise injection in each dataset, reflecting real-world biases and data quality issues that can affect machine learning fairness.

## Overview

The noise conditions are designed to simulate realistic label corruption patterns that occur in practice due to:
- Systematic biases in data collection
- Human annotation errors correlated with demographic factors
- Missing or incomplete demographic information
- Measurement errors in medical/sensor data
- Historical discrimination patterns

## Dataset Conditions

### Healthcare Datasets

#### Diabetes Readmission (`diabetes_readmission`)
**Biased Condition**: Race (race == 0)
**Combo Conditions**: Race AND/OR Gender
**Rationale**: Healthcare disparities affecting minority populations are well-documented, with studies showing differential treatment and outcomes based on race. Gender-based medical biases also contribute to misdiagnosis patterns.

#### MIMIC ICU Length of Stay (`mimic_extract_los_3`) and Hospital Mortality (`mimic_extract_mort_hosp`)
**Biased Condition**: Age (age > 70th percentile)
**Combo Conditions**: Age AND/OR Gender
**Rationale**: Age-related biases in medical decision-making are common, with older patients potentially receiving different care standards. Combined with gender, this reflects intersectional healthcare disparities.

#### BRFSS Diabetes (`brfss_diabetes`)
**Biased Condition**: High BMI (BMI5CAT_40 == 1)
**Combo Conditions**: High BMI AND/OR Smoking Status
**Rationale**: Weight bias in healthcare leads to assumptions about diabetes risk, potentially causing over-diagnosis in high-BMI individuals while under-diagnosing others.

#### BRFSS Blood Pressure (`brfss_blood_pressure`)
**Biased Condition**: Poverty Status (POVERTY == specific value)
**Combo Conditions**: Poverty AND/OR Age (>65)
**Rationale**: Socioeconomic status affects healthcare access and quality, leading to systematic differences in diagnosis accuracy for hypertension.

#### NHANES Lead Exposure (`nhanes_lead`)
**Biased Condition**: Missing Marital Status (DMDMARTL_50 == 1.0)
**Combo Conditions**: Missing Demographics AND/OR Marital Status
**Rationale**: Missing demographic information often correlates with vulnerable populations, affecting the accuracy of health assessments.

#### PhysioNet Sepsis (`physionet`)
**Biased Condition**: Heart Rate Measurement Errors (HR == -1.0)
**Combo Conditions**: HR Errors AND/OR Temperature Errors
**Rationale**: Sensor malfunctions and measurement errors can lead to systematic misclassification, particularly affecting critical care decisions.

### Socioeconomic Datasets

#### ACS Income (`acsincome`)
**Biased Condition**: Race (RAC1P == 0)
**Combo Conditions**: Race AND/OR Age (>50)
**Rationale**: Historical and ongoing income discrimination based on race is well-documented. Age discrimination in employment also affects income reporting accuracy.

#### ACS Unemployment (`acsunemployment`)
**Biased Condition**: Race (RAC1P == 0)
**Combo Conditions**: Race AND/OR Migration Status
**Rationale**: Racial disparities in employment and unemployment benefits, combined with migration-based discrimination, affect data quality.

#### ACS Public Coverage (`acspubcov`)
**Biased Condition**: Race (RAC1P == 0)
**Combo Conditions**: Race AND/OR Employment Status
**Rationale**: Systematic barriers to public health coverage access for minority populations, compounded by employment status dependencies.

#### ACS Food Stamps (`acsfoodstamps`)
**Biased Condition**: Race (RAC1P == 0)
**Combo Conditions**: Race AND/OR Poverty Status
**Rationale**: Food assistance program access disparities and stigma affecting minority communities, particularly those below poverty line.

### Financial Datasets

#### FICO HELOC (`heloc`)
**Biased Condition**: Delinquency History (MaxDelqEver_-90 == 1)
**Combo Conditions**: Delinquency AND/OR Total Trades
**Rationale**: Credit history biases and discriminatory lending practices that disproportionately affect certain demographic groups.

### Educational Datasets

#### College Scorecard (`college_scorecard`)
**Biased Condition**: Veteran Status (veteran != -1.0)
**Combo Conditions**: Veteran Status AND/OR Age (>24)
**Rationale**: Educational outcome reporting may differ for veterans and non-traditional students, affecting completion rate accuracy.

#### ASSISTments (`assistments`)
**Biased Condition**: High Attempt Count (attempt_count > 0)
**Combo Conditions**: High Attempts AND/OR Hint Usage
**Rationale**: Student engagement patterns may be misinterpreted as performance indicators, potentially biasing educational assessments.

### Political/Survey Datasets

#### ANES Voting (`anes`)
**Biased Condition**: Missing Responses (VCF0302_no answer == 1)
**Combo Conditions**: Missing Responses AND/OR Multiple Missing Answers
**Rationale**: Non-response patterns in political surveys often correlate with demographic factors, introducing systematic bias in voting behavior data.

## Condition Types Explained

### Single (Biased) Conditions
Focus on one primary demographic or behavioral factor known to introduce systematic bias in each domain.

### Combo Conditions
- **AND conditions**: Intersectional bias affecting individuals with multiple risk factors
- **OR conditions**: Broader bias affecting individuals with any of several risk factors

## Research Foundation

These conditions are grounded in empirical research showing:

1. **Healthcare Disparities**: Documented differences in care quality and diagnostic accuracy across demographic groups
2. **Socioeconomic Bias**: Systematic differences in benefit access and economic opportunity reporting
3. **Educational Inequity**: Varying assessment and outcome measurement across student populations
4. **Financial Discrimination**: Historical and ongoing bias in credit and lending decisions
5. **Survey Non-Response**: Demographic patterns in survey participation and response accuracy

## Implementation Notes

- Conditions use actual feature values from each dataset
- Quantile-based thresholds (e.g., 70th percentile for age) ensure sufficient sample sizes
- Missing data indicators (-1.0, specific encoded values) reflect real data quality issues
- Print statements in functions show condition prevalence for validation

## Usage in Experiments

These conditions enable researchers to:
- Test model robustness to realistic bias patterns
- Evaluate fairness metrics under demographically-correlated noise
- Compare random vs. systematic noise effects
- Develop bias-aware data cleaning strategies

This systematic approach to noise injection provides a more realistic assessment of model behavior under the types of data quality issues encountered in real-world deployments.
