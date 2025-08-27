# Extending Fault Lines Benchmark for Multi-Class and Regression

## Multi-Class Classification Extension

### Noise Types
- **Random Noise**: For a label $y_i \in \{0, 1, ..., C-1\}$ with $C$ classes, flip to any other class with probability $p/(C-1)$ per class. For instance, if $y_i = 0$ and $C = 4$, the label could flip to 1, 2, or 3, each with probability $p/3$. This mimics random data entry errors across all classes.
- **Conditional Feature Noise**: When a feature condition $f_j = x$ is met, flip the label to one of the $C-1$ other classes with total probability $p$, distributed uniformly (e.g., $p/(C-1)$ per class). This simulates errors triggered by specific feature values, such as mislabeling based on a demographic attribute.
- **Correlated Feature Noise**: Flip the label when multiple conditions are met (e.g., $f_j = x$ and $f_k = y$), with $p$ split uniformly across $C-1$ classes. This captures complex interactions, like mislabeling due to combined socioeconomic and health factors.
- **Concatenated Feature Noise**: Apply flips when any of multiple conditions are satisfied (e.g., $f_j = x$ or $f_k = y$), distributing $p$ uniformly over $C-1$ classes. This models errors from overlapping risk factors, such as misclassification in educational outcomes.
- **Temporal/Contextual Noise**: Vary the flip probability $p(t_i)$ based on time or context (e.g., linearly from $p_{\min}$ to $p_{\max}$), with flips to any of the $C-1$ classes. For example, older records might have higher noise due to outdated labeling practices.
- **Class-Conditional Noise**: Assign a class-specific flip probability $p_c$ for each class $c$, where flips occur to other classes with probabilities summing to $p_c$. The distribution can be uniform or skewed (e.g., based on confusion matrix data), reflecting class-specific labeling biases.
- **Class-Transition Noise**: Introduce targeted flips between specific class pairs (e.g., 0 to 1, 1 to 2) with probability $p_{c,c'}$, derived from domain knowledge or real-world error patterns (e.g., confusion between "mild" and "moderate" disease states).
- **Implementation Details**: Update the noise injection pipeline to handle multi-class labels, ensuring $p$ is normalized across $C-1$ classes. Validate with synthetic datasets (e.g., MNIST with induced noise) to ensure realistic error distributions.

### Evaluation/Fairness Metrics
- **Performance Metrics**:
  - **Accuracy**: $\frac{1}{n} \sum_{i=1}^n \mathbb{I}(y_i = \hat{y}_i)$, the proportion of correct predictions.
  - **Macro-Averaged F1 Score**: Compute F1 for each class and average: $F1_{\text{macro}} = \frac{1}{C} \sum_{c=1}^C 2 \cdot \frac{\text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$, handling imbalance.
  - **Micro-Averaged F1 Score**: Aggregate true positives, false positives, and false negatives across all classes for a single F1 score.
  - **Cohen’s Kappa**: $\kappa = 1 - \frac{1 - p_o}{1 - p_e}$, where $p_o$ is observed agreement and $p_e$ is expected agreement, adjusted for chance.
- **Fairness Metrics**:
  - **Equal Opportunity**: Ensure TPR equality across groups $A = a$ and $A = b$ for each class $c$: $Pr(\hat{Y} = c | Y = c, A = a) = Pr(\hat{Y} = c | Y = c, A = b)$. This ensures equitable positive prediction rates.
  - **Demographic Parity**: Require $Pr(\hat{Y} = c | A = a) = Pr(\hat{Y} = c | A = b)$ for all $c$, promoting equal positive prediction probabilities across groups.
  - **Equalized Odds**: Balance TPR and FPR across groups for each class: $Pr(\hat{Y} = c | Y = c, A = a) = Pr(\hat{Y} = c | Y = c, A = b)$ and $Pr(\hat{Y} = c | Y \neq c, A = a) = Pr(\hat{Y} = c | Y \neq c, A = b)$.
  - **Predictive Parity**: Ensure precision equality: $Pr(Y = c | \hat{Y} = c, A = a) = Pr(Y = c | \hat{Y} = c, A = b)$ for each $c$, focusing on prediction reliability.
- **Implementation**: Compute metrics across all classes using the top five trials by macro F1 score, assessing trade-offs between performance and fairness. Extend the evaluation framework to handle multi-class test sets.

## Regression Extension

### Noise Types
- **Random Noise**: Add Gaussian noise $\mathcal{N}(0, \sigma^2)$ to labels, where $\sigma = p \cdot \text{std}(y)$ scales with the noise rate $p$. This simulates random measurement errors.
- **Conditional Feature Noise**: Apply $\mathcal{N}(0, \sigma^2)$ when $f_j = x$, with $\sigma = p \cdot \text{std}(y)$, modeling errors tied to specific conditions (e.g., income level).
- **Correlated Feature Noise**: Add $\mathcal{N}(0, \sigma^2)$ when $f_j = x$ and $f_k = y$, capturing joint feature effects on labeling errors.
- **Concatenated Feature Noise**: Introduce noise when $f_j = x$ or $f_k = y$, with $\sigma = p \cdot \text{std}(y)$, reflecting cumulative risk factors.
- **Temporal/Contextual Noise**: Vary $\sigma(t_i)$ over time or context (e.g., $\sigma(t_i) = \sigma_{\min} + (t_{\max} - t_i)/(t_{\max} - t_{\min}) \cdot (\sigma_{\max} - \sigma_{\min})$), simulating evolving data quality.
- **Class-Conditional Noise**: Partition labels into quantiles (e.g., tertiles), applying $\mathcal{N}(0, \sigma_c^2)$ based on quantile $c$, where $\sigma_c = p \cdot \text{std}(y_c)$.
- **Trend Noise**: Add a trend-based perturbation (e.g., $\epsilon_i = a \cdot y_i + b \cdot t_i + \mathcal{N}(0, \sigma^2)$), where $a$ and $b$ are coefficients, modeling systematic label drift.
- **Implementation Details**: Calibrate $\sigma$ to preserve label distribution realism, testing with synthetic regression datasets (e.g., Boston Housing with noise).

### Evaluation/Fairness Metrics
- **Performance Metrics**:
  - **Mean Squared Error (MSE)**: $\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$, measuring average squared error.
  - **Mean Absolute Error (MAE)**: $\frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$, robust to outliers.
  - **R² Score**: $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$, proportion of variance explained.
- **Fairness Metrics**:
  - **Equal Opportunity**: Ensure expected predictions for values above a threshold $\tau$ are equal across groups: $E[\hat{Y} | Y > \tau, A = a] = E[\hat{Y} | Y > \tau, A = b]$, where $\tau$ is domain-relevant (e.g., median).
  - **Demographic Parity**: Require $E[\hat{Y} | A = a] = E[\hat{Y} | A = b]$, ensuring no group bias in predicted means.
  - **Equalized Odds**: Balance error distributions: $E[(\hat{Y} - Y)^2 | A = a] = E[(\hat{Y} - Y)^2 | A = b]$, focusing on error equity.
  - **Predictive Parity**: Ensure conditional expectations match: $E[Y | \hat{Y} > \tau, A = a] = E[Y | \hat{Y} > \tau, A = b]$, emphasizing reliability for high predictions.
- **Implementation**: Use the top five trials by MSE to evaluate performance-fairness trade-offs, adapting in-distribution and out-of-distribution tests for continuous outcomes.

### Notes on Fairness in Regression
Fairness in regression is underexplored compared to classification, as research has focused on discrete outcomes (e.g., loan approval) in high-stakes domains. Regression's continuous nature requires new metrics, and the proposed threshold-based formulations (e.g., $E[\hat{Y} | Y > \tau, A = a] = E[\hat{Y} | Y > \tau, A = b]$) are tailored to the benchmark's diverse noise types (e.g., Conditional Feature Noise) and datasets (e.g., Hypertension). These non-standard adaptations address systematic biases in datasets like FICO HELOC, ensuring relevance to real-world applications.