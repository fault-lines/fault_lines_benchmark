import numpy as np
from sklearn.metrics import accuracy_score

class FairRS:
    """
    A Fairlearn-compatible reweighing implementation with random search over a scaling parameter,
    replicating the 'RS Reweighing' method from the FairGBM paper.
    """

    def __init__(self, estimator, constraints, n_trials=1, weight_limit=2.0, alpha=0.75, random_seed=None):
        """
        Initialize the random search reweighing model.

        Parameters:
        - estimator: Base classifier (e.g., LGBMClassifier)
        - constraints: Fairlearn Moment object (e.g., EqualOpportunity)
        - n_trials: Number of random weight configurations to test
        - weight_limit: Maximum scaling factor for reweighing (0 to weight_limit)
        - alpha: Trade-off parameter for performance vs. fairness (alpha * perf + (1 - alpha) * fair)
        - random_seed: Seed for reproducibility
        """
        self.estimator = estimator
        self.constraints = constraints
        self.n_trials = n_trials
        self.weight_limit = weight_limit
        self.alpha = alpha  # Trade-off parameter from FairGBM
        self.random_seed = random_seed
        self.best_model = None
        self.best_weights = None
        self.best_score = -float('inf')  # Maximize trade-off score

    def _compute_reweighing_weights(self, X, y, sensitive_features):
        """
        Compute base reweighing weights to balance group outcomes, following Kamiran & Calders (2012).
        """
        # Load data into constraints object
        self.constraints.load_data(X, y, sensitive_features=sensitive_features)

        # Get group sizes and overall proportions
        n_samples = len(y)
        unique_groups = np.unique(sensitive_features)
        group_counts = {g: np.sum(sensitive_features == g) for g in unique_groups}
        overall_pos_rate = np.mean(y)  # P(Y=1)

        weights = np.ones(n_samples)

        # Basic reweighing: adjust weights to match overall positive rate per group
        # This is a simplified version, as FairRS might not use gamma iteratively
        for group in unique_groups:
            group_mask = (sensitive_features == group)
            group_y = y[group_mask]
            group_size = group_counts[group]
            group_pos_rate = np.mean(group_y)

            if group_pos_rate > 0 and overall_pos_rate > 0:  # Avoid division by zero
                weight_factor = overall_pos_rate / group_pos_rate
                weights[group_mask] *= weight_factor

        # Normalize weights to sum to n_samples
        weights = weights * (n_samples / np.sum(weights))
        weights = np.clip(weights, 0, None)  # Ensure non-negative
        return weights

    def fit(self, X, y, sensitive_features):
        """
        Fit the model using random search over reweighing scale factors, mimicking FairRS.
        """
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        # Compute base reweighing weights (static, not updated per trial)
        base_weights = self._compute_reweighing_weights(X, y, sensitive_features)

        # Randomly sample scaling factors (FairRS uses random search)
        scale_factors = np.random.uniform(0, self.weight_limit, self.n_trials)

        for scale in scale_factors:
            # Interpolate weights: scale=0 -> uniform, scale=weight_limit -> scaled reweighing
            sample_weights = (1 - scale) * np.ones(len(y)) + scale * base_weights

            # Train model with these weights
            model = self.estimator.__class__(**self.estimator.get_params())
            model.fit(X, y, sample_weight=sample_weights)

            # Evaluate: performance (accuracy) and fairness (constraint satisfaction)
            y_pred = model.predict(X)
            perf = accuracy_score(y, y_pred)  # Performance metric (Folktables uses accuracy)
            fairness_violation = np.max(np.abs(self.constraints.gamma(lambda x: y_pred)))  # Max violation
            fairness = 1 - fairness_violation  # Convert to [0, 1] scale (1 = perfect fairness)

            # Trade-off score as in FairGBM: alpha * perf + (1 - alpha) * fairness
            score = self.alpha * perf + (1 - self.alpha) * fairness

            if score > self.best_score:
                self.best_score = score
                self.best_model = model
                self.best_weights = sample_weights

        # Final fit with best weights
        self.best_model.fit(X, y, sample_weight=self.best_weights)
        return self

    def predict(self, X):
        """Predict using the best model."""
        return self.best_model.predict(X)

    def predict_proba(self, X):
        """Predict probabilities using the best model."""
        return self.best_model.predict_proba(X)