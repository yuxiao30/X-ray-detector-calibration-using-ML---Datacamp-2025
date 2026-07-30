
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import skew, kurtosis
from sklearn.ensemble import HistGradientBoostingRegressor


class Model:
    def __init__(self):
        """Initialize an ensemble of residual regressors."""
        self.models = [
            HistGradientBoostingRegressor(
                loss="squared_error",
                max_iter=500,
                max_depth=5,
                l2_regularization=1.0,
                random_state=42,
            ),
            HistGradientBoostingRegressor(
                loss="absolute_error",
                max_iter=400,
                max_depth=4,
                learning_rate=0.07,
                random_state=2026,
            ),
            HistGradientBoostingRegressor(
                loss="squared_error",
                max_iter=450,
                max_depth=6,
                learning_rate=0.03,
                l2_regularization=0.5,
                random_state=123,
            ),
        ]

    def _find_threshold_crossings(self, X_smooth, peak_values, peak_indices, levels):
        """
        For each relative level, find the first crossing position after the peak.
        """
        n_samples, n_points = X_smooth.shape
        positions = []

        col_idx = np.arange(n_points)[None, :]
        right_of_peak = col_idx > peak_indices[:, None]

        for level in levels:
            threshold = peak_values * level
            crossed = (X_smooth < threshold[:, None]) & right_of_peak

            first_idx = np.argmax(crossed, axis=1)
            has_crossing = np.any(crossed, axis=1)
            first_idx[~has_crossing] = n_points - 1

            positions.append(first_idx.astype(float) * 2.0)

        return positions

    def _extract_features(self, X):
        """
        Convert each signal into a compact set of robust geometric descriptors.
        """
        X = np.asarray(X, dtype=float)
        X_smooth = gaussian_filter1d(X, sigma=3.5, axis=1)

        n_samples, n_points = X_smooth.shape
        row_idx = np.arange(n_samples)

        peak_values = X_smooth.max(axis=1)
        peak_indices = X_smooth.argmax(axis=1)

        threshold_levels = [0.01, 0.05, 0.15, 0.30, 0.50]
        crossing_features = self._find_threshold_crossings(
            X_smooth,
            peak_values,
            peak_indices,
            threshold_levels,
        )

        grad_1 = np.gradient(X_smooth, axis=1)
        grad_1_scaled = grad_1 / (peak_values[:, None] + 1e-8)

        reference_idx = (crossing_features[1] / 2.0).astype(int).clip(0, n_points - 1)

        local_gradient_features = []
        for shift in (-8, -4, 0, 4, 8):
            sample_idx = (reference_idx + shift).clip(0, n_points - 1)
            local_gradient_features.append(grad_1_scaled[row_idx, sample_idx])

        grad_2 = np.gradient(grad_1, axis=1)
        grad_2_scaled = grad_2 / (peak_values[:, None] + 1e-8)
        curvature_at_ref = grad_2_scaled[row_idx, reference_idx]

        X_scaled = X / (peak_values[:, None] + 1e-8)
        normalized_energy = np.sum(X_scaled ** 2, axis=1)

        feature_blocks = []
        feature_blocks.extend(crossing_features)
        feature_blocks.extend(local_gradient_features)

        feature_blocks.append(peak_values)
        feature_blocks.append(peak_indices.astype(float) * 2.0)

        feature_blocks.append(skew(X, axis=1))
        feature_blocks.append(kurtosis(X, axis=1))

        # Span between 50% and 1% thresholds
        feature_blocks.append(crossing_features[4] - crossing_features[0])

        # Second derivative at reference point
        feature_blocks.append(curvature_at_ref)

        # Width between 30% and 5% thresholds
        feature_blocks.append(crossing_features[3] - crossing_features[1])

        # Energy and peak/median contrast
        feature_blocks.append(normalized_energy)
        feature_blocks.append(peak_values / (np.median(X, axis=1) + 1e-8))

        return np.column_stack(feature_blocks)

    def fit(self, X, y, X_adapt=None):
        """
        Train all regressors to predict the residual with respect to the 5% crossing.
        """
        y = np.asarray(y).ravel().astype(float)

        features = self._extract_features(X)
        baseline = features[:, 1]   # 5% threshold location
        residual = y - baseline

        for reg in self.models:
            reg.fit(features, residual)

    def predict(self, X):
        """
        Predict by adding the ensemble residual correction to the baseline estimate.
        """
        features = self._extract_features(X)
        baseline = features[:, 1]

        residual_predictions = []
        for reg in self.models:
            residual_predictions.append(reg.predict(features).ravel())

        residual_predictions = np.column_stack(residual_predictions)
        correction = np.median(residual_predictions, axis=1)

        return baseline + correction
