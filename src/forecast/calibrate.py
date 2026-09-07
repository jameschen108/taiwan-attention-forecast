"""Probability calibration for classification models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def fit_sigmoid_calibrator(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> LogisticRegression:
    """Platt scaling on validation probabilities."""
    cal = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
    cal.fit(y_prob.reshape(-1, 1), y_true)
    return cal


def apply_calibrator(calibrator: LogisticRegression, y_prob: np.ndarray) -> np.ndarray:
    return calibrator.predict_proba(y_prob.reshape(-1, 1))[:, 1]


def fit_calibrator_from_probs(y_true: np.ndarray, y_prob: np.ndarray) -> LogisticRegression | None:
    """Fit Platt scaler when enough samples exist."""
    mask = np.isfinite(y_true) & np.isfinite(y_prob)
    y_true, y_prob = y_true[mask], y_prob[mask]
    if len(y_true) < 100 or len(np.unique(y_true)) < 2:
        return None
    return fit_sigmoid_calibrator(y_true, y_prob)


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob - y_true) ** 2))


def reliability_bins(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    return pd.DataFrame({"mean_predicted": prob_pred, "fraction_positive": prob_true})


def paired_brier_delta(
    y_true: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
) -> dict:
    return {
        "brier_a": brier_score(y_true, prob_a),
        "brier_b": brier_score(y_true, prob_b),
        "delta_b_minus_a": brier_score(y_true, prob_a) - brier_score(y_true, prob_b),
    }
