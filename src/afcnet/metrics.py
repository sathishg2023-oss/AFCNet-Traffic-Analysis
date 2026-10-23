"""Regression metrics for frame-level vehicle counting."""
import numpy as np

def regression_metrics(targets, predictions):
    y = np.asarray(targets, dtype=np.float64).reshape(-1)
    p = np.asarray(predictions, dtype=np.float64).reshape(-1)
    if y.size == 0 or y.size != p.size: raise ValueError('Targets/predictions must be non-empty and equal length.')
    err = p - y
    return {'n': int(y.size), 'mae': float(np.mean(np.abs(err))),
            'mse': float(np.mean(err ** 2)), 'rmse': float(np.sqrt(np.mean(err ** 2)))}
