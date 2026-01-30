#!/usr/bin/env python3
"""
Bootstrap Confidence Intervals for Fairness Metrics

Non-parametric method for estimating confidence intervals of fairness metrics.

Citation: Efron, B. (1979). Bootstrap Methods: Another Look at the Jackknife.

Usage:
    python bootstrap_confidence_intervals.py \
        --data results.csv \
        --protected race \
        --compare \
        --baseline-col pred_before \
        --intervention-col pred_after \
        --metric demographic_parity \
        --json
"""

import numpy as np
import pandas as pd
import argparse
import json
from typing import Dict, Any, Optional


def demographic_parity(y_pred: np.ndarray, protected: np.ndarray) -> float:
    groups = np.unique(protected)
    if len(groups) < 2:
        return 1.0
    rates = [y_pred[protected == g].mean() for g in groups if (protected == g).sum() > 0]
    if len(rates) < 2 or min(rates) == 0 or max(rates) == 0:
        return 0.0
    return min(rates) / max(rates)


def equalized_odds(y_pred: np.ndarray, y_true: np.ndarray, protected: np.ndarray) -> float:
    groups = np.unique(protected)
    if len(groups) < 2:
        return 1.0
    tpr_rates, fpr_rates = [], []
    for g in groups:
        mask = protected == g
        pos_mask = mask & (y_true == 1)
        neg_mask = mask & (y_true == 0)
        if pos_mask.sum() > 0:
            tpr_rates.append(y_pred[pos_mask].mean())
        if neg_mask.sum() > 0:
            fpr_rates.append(y_pred[neg_mask].mean())
    tpr_parity = min(tpr_rates) / max(tpr_rates) if len(tpr_rates) >= 2 and max(tpr_rates) > 0 else 1.0
    fpr_parity = min(fpr_rates) / max(fpr_rates) if len(fpr_rates) >= 2 and max(fpr_rates) > 0 else 1.0
    return (tpr_parity + fpr_parity) / 2


def equal_opportunity(y_pred: np.ndarray, y_true: np.ndarray, protected: np.ndarray) -> float:
    groups = np.unique(protected)
    tpr_rates = []
    for g in groups:
        mask = (protected == g) & (y_true == 1)
        if mask.sum() > 0:
            tpr_rates.append(y_pred[mask].mean())
    if len(tpr_rates) < 2 or max(tpr_rates) == 0:
        return 1.0
    return min(tpr_rates) / max(tpr_rates)


METRICS = {
    'demographic_parity': ('simple', demographic_parity),
    'dp': ('simple', demographic_parity),
    'equalized_odds': ('with_label', equalized_odds),
    'eo': ('with_label', equalized_odds),
    'equal_opportunity': ('with_label', equal_opportunity),
    'eop': ('with_label', equal_opportunity),
}


def bootstrap_ci(
    y_pred: np.ndarray,
    protected: np.ndarray,
    y_true: Optional[np.ndarray] = None,
    metric: str = 'demographic_parity',
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """Compute bootstrap confidence intervals for a fairness metric."""
    if seed is not None:
        np.random.seed(seed)
    
    metric_type, metric_func = METRICS.get(metric.lower(), (None, None))
    if metric_func is None:
        raise ValueError(f"Unknown metric: {metric}")
    
    if metric_type == 'with_label' and y_true is None:
        raise ValueError(f"y_true required for {metric}")
    
    n_samples = len(y_pred)
    
    # Point estimate
    if metric_type == 'simple':
        point_estimate = metric_func(y_pred, protected)
    else:
        point_estimate = metric_func(y_pred, y_true, protected)
    
    # Bootstrap
    bootstrap_estimates = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        idx = np.random.choice(n_samples, size=n_samples, replace=True)
        if metric_type == 'simple':
            bootstrap_estimates[i] = metric_func(y_pred[idx], protected[idx])
        else:
            bootstrap_estimates[i] = metric_func(y_pred[idx], y_true[idx], protected[idx])
    
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_estimates, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_estimates, 100 * (1 - alpha / 2))
    
    return {
        'technique': 'Bootstrap Confidence Intervals for Fairness Metrics',
        'citation': 'Efron, B. (1979). Bootstrap Methods: Another Look at the Jackknife.',
        'metric': metric,
        'point_estimate': float(point_estimate),
        'confidence_level': confidence,
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'ci_width': float(ci_upper - ci_lower),
        'standard_error': float(bootstrap_estimates.std()),
        'n_bootstrap': n_bootstrap
    }


def compare_bootstrap_ci(
    baseline_pred: np.ndarray,
    intervention_pred: np.ndarray,
    protected: np.ndarray,
    y_true: Optional[np.ndarray] = None,
    metric: str = 'demographic_parity',
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """Compare baseline vs intervention with bootstrap CIs."""
    if seed is not None:
        np.random.seed(seed)
    
    metric_type, metric_func = METRICS.get(metric.lower(), (None, None))
    if metric_func is None:
        raise ValueError(f"Unknown metric: {metric}")
    
    n_samples = len(baseline_pred)
    
    # Point estimates
    if metric_type == 'simple':
        baseline_est = metric_func(baseline_pred, protected)
        intervention_est = metric_func(intervention_pred, protected)
    else:
        baseline_est = metric_func(baseline_pred, y_true, protected)
        intervention_est = metric_func(intervention_pred, y_true, protected)
    
    diff_est = intervention_est - baseline_est
    
    # Bootstrap
    baseline_boots = np.zeros(n_bootstrap)
    intervention_boots = np.zeros(n_bootstrap)
    diff_boots = np.zeros(n_bootstrap)
    
    for i in range(n_bootstrap):
        idx = np.random.choice(n_samples, size=n_samples, replace=True)
        if metric_type == 'simple':
            baseline_boots[i] = metric_func(baseline_pred[idx], protected[idx])
            intervention_boots[i] = metric_func(intervention_pred[idx], protected[idx])
        else:
            baseline_boots[i] = metric_func(baseline_pred[idx], y_true[idx], protected[idx])
            intervention_boots[i] = metric_func(intervention_pred[idx], y_true[idx], protected[idx])
        diff_boots[i] = intervention_boots[i] - baseline_boots[i]
    
    alpha = 1 - confidence
    
    return {
        'technique': 'Bootstrap Confidence Intervals for Fairness Metrics',
        'citation': 'Efron, B. (1979). Bootstrap Methods: Another Look at the Jackknife.',
        'metric': metric,
        'baseline': {
            'point_estimate': float(baseline_est),
            'ci_lower': float(np.percentile(baseline_boots, 100 * alpha / 2)),
            'ci_upper': float(np.percentile(baseline_boots, 100 * (1 - alpha / 2)))
        },
        'intervention': {
            'point_estimate': float(intervention_est),
            'ci_lower': float(np.percentile(intervention_boots, 100 * alpha / 2)),
            'ci_upper': float(np.percentile(intervention_boots, 100 * (1 - alpha / 2)))
        },
        'difference': {
            'point_estimate': float(diff_est),
            'ci_lower': float(np.percentile(diff_boots, 100 * alpha / 2)),
            'ci_upper': float(np.percentile(diff_boots, 100 * (1 - alpha / 2))),
            'significant': not (np.percentile(diff_boots, 100 * alpha / 2) <= 0 <= np.percentile(diff_boots, 100 * (1 - alpha / 2)))
        },
        'confidence_level': confidence,
        'n_bootstrap': n_bootstrap
    }


def main():
    parser = argparse.ArgumentParser(description='Bootstrap Confidence Intervals for Fairness Metrics')
    
    parser.add_argument('--data', required=True, help='CSV file')
    parser.add_argument('--protected', required=True, help='Protected attribute column')
    parser.add_argument('--prediction', help='Prediction column (single model)')
    parser.add_argument('--outcome', help='True outcome column')
    parser.add_argument('--compare', action='store_true', help='Compare baseline vs intervention')
    parser.add_argument('--baseline-col', default='pred_baseline')
    parser.add_argument('--intervention-col', default='pred_intervention')
    parser.add_argument('--metric', default='demographic_parity',
                        choices=['demographic_parity', 'dp', 'equalized_odds', 'eo', 'equal_opportunity', 'eop'])
    parser.add_argument('--n-bootstrap', type=int, default=10000)
    parser.add_argument('--confidence', type=float, default=0.95)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--output', help='Save results to file')
    
    args = parser.parse_args()
    
    df = pd.read_csv(args.data)
    protected = df[args.protected].values
    y_true = df[args.outcome].values if args.outcome else None
    
    if args.compare:
        results = compare_bootstrap_ci(
            df[args.baseline_col].values, df[args.intervention_col].values,
            protected, y_true, args.metric, args.n_bootstrap, args.confidence, args.seed
        )
    else:
        if not args.prediction:
            parser.error('--prediction required when not using --compare')
        results = bootstrap_ci(
            df[args.prediction].values, protected, y_true,
            args.metric, args.n_bootstrap, args.confidence, args.seed
        )
    
    if args.json:
        output = json.dumps(results, indent=2)
        print(output)
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output)
    else:
        print(f"\n{'='*50}")
        print("BOOTSTRAP CONFIDENCE INTERVALS")
        print(f"{'='*50}")
        print(f"Metric: {results['metric']}")
        print(f"Confidence: {results['confidence_level']*100:.0f}%")
        if args.compare:
            print(f"\nBaseline: {results['baseline']['point_estimate']:.4f} [{results['baseline']['ci_lower']:.4f}, {results['baseline']['ci_upper']:.4f}]")
            print(f"Intervention: {results['intervention']['point_estimate']:.4f} [{results['intervention']['ci_lower']:.4f}, {results['intervention']['ci_upper']:.4f}]")
            print(f"Difference: {results['difference']['point_estimate']:+.4f} [{results['difference']['ci_lower']:.4f}, {results['difference']['ci_upper']:.4f}]")
            print(f"Significant: {'YES' if results['difference']['significant'] else 'NO'}")
        else:
            print(f"Estimate: {results['point_estimate']:.4f}")
            print(f"CI: [{results['ci_lower']:.4f}, {results['ci_upper']:.4f}]")
        print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
