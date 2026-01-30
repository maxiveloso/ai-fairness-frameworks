#!/usr/bin/env python3
"""
Permutation Test for Fairness Metrics

Non-parametric hypothesis test for determining statistical significance of 
fairness metric differences between baseline and intervention models.

Citation: Fisher, R.A. (1935). The Design of Experiments. Oliver & Boyd.

Usage:
    python permutation_test_fairness.py \
        --data results.csv \
        --protected race \
        --baseline-col pred_before \
        --intervention-col pred_after \
        --metric demographic_parity \
        --iterations 10000 \
        --json
"""

import numpy as np
import pandas as pd
import argparse
import json
from typing import Dict, Any, Optional


def demographic_parity(y_pred: np.ndarray, protected: np.ndarray) -> float:
    """Calculate demographic parity ratio: min(rate_g) / max(rate_g)"""
    groups = np.unique(protected)
    if len(groups) < 2:
        return 1.0
    rates = [y_pred[protected == g].mean() for g in groups if (protected == g).sum() > 0]
    if len(rates) < 2 or min(rates) == 0:
        return 0.0
    return min(rates) / max(rates)


def equalized_odds(y_pred: np.ndarray, y_true: np.ndarray, protected: np.ndarray) -> float:
    """Calculate equalized odds as average of TPR and FPR parity."""
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
    """Calculate equal opportunity (TPR parity)."""
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


def permutation_test(
    baseline_pred: np.ndarray,
    intervention_pred: np.ndarray,
    protected: np.ndarray,
    y_true: Optional[np.ndarray] = None,
    metric: str = 'demographic_parity',
    n_iterations: int = 10000,
    seed: Optional[int] = None,
    alternative: str = 'greater'
) -> Dict[str, Any]:
    """
    Perform permutation test for fairness metric improvement.
    
    Tests H0: intervention metric <= baseline metric
    vs H1: intervention metric > baseline metric
    """
    if seed is not None:
        np.random.seed(seed)
    
    metric_type, metric_func = METRICS.get(metric.lower(), (None, None))
    if metric_func is None:
        raise ValueError(f"Unknown metric: {metric}")
    
    # Calculate observed metrics
    if metric_type == 'with_label':
        if y_true is None:
            raise ValueError(f"y_true required for {metric}")
        baseline_metric = metric_func(baseline_pred, y_true, protected)
        intervention_metric = metric_func(intervention_pred, y_true, protected)
    else:
        baseline_metric = metric_func(baseline_pred, protected)
        intervention_metric = metric_func(intervention_pred, protected)
    
    observed_diff = intervention_metric - baseline_metric
    
    # Generate null distribution
    combined = np.concatenate([baseline_pred, intervention_pred])
    n = len(baseline_pred)
    null_distribution = np.zeros(n_iterations)
    
    for i in range(n_iterations):
        perm_idx = np.random.permutation(len(combined))
        perm_baseline = combined[perm_idx[:n]]
        perm_intervention = combined[perm_idx[n:]]
        
        if metric_type == 'with_label':
            perm_base_m = metric_func(perm_baseline, y_true, protected)
            perm_int_m = metric_func(perm_intervention, y_true, protected)
        else:
            perm_base_m = metric_func(perm_baseline, protected)
            perm_int_m = metric_func(perm_intervention, protected)
        
        null_distribution[i] = perm_int_m - perm_base_m
    
    # Calculate p-value
    if alternative == 'greater':
        p_value = (null_distribution >= observed_diff).mean()
    elif alternative == 'less':
        p_value = (null_distribution <= observed_diff).mean()
    else:
        p_value = (np.abs(null_distribution) >= np.abs(observed_diff)).mean()
    
    return {
        'technique': 'Permutation Test for Fairness Metrics',
        'citation': 'Fisher, R.A. (1935). The Design of Experiments.',
        'metric': metric,
        'baseline_value': float(baseline_metric),
        'intervention_value': float(intervention_metric),
        'observed_difference': float(observed_diff),
        'p_value': float(p_value),
        'n_iterations': n_iterations,
        'alternative': alternative,
        'significant_at_0.05': p_value < 0.05,
        'significant_at_0.01': p_value < 0.01,
        'significant_at_0.001': p_value < 0.001,
        'null_mean': float(null_distribution.mean()),
        'null_std': float(null_distribution.std()),
        'effect_size': float(observed_diff / null_distribution.std()) if null_distribution.std() > 0 else float('inf')
    }


def main():
    parser = argparse.ArgumentParser(description='Permutation Test for Fairness Metrics')
    
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument('--data', help='Single CSV with baseline and intervention columns')
    data_group.add_argument('--baseline', help='CSV with baseline predictions')
    
    parser.add_argument('--intervention', help='CSV with intervention predictions (with --baseline)')
    parser.add_argument('--protected', required=True, help='Protected attribute column')
    parser.add_argument('--outcome', help='True outcome column (for EO metrics)')
    parser.add_argument('--baseline-col', default='pred_baseline', help='Baseline column name')
    parser.add_argument('--intervention-col', default='pred_intervention', help='Intervention column name')
    parser.add_argument('--pred-col', default='prediction', help='Prediction column (--baseline mode)')
    parser.add_argument('--metric', default='demographic_parity', 
                        choices=['demographic_parity', 'dp', 'equalized_odds', 'eo', 'equal_opportunity', 'eop'])
    parser.add_argument('--iterations', type=int, default=10000)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--alternative', default='greater', choices=['greater', 'less', 'two-sided'])
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--output', help='Save results to file')
    
    args = parser.parse_args()
    
    if args.data:
        df = pd.read_csv(args.data)
        baseline_pred = df[args.baseline_col].values
        intervention_pred = df[args.intervention_col].values
        protected = df[args.protected].values
        y_true = df[args.outcome].values if args.outcome else None
    else:
        if not args.intervention:
            parser.error('--intervention required with --baseline')
        df_b = pd.read_csv(args.baseline)
        df_i = pd.read_csv(args.intervention)
        baseline_pred = df_b[args.pred_col].values
        intervention_pred = df_i[args.pred_col].values
        protected = df_b[args.protected].values
        y_true = df_b[args.outcome].values if args.outcome else None
    
    results = permutation_test(
        baseline_pred, intervention_pred, protected, y_true,
        args.metric, args.iterations, args.seed, args.alternative
    )
    
    if args.json:
        output = json.dumps(results, indent=2)
        print(output)
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output)
    else:
        print(f"\n{'='*50}")
        print("PERMUTATION TEST RESULTS")
        print(f"{'='*50}")
        print(f"Metric: {results['metric']}")
        print(f"Baseline: {results['baseline_value']:.4f}")
        print(f"Intervention: {results['intervention_value']:.4f}")
        print(f"Difference: {results['observed_difference']:+.4f}")
        print(f"\np-value: {results['p_value']:.6f}")
        print(f"Significant at α=0.001: {'YES' if results['significant_at_0.001'] else 'NO'}")
        print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
