import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from scipy import stats
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings

def intersectional_fairness_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected_attributes: Dict[str, np.ndarray],
    skin_colors: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    include_infofair: bool = True,
    include_mantel_haenszel: bool = True
) -> Dict[str, Union[float, Dict, pd.DataFrame]]:
    """
    Evaluate intersectional fairness across multiple protected attributes.
    
    This implementation follows Buolamwini & Gebru (2018) approach for evaluating
    ML fairness across intersectional subgroups rather than single protected attributes.
    Includes Fitzpatrick skin type classification, differential fairness metrics,
    and modern extensions like InfoFair and generalized Mantel-Haenszel tests.
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1)
    y_pred : np.ndarray
        Predicted binary labels (0 or 1)
    protected_attributes : Dict[str, np.ndarray]
        Dictionary mapping attribute names to arrays of attribute values
        (e.g., {'gender': gender_array, 'age_group': age_array})
    skin_colors : np.ndarray, optional
        CIELAB color values for Fitzpatrick skin type classification
        Shape: (n_samples, 3) for L*, a*, b* values
    alpha : float, default=0.05
        Significance level for statistical tests
    include_infofair : bool, default=True
        Whether to compute information-theoretic fairness metrics
    include_mantel_haenszel : bool, default=True
        Whether to compute generalized Mantel-Haenszel test
        
    Returns
    -------
    Dict[str, Union[float, Dict, pd.DataFrame]]
        Dictionary containing:
        - 'subgroup_metrics': DataFrame with accuracy/error rates per subgroup
        - 'fairness_gaps': Dict with max/min accuracy differences
        - 'fitzpatrick_classification': Array of skin types (if skin_colors provided)
        - 'differential_fairness': Statistical test results
        - 'infofair_score': Information-theoretic fairness measure
        - 'mantel_haenszel': Generalized MH test results
        - 'intersectional_bias': Overall bias assessment
    """
    
    # Input validation
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")
    
    if not all(len(attr) == len(y_true) for attr in protected_attributes.values()):
        raise ValueError("All protected attributes must have same length as y_true")
    
    if not np.all(np.isin(y_true, [0, 1])) or not np.all(np.isin(y_pred, [0, 1])):
        raise ValueError("y_true and y_pred must contain only 0s and 1s")
    
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    
    results = {}
    
    # Fitzpatrick skin type classification using Individual Typology Angle (ITA)
    if skin_colors is not None:
        fitzpatrick_types = _classify_fitzpatrick_skin_type(skin_colors)
        protected_attributes['skin_type'] = fitzpatrick_types
        results['fitzpatrick_classification'] = fitzpatrick_types
    
    # Create intersectional subgroups
    subgroups_df = pd.DataFrame(protected_attributes)
    subgroups_df['y_true'] = y_true
    subgroups_df['y_pred'] = y_pred
    
    # Create intersectional group identifier
    group_cols = list(protected_attributes.keys())
    subgroups_df['intersectional_group'] = subgroups_df[group_cols].apply(
        lambda x: '_'.join(str(v) for v in x), axis=1
    )
    
    # Compute metrics for each intersectional subgroup
    subgroup_metrics = []
    
    for group_name, group_data in subgroups_df.groupby('intersectional_group'):
        if len(group_data) == 0:
            continue
            
        group_y_true = group_data['y_true'].values
        group_y_pred = group_data['y_pred'].values
        
        # Basic metrics
        accuracy = accuracy_score(group_y_true, group_y_pred)
        error_rate = 1 - accuracy
        
        # Confusion matrix metrics
        tn, fp, fn, tp = confusion_matrix(group_y_true, group_y_pred, labels=[0, 1]).ravel()
        
        # Fairness-specific metrics
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
        
        subgroup_metrics.append({
            'intersectional_group': group_name,
            'sample_size': len(group_data),
            'accuracy': accuracy,
            'error_rate': error_rate,
            'true_positive_rate': tpr,
            'false_positive_rate': fpr,
            'true_negative_rate': tnr,
            'false_negative_rate': fnr,
            **{col: group_data[col].iloc[0] for col in group_cols}
        })
    
    subgroup_metrics_df = pd.DataFrame(subgroup_metrics)
    results['subgroup_metrics'] = subgroup_metrics_df
    
    # Compute fairness gaps (max difference in performance across groups)
    if len(subgroup_metrics_df) > 1:
        accuracy_gap = subgroup_metrics_df['accuracy'].max() - subgroup_metrics_df['accuracy'].min()
        tpr_gap = subgroup_metrics_df['true_positive_rate'].max() - subgroup_metrics_df['true_positive_rate'].min()
        fpr_gap = subgroup_metrics_df['false_positive_rate'].max() - subgroup_metrics_df['false_positive_rate'].min()
        
        results['fairness_gaps'] = {
            'accuracy_gap': accuracy_gap,
            'tpr_gap': tpr_gap,
            'fpr_gap': fpr_gap,
            'max_accuracy_group': subgroup_metrics_df.loc[subgroup_metrics_df['accuracy'].idxmax(), 'intersectional_group'],
            'min_accuracy_group': subgroup_metrics_df.loc[subgroup_metrics_df['accuracy'].idxmin(), 'intersectional_group']
        }
    
    # Differential fairness test (comparing error rates across groups)
    if len(subgroup_metrics_df) > 1:
        differential_fairness_results = _compute_differential_fairness(subgroup_metrics_df, alpha)
        results['differential_fairness'] = differential_fairness_results
    
    # Information-theoretic fairness (InfoFair)
    if include_infofair and len(subgroup_metrics_df) > 1:
        infofair_score = _compute_infofair_score(y_true, y_pred, subgroups_df['intersectional_group'].values)
        results['infofair_score'] = infofair_score
    
    # Generalized Mantel-Haenszel test for intersectional bias
    if include_mantel_haenszel and len(group_cols) >= 1:
        mh_results = _compute_mantel_haenszel_test(subgroups_df, group_cols, alpha)
        results['mantel_haenszel'] = mh_results
    
    # Overall intersectional bias assessment
    if len(subgroup_metrics_df) > 1:
        bias_assessment = _assess_intersectional_bias(subgroup_metrics_df, alpha)
        results['intersectional_bias'] = bias_assessment
    
    return results


def _classify_fitzpatrick_skin_type(skin_colors: np.ndarray) -> np.ndarray:
    """
    Classify Fitzpatrick skin types using Individual Typology Angle (ITA).
    
    ITA = arctan((L* - 50) / b*) * 180 / π
    
    Fitzpatrick Scale:
    Type I-II: ITA > 55° (Very Light)
    Type III: 41° < ITA ≤ 55° (Light)  
    Type IV: 28° < ITA ≤ 41° (Medium)
    Type V: 10° < ITA ≤ 28° (Dark)
    Type VI: ITA ≤ 10° (Very Dark)
    """
    if skin_colors.shape[1] != 3:
        raise ValueError("skin_colors must have shape (n_samples, 3) for L*, a*, b* values")
    
    L_star = skin_colors[:, 0]
    b_star = skin_colors[:, 2]
    
    # Calculate Individual Typology Angle
    ita = np.arctan2(L_star - 50, b_star) * 180 / np.pi
    
    # Classify into Fitzpatrick types
    skin_types = np.zeros(len(ita), dtype=int)
    skin_types[ita > 55] = 1  # Type I-II (Very Light)
    skin_types[(ita > 41) & (ita <= 55)] = 2  # Type III (Light)
    skin_types[(ita > 28) & (ita <= 41)] = 3  # Type IV (Medium)
    skin_types[(ita > 10) & (ita <= 28)] = 4  # Type V (Dark)
    skin_types[ita <= 10] = 5  # Type VI (Very Dark)
    
    return skin_types


def _compute_differential_fairness(subgroup_metrics_df: pd.DataFrame, alpha: float) -> Dict:
    """
    Compute differential fairness using chi-square test for homogeneity of error rates.
    """
    # Prepare contingency table for chi-square test
    observed_correct = (subgroup_metrics_df['accuracy'] * subgroup_metrics_df['sample_size']).astype(int)
    observed_incorrect = subgroup_metrics_df['sample_size'] - observed_correct
    
    contingency_table = np.array([observed_correct, observed_incorrect])
    
    # Chi-square test for homogeneity
    chi2_stat, p_value = stats.chi2_contingency(contingency_table)[:2]
    
    # Effect size (Cramér's V)
    n = subgroup_metrics_df['sample_size'].sum()
    cramers_v = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))
    
    return {
        'chi2_statistic': chi2_stat,
        'p_value': p_value,
        'significant': p_value < alpha,
        'cramers_v': cramers_v,
        'degrees_of_freedom': len(subgroup_metrics_df) - 1
    }


def _compute_infofair_score(y_true: np.ndarray, y_pred: np.ndarray, groups: np