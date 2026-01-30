import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from typing import Dict, Any, Optional, Union
from scipy import stats

def difference_in_differences(
    data: pd.DataFrame,
    outcome_col: str,
    group_col: str,
    time_col: str,
    treatment_value: Union[str, int, float] = 1,
    post_value: Union[str, int, float] = 1,
    covariates: Optional[list] = None
) -> Dict[str, Any]:
    """
    Implement Difference-in-Differences (DiD) estimation following Card and Krueger (1994).
    
    The DiD method estimates causal effects by comparing changes in outcomes over time
    between treatment and control groups. The core assumption is parallel trends:
    treatment and control groups would have followed parallel paths in the absence
    of treatment.
    
    Core model: Y = β₀ + β₁*Treatment + β₂*Post + β₃*Treatment*Post + ε
    where β₃ is the DiD estimator (treatment effect).
    
    Parameters:
    -----------
    data : pd.DataFrame
        Panel dataset with observations for treatment/control groups before/after intervention
    outcome_col : str
        Name of the dependent variable column
    group_col : str
        Name of the treatment group indicator column
    time_col : str
        Name of the time period indicator column
    treatment_value : Union[str, int, float], default=1
        Value in group_col that indicates treatment group
    post_value : Union[str, int, float], default=1
        Value in time_col that indicates post-treatment period
    covariates : Optional[list], default=None
        List of additional control variables to include
    
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing:
        - 'did_coefficient': DiD treatment effect estimate
        - 'did_pvalue': p-value for DiD coefficient
        - 'did_ci_lower': Lower bound of 95% confidence interval
        - 'did_ci_upper': Upper bound of 95% confidence interval
        - 'group_effect': Main effect of treatment group
        - 'time_effect': Main effect of time period
        - 'r_squared': Model R-squared
        - 'n_observations': Number of observations
        - 'model_summary': Full regression results
        - 'means_table': 2x2 table of group means by period
    
    Raises:
    -------
    ValueError
        If required columns are missing or data validation fails
    """
    
    # Input validation
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data must be a pandas DataFrame")
    
    required_cols = [outcome_col, group_col, time_col]
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    if data[required_cols].isnull().any().any():
        raise ValueError("Missing values found in required columns")
    
    if len(data) < 4:
        raise ValueError("Insufficient observations for DiD estimation")
    
    # Create working copy of data
    df = data.copy()
    
    # Create binary treatment and post indicators
    df['treatment'] = (df[group_col] == treatment_value).astype(int)
    df['post'] = (df[time_col] == post_value).astype(int)
    df['treatment_post'] = df['treatment'] * df['post']  # Interaction term
    
    # Validate that we have all four groups (2x2 design)
    group_counts = df.groupby(['treatment', 'post']).size()
    if len(group_counts) != 4:
        raise ValueError("DiD requires observations for all four groups (treatment/control × pre/post)")
    
    # Build regression formula
    formula = f"{outcome_col} ~ treatment + post + treatment_post"
    if covariates:
        missing_covs = [cov for cov in covariates if cov not in df.columns]
        if missing_covs:
            raise ValueError(f"Missing covariate columns: {missing_covs}")
        formula += " + " + " + ".join(covariates)
    
    # Fit OLS regression
    try:
        model = smf.ols(formula, data=df).fit()
    except Exception as e:
        raise ValueError(f"Regression estimation failed: {str(e)}")
    
    # Extract DiD coefficient (interaction term)
    did_coef = model.params['treatment_post']
    did_pvalue = model.pvalues['treatment_post']
    did_ci = model.conf_int().loc['treatment_post']
    
    # Calculate means table for interpretation
    means_table = df.groupby(['treatment', 'post'])[outcome_col].mean().unstack()
    means_table.index = ['Control', 'Treatment']
    means_table.columns = ['Pre', 'Post']
    
    # Manual DiD calculation for verification
    control_pre = means_table.loc['Control', 'Pre']
    control_post = means_table.loc['Control', 'Post']
    treatment_pre = means_table.loc['Treatment', 'Pre']
    treatment_post = means_table.loc['Treatment', 'Post']
    
    manual_did = (treatment_post - treatment_pre) - (control_post - control_pre)
    
    return {
        'did_coefficient': float(did_coef),
        'did_pvalue': float(did_pvalue),
        'did_ci_lower': float(did_ci.iloc[0]),
        'did_ci_upper': float(did_ci.iloc[1]),
        'group_effect': float(model.params.get('treatment', np.nan)),
        'time_effect': float(model.params.get('post', np.nan)),
        'r_squared': float(model.rsquared),
        'adj_r_squared': float(model.rsquared_adj),
        'n_observations': int(model.nobs),
        'model_summary': model.summary(),
        'means_table': means_table,
        'manual_did_check': float(manual_did),
        'f_statistic': float(model.fvalue),
        'f_pvalue': float(model.f_pvalue)
    }

if __name__ == "__main__":
    # Example: Minimum wage impact on employment (Card & Krueger style)
    np.random.seed(42)
    
    # Generate synthetic data mimicking fast-food employment study
    n_per_group = 100
    
    # Control group (Pennsylvania) - no minimum wage increase
    control_pre = pd.DataFrame({
        'employment': np.random.normal(20, 3, n_per_group),  # Average employment
        'state': 'PA',
        'period': 'before',
        'store_size': np.random.normal(1000, 200, n_per_group)
    })
    
    control_post = pd.DataFrame({
        'employment': np.random.normal(20.5, 3, n_per_group),  # Slight natural increase
        'state': 'PA',
        'period': 'after',
        'store_size': np.random.normal(1000, 200, n_per_group)
    })
    
    # Treatment group (New Jersey) - minimum wage increase
    treatment_pre = pd.DataFrame({
        'employment': np.random.normal(21, 3, n_per_group),  # Slightly higher baseline
        'state': 'NJ',
        'period': 'before',
        'store_size': np.random.normal(1000, 200, n_per_group)
    })
    
    treatment_post = pd.DataFrame({
        'employment': np.random.normal(22.8, 3, n_per_group),  # Increase due to treatment + trend
        'state': 'NJ',
        'period': 'after',
        'store_size': np.random.normal(1000, 200, n_per_group)
    })
    
    # Combine data
    data = pd.concat([control_pre, control_post, treatment_pre, treatment_post], 
                     ignore_index=True)
    
    # Run DiD analysis
    results = difference_in_differences(
        data=data,
        outcome_col='employment',
        group_col='state',
        time_col='period',
        treatment_value='NJ',
        post_value='after',
        covariates=['store_size']
    )
    
    print("Difference-in-Differences Results")
    print("=" * 40)
    print(f"Treatment Effect (DiD): {results['did_coefficient']:.3f}")
    print(f"P-value: {results['did_pvalue']:.3f}")
    print(f"95% CI: [{results['did_ci_lower']:.3f}, {results['did_ci_upper']:.3f}]")
    print(f"R-squared: {results['r_squared']:.3f}")
    print(f"N observations: {results['n_observations']}")
    
    print("\nMeans Table:")
    print(results['means_table'].round(2))
    
    print(f"\nManual DiD calculation: {results['manual_did_check']:.3f}")
    
    # Interpretation
    if results['did_pvalue'] < 0.05:
        print(f"\nThe minimum wage increase had a statistically significant")
        print(f"effect of {results['did_coefficient']:.2f} on employment.")
    else:
        print(f"\nNo statistically significant effect found.")