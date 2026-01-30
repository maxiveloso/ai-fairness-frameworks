import numpy as np
import pandas as pd
from typing import Union, Dict, Optional, Tuple
from scipy import stats
import warnings

def e_value_approach(
    effect_estimate: float,
    confidence_interval: Optional[Tuple[float, float]] = None,
    effect_type: str = "risk_ratio",
    alpha: float = 0.05
) -> Dict[str, Union[float, str, Dict]]:
    """
    Calculate E-values for sensitivity analysis of unmeasured confounding.
    
    The E-value quantifies the minimum strength of association on the risk ratio scale
    that an unmeasured confounder would need to have with both the treatment and outcome
    to fully explain away a specific treatment-outcome association.
    
    Parameters
    ----------
    effect_estimate : float
        The observed effect estimate (risk ratio, odds ratio, hazard ratio, or 
        standardized mean difference)
    confidence_interval : tuple of float, optional
        Lower and upper bounds of the confidence interval for the effect estimate
    effect_type : str, default "risk_ratio"
        Type of effect measure. Options: "risk_ratio", "odds_ratio", "hazard_ratio", "smd"
    alpha : float, default 0.05
        Significance level for confidence interval (if not provided)
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'e_value_point': E-value for the point estimate
        - 'e_value_ci': E-value for the confidence interval bound closest to null
        - 'interpretation': Text interpretation of results
        - 'effect_estimate': Original effect estimate
        - 'effect_type': Type of effect measure
        - 'sensitivity_parameters': Dictionary with RRUD and RREU values
        
    Raises
    ------
    ValueError
        If effect_estimate is invalid or effect_type is not supported
    """
    
    # Input validation
    if not isinstance(effect_estimate, (int, float)) or np.isnan(effect_estimate):
        raise ValueError("effect_estimate must be a valid number")
    
    if effect_type not in ["risk_ratio", "odds_ratio", "hazard_ratio", "smd"]:
        raise ValueError("effect_type must be one of: 'risk_ratio', 'odds_ratio', 'hazard_ratio', 'smd'")
    
    if confidence_interval is not None:
        if len(confidence_interval) != 2:
            raise ValueError("confidence_interval must be a tuple of length 2")
        if confidence_interval[0] >= confidence_interval[1]:
            raise ValueError("Lower CI bound must be less than upper CI bound")
    
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    
    def _calculate_e_value_rr(rr: float) -> float:
        """
        Calculate E-value for risk ratio scale.
        
        Formula: E-value = RR + sqrt(RR × (RR - 1)) for RR > 1
        For RR < 1, calculate E-value for 1/RR first, then interpret
        """
        if rr <= 0:
            return np.nan
        
        if rr == 1:
            return 1.0
        
        # For RR < 1, work with the reciprocal
        if rr < 1:
            rr = 1 / rr
        
        # Apply E-value formula
        e_value = rr + np.sqrt(rr * (rr - 1))
        return e_value
    
    def _convert_to_rr_scale(estimate: float, measure_type: str) -> float:
        """Convert different effect measures to risk ratio scale for E-value calculation."""
        
        if measure_type == "risk_ratio" or measure_type == "hazard_ratio":
            return estimate
        
        elif measure_type == "odds_ratio":
            # For rare outcomes, OR approximates RR
            # For common outcomes, this is an approximation
            if estimate <= 0:
                return np.nan
            return estimate
        
        elif measure_type == "smd":
            # Convert standardized mean difference to risk ratio scale
            # Using approximation: RR ≈ exp(0.91 × SMD) for SMD > 0
            if estimate == 0:
                return 1.0
            elif estimate > 0:
                return np.exp(0.91 * estimate)
            else:
                return np.exp(0.91 * estimate)
        
        return estimate
    
    # Convert effect estimate to risk ratio scale
    rr_estimate = _convert_to_rr_scale(effect_estimate, effect_type)
    
    # Calculate E-value for point estimate
    e_value_point = _calculate_e_value_rr(rr_estimate)
    
    # Calculate E-value for confidence interval if provided
    e_value_ci = None
    ci_bound_used = None
    
    if confidence_interval is not None:
        # Convert CI bounds to RR scale
        ci_lower_rr = _convert_to_rr_scale(confidence_interval[0], effect_type)
        ci_upper_rr = _convert_to_rr_scale(confidence_interval[1], effect_type)
        
        # Use the bound closest to the null (1.0)
        if abs(ci_lower_rr - 1) < abs(ci_upper_rr - 1):
            e_value_ci = _calculate_e_value_rr(ci_lower_rr)
            ci_bound_used = "lower"
        else:
            e_value_ci = _calculate_e_value_rr(ci_upper_rr)
            ci_bound_used = "upper"
    
    # Calculate sensitivity parameters (RRUD and RREU)
    # These represent the strength of association needed between
    # unmeasured confounder and treatment (RRUD) and outcome (RREU)
    rrud = rreu = e_value_point if not np.isnan(e_value_point) else 1.0
    
    # Generate interpretation
    interpretation = _generate_interpretation(
        e_value_point, e_value_ci, effect_estimate, effect_type
    )
    
    results = {
        'e_value_point': e_value_point,
        'e_value_ci': e_value_ci,
        'ci_bound_used': ci_bound_used,
        'interpretation': interpretation,
        'effect_estimate': effect_estimate,
        'effect_type': effect_type,
        'sensitivity_parameters': {
            'RRUD': rrud,  # Risk ratio for unmeasured confounder-treatment association
            'RREU': rreu   # Risk ratio for unmeasured confounder-outcome association
        },
        'rr_scale_estimate': rr_estimate
    }
    
    return results

def _generate_interpretation(
    e_value_point: float, 
    e_value_ci: Optional[float], 
    effect_estimate: float,
    effect_type: str
) -> str:
    """Generate text interpretation of E-value results."""
    
    if np.isnan(e_value_point):
        return "E-value could not be calculated due to invalid effect estimate."
    
    interpretation = f"To explain away the observed {effect_type} of {effect_estimate:.3f}, "
    interpretation += f"an unmeasured confounder would need to be associated with both "
    interpretation += f"the treatment and outcome by risk ratios of {e_value_point:.2f}-fold each, "
    interpretation += f"above and beyond the measured confounders."
    
    if e_value_ci is not None:
        interpretation += f"\n\nTo shift the confidence interval to include the null, "
        interpretation += f"an unmeasured confounder would need associations of "
        interpretation += f"{e_value_ci:.2f}-fold each."
    
    # Add contextual guidance
    if e_value_point < 1.5:
        interpretation += f"\n\nInterpretation: The required confounder strength is relatively weak, "
        interpretation += f"suggesting the observed association could be easily explained by unmeasured confounding."
    elif e_value_point < 3:
        interpretation += f"\n\nInterpretation: The required confounder strength is moderate. "
        interpretation += f"Consider whether such confounding is plausible in your study context."
    else:
        interpretation += f"\n\nInterpretation: The required confounder strength is large, "
        interpretation += f"suggesting the observed association is relatively robust to unmeasured confounding."
    
    return interpretation

if __name__ == "__main__":
    # Example 1: Risk ratio with confidence interval
    print("Example 1: Risk Ratio Analysis")
    print("=" * 50)
    
    result1 = e_value_approach(
        effect_estimate=2.5,
        confidence_interval=(1.8, 3.5),
        effect_type="risk_ratio"
    )
    
    print(f"Effect estimate (RR): {result1['effect_estimate']}")
    print(f"E-value for point estimate: {result1['e_value_point']:.2f}")
    print(f"E-value for CI: {result1['e_value_ci']:.2f}")
    print(f"Sensitivity parameters - RRUD: {result1['sensitivity_parameters']['RRUD']:.2f}")
    print(f"Sensitivity parameters - RREU: {result1['sensitivity_parameters']['RREU']:.2f}")
    print(f"\n{result1['interpretation']}")
    
    print("\n" + "=" * 50)
    
    # Example 2: Odds ratio
    print("Example 2: Odds Ratio Analysis")
    print("=" * 50)
    
    result2 = e_value_approach(
        effect_estimate=1.8,
        confidence_interval=(1.2, 2.7),
        effect_type="odds_ratio"
    )
    
    print(f"Effect estimate (OR): {result2['effect_estimate']}")
    print(f"E-value for point estimate: {result2['e_value_point']:.2f}")
    print(f"E-value for CI: {result2['e_value_ci']:.2f}")
    print(f"\n{result2['interpretation']}")
    
    print("\n" + "=" * 50)
    
    # Example 3: Standardized mean difference
    print("Example 3: Standardized Mean Difference")
    print("=" * 50)
    
    result3 = e_value_approach(
        effect_estimate=0.5,
        confidence_interval=(0.2, 0.8),
        effect_type="smd"
    )
    
    print(f"Effect estimate (SMD): {result3['effect_estimate']}")
    print(f"E-value for point estimate: {result3['e_value_point']:.2f}")
    print(f"E-value for CI: {result3['e_value_ci']:.2f}")
    print(f"\n{result3['interpretation']}")
    
    print("\n" + "=" * 50)
    
    # Example 4: Effect estimate less than 1 (protective effect)
    print("Example 4: Protective Effect (RR < 1)")
    print("=" * 50)
    
    result4 = e_value_approach(
        effect_estimate=0.6,
        confidence_interval=(0.4, 0.9),
        effect_type="risk_ratio"
    )
    
    print(f"Effect estimate (RR): {result4['effect_estimate']}")
    print(f"E-value for point estimate: {result4['e_value_point']:.2f}")
    print(f"E-value for CI: {result4['e_value_ci']:.2f}")
    print(f"\n{result4['interpretation']}")