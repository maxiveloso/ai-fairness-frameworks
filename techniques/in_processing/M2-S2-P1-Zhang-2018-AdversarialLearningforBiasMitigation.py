import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, Tuple
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings

def adversarial_learning_bias_mitigation(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    protected_attribute: Union[np.ndarray, pd.Series],
    adversarial_loss_weight: float = 1.0,
    predictor_hidden_layers: Tuple[int, ...] = (100, 50),
    adversary_hidden_layers: Tuple[int, ...] = (50,),
    max_iter: int = 1000,
    learning_rate: float = 0.001,
    fairness_criterion: str = 'demographic_parity',
    test_size: float = 0.2,
    random_state: Optional[int] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Implement adversarial learning for bias mitigation in machine learning models.
    
    This technique uses an adversarial network architecture where a predictor network
    maximizes accuracy on the main task while an adversary network tries to predict
    the protected attribute from the predictor's hidden representations. The adversarial
    training encourages the predictor to learn representations that are less correlated
    with the protected attribute, thereby reducing bias.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix
    y : array-like of shape (n_samples,)
        Target variable (binary classification)
    protected_attribute : array-like of shape (n_samples,)
        Protected attribute (e.g., gender, race) - binary
    adversarial_loss_weight : float, default=1.0
        Weight for adversarial loss component in total loss
    predictor_hidden_layers : tuple of int, default=(100, 50)
        Hidden layer sizes for predictor network
    adversary_hidden_layers : tuple of int, default=(50,)
        Hidden layer sizes for adversary network
    max_iter : int, default=1000
        Maximum number of training iterations
    learning_rate : float, default=0.001
        Learning rate for optimization
    fairness_criterion : str, default='demographic_parity'
        Fairness criterion to evaluate ('demographic_parity', 'equality_of_odds')
    test_size : float, default=0.2
        Proportion of data to use for testing
    random_state : int, optional
        Random seed for reproducibility
    verbose : bool, default=False
        Whether to print training progress
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'predictor_accuracy': Accuracy of main prediction task
        - 'adversary_accuracy': Accuracy of adversary in predicting protected attribute
        - 'demographic_parity_difference': Difference in positive prediction rates
        - 'equality_of_odds_difference': Difference in TPR between groups
        - 'equality_of_opportunity_difference': Difference in TPR for positive class
        - 'predictor_model': Trained predictor model
        - 'fairness_metrics': Dictionary of fairness evaluation metrics
        - 'training_history': Training loss history if verbose=True
    """
    
    # Input validation
    X = np.asarray(X)
    y = np.asarray(y)
    protected_attribute = np.asarray(protected_attribute)
    
    if X.shape[0] != len(y) or X.shape[0] != len(protected_attribute):
        raise ValueError("X, y, and protected_attribute must have same number of samples")
    
    if len(np.unique(y)) != 2:
        raise ValueError("Target variable y must be binary")
    
    if len(np.unique(protected_attribute)) != 2:
        raise ValueError("Protected attribute must be binary")
    
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    
    if adversarial_loss_weight < 0:
        raise ValueError("adversarial_loss_weight must be non-negative")
    
    valid_criteria = ['demographic_parity', 'equality_of_odds', 'equality_of_opportunity']
    if fairness_criterion not in valid_criteria:
        raise ValueError(f"fairness_criterion must be one of {valid_criteria}")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test, z_train, z_test = train_test_split(
        X_scaled, y, protected_attribute, test_size=test_size, 
        random_state=random_state, stratify=y
    )
    
    # Custom adversarial classifier implementation
    class AdversarialClassifier(BaseEstimator, ClassifierMixin):
        def __init__(self, predictor_layers, adversary_layers, adv_weight, 
                     max_iter, learning_rate, random_state):
            self.predictor_layers = predictor_layers
            self.adversary_layers = adversary_layers
            self.adv_weight = adv_weight
            self.max_iter = max_iter
            self.learning_rate = learning_rate
            self.random_state = random_state
            self.training_history = []
            
        def fit(self, X, y, protected_attr):
            # Initialize predictor and adversary networks
            self.predictor = MLPClassifier(
                hidden_layer_sizes=self.predictor_layers,
                max_iter=1,
                learning_rate_init=self.learning_rate,
                random_state=self.random_state,
                warm_start=True
            )
            
            self.adversary = MLPClassifier(
                hidden_layer_sizes=self.adversary_layers,
                max_iter=1,
                learning_rate_init=self.learning_rate,
                random_state=self.random_state,
                warm_start=True
            )
            
            # Initial fit to initialize networks
            self.predictor.fit(X, y)
            
            # Adversarial training loop
            for iteration in range(self.max_iter):
                # Train predictor on main task
                self.predictor.fit(X, y)
                
                # Get predictor's hidden representation (approximated by decision function)
                try:
                    hidden_repr = self.predictor.decision_function(X).reshape(-1, 1)
                except:
                    # Fallback if decision_function not available
                    hidden_repr = self.predictor.predict_proba(X)
                
                # Train adversary to predict protected attribute from hidden representation
                self.adversary.fit(hidden_repr, protected_attr)
                
                # Calculate adversarial loss (simplified version)
                pred_acc = accuracy_score(y, self.predictor.predict(X))
                adv_acc = accuracy_score(protected_attr, self.adversary.predict(hidden_repr))
                
                # Store training history
                self.training_history.append({
                    'iteration': iteration,
                    'predictor_accuracy': pred_acc,
                    'adversary_accuracy': adv_acc
                })
                
                # Early stopping if adversary accuracy is close to random
                if adv_acc < 0.55:  # Close to random guessing
                    break
                    
            return self
        
        def predict(self, X):
            return self.predictor.predict(X)
        
        def predict_proba(self, X):
            return self.predictor.predict_proba(X)
    
    # Train adversarial model
    adv_model = AdversarialClassifier(
        predictor_layers=predictor_hidden_layers,
        adversary_layers=adversary_hidden_layers,
        adv_weight=adversarial_loss_weight,
        max_iter=max_iter,
        learning_rate=learning_rate,
        random_state=random_state
    )
    
    adv_model.fit(X_train, y_train, z_train)
    
    # Make predictions
    y_pred = adv_model.predict(X_test)
    
    # Calculate fairness metrics
    def calculate_fairness_metrics(y_true, y_pred, protected_attr):
        """Calculate various fairness metrics"""
        
        # Demographic parity: P(Y_hat=1|Z=0) - P(Y_hat=1|Z=1)
        group_0_mask = protected_attr == 0
        group_1_mask = protected_attr == 1
        
        if np.sum(group_0_mask) == 0 or np.sum(group_1_mask) == 0:
            warnings.warn("One of the protected attribute groups is empty")
            return {}
        
        pos_rate_0 = np.mean(y_pred[group_0_mask])
        pos_rate_1 = np.mean(y_pred[group_1_mask])
        demographic_parity_diff = abs(pos_rate_0 - pos_rate_1)
        
        # Equality of odds: difference in TPR and FPR between groups
        cm_0 = confusion_matrix(y_true[group_0_mask], y_pred[group_0_mask])
        cm_1 = confusion_matrix(y_true[group_1_mask], y_pred[group_1_mask])
        
        # Handle case where confusion matrix might be 1x1
        if cm_0.shape == (1, 1):
            tpr_0 = 1.0 if cm_0[0, 0] > 0 else 0.0
            fpr_0 = 0.0
        else:
            tpr_0 = cm_0[1, 1] / (cm_0[1, 1] + cm_0[1, 0]) if (cm_0[1, 1] + cm_0[1, 0]) > 0 else 0
            fpr_0 = cm_0[0, 1] / (cm_0[0, 1] + cm_0[0, 0]) if (cm_0[0, 1] + cm_0[0, 0]) > 0 else 0
        
        if cm_1.shape == (1, 1):
            tpr_1 = 1.0 if cm_1[0, 0] > 0 else 0.0
            fpr_1 = 0.0
        else:
            tpr_1 = cm_1[1, 1] / (cm_1[1, 1] + cm_1[1, 0]) if (cm_1[1, 1] + cm_1[1, 0]) > 0 else 0
            fpr_1 = cm_1[0, 1] / (cm_1[0, 1] + cm_1[0, 0]) if (cm_1[0, 1] + cm_1[0, 0]) > 0 else 0
        
        equality_of_odds_diff = abs(tpr_0 - tpr_1) + abs(fpr_0 - fpr_1)
        equality_of_opportunity_diff = abs(tpr_0 - tpr_1)
        
        return {
            'demographic_parity_difference': demographic_parity_diff,
            'equality_of_odds_difference': equality_of_odds_diff,
            'equality_of_opportunity_difference': equality_of_opportunity_diff,
            'group_0_positive_rate': pos_rate_0,
            'group_1_positive_rate': pos_rate_1,
            'group_0_tpr': tpr_0,
            'group_1_tpr': tpr_1,
            'group_0_fpr': fpr_0,
            'group_1_fpr': fpr_1
        }

# ============================================================================
# CLI WRAPPER (added for case study execution)
# ============================================================================

def main():
    import argparse
    import json
    import sys
    
    parser = argparse.ArgumentParser(
        description="Adversarial Learning for Bias Mitigation - Zhang et al. (2018)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--data", required=True, help="Path to CSV data file")
    parser.add_argument("--protected", required=True, help="Protected attribute column name")
    parser.add_argument("--outcome", required=True, help="Outcome column name")
    parser.add_argument("--lambda", dest="adversary_weight", type=float, default=0.5,
                        help="Adversary loss weight (default: 0.5)")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs (default: 100)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--output-model", default="debiased_model.pkl", help="Output model path")
    parser.add_argument("--output-preds", default="debiased_predictions.csv", help="Output predictions")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("--verbose", action="store_true", help="Print detailed output")
    
    args = parser.parse_args()
    
    data = pd.read_csv(args.data)
    
    # Run technique (adjust function name as needed)
    result = adversarial_debiasing(
        X=data,
        protected_attribute=args.protected,
        outcome=args.outcome,
        adversary_loss_weight=args.adversary_weight,
        epochs=args.epochs,
        learning_rate=args.lr
    )
    
    if args.json:
        output = {
            "technique": "Adversarial Learning for Bias Mitigation",
            "technique_id": 6,
            "citation": "Zhang et al. (2018)",
            "parameters": {
                "protected_attribute": args.protected,
                "adversary_loss_weight": args.adversary_weight,
                "epochs": args.epochs,
                "learning_rate": args.lr,
            },
            "results": {
                "demographic_parity_before": result.get('dp_before'),
                "demographic_parity_after": result.get('dp_after'),
                "accuracy_before": result.get('accuracy_before'),
                "accuracy_after": result.get('accuracy_after'),
            },
            "output_model": args.output_model,
            "output_predictions": args.output_preds
        }
        print(json.dumps(output, indent=2, default=str))
    elif args.verbose:
        print(f"\nAdversarial Debiasing Results:")
        print(f"  Epochs: {args.epochs}")
        print(f"  Adversary weight: {args.adversary_weight}")
        for k, v in result.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
