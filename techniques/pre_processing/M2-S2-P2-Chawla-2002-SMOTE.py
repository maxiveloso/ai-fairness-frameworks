import numpy as np
import pandas as pd
from typing import Tuple, Union, Dict, Any
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_array, check_X_y
from sklearn.preprocessing import LabelEncoder
import warnings

def smote(X: Union[np.ndarray, pd.DataFrame], 
          y: Union[np.ndarray, pd.Series],
          k_neighbors: int = 5,
          sampling_strategy: Union[str, float, Dict] = 'auto',
          random_state: int = None) -> Dict[str, Any]:
    """
    Synthetic Minority Oversampling Technique (SMOTE) for handling imbalanced datasets.
    
    SMOTE generates synthetic samples for minority classes by creating new instances
    along line segments joining k minority class nearest neighbors. For each minority
    sample, k nearest neighbors are found, and synthetic samples are generated using
    linear interpolation: Synthetic = Original + λ × (Neighbor - Original) where λ ∈ [0,1].
    
    This technique helps balance datasets by oversampling minority classes rather than
    undersampling majority classes, preserving important information while reducing
    class imbalance bias in machine learning models.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input feature matrix
    y : array-like of shape (n_samples,)
        Target class labels
    k_neighbors : int, default=5
        Number of nearest neighbors to consider for synthetic sample generation
    sampling_strategy : str, float, or dict, default='auto'
        Sampling strategy:
        - 'auto': resample all classes but majority class
        - float: ratio of number of samples in minority class over majority class
        - dict: keys are classes, values are desired number of samples
    random_state : int, default=None
        Random seed for reproducibility
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'X_resampled': Resampled feature matrix
        - 'y_resampled': Resampled target labels
        - 'original_distribution': Original class distribution
        - 'resampled_distribution': New class distribution
        - 'synthetic_samples_per_class': Number of synthetic samples generated per class
        - 'sampling_strategy_used': Actual sampling strategy applied
        
    Raises
    ------
    ValueError
        If k_neighbors >= number of minority samples or invalid parameters
    """
    
    # Input validation
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values
        
    X, y = check_X_y(X, y, accept_sparse=False)
    
    if k_neighbors <= 0:
        raise ValueError("k_neighbors must be positive")
        
    if random_state is not None:
        np.random.seed(random_state)
    
    # Encode labels to ensure they are numeric
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    classes = np.unique(y_encoded)
    
    # Calculate original class distribution
    unique_classes, class_counts = np.unique(y_encoded, return_counts=True)
    original_distribution = dict(zip(label_encoder.inverse_transform(unique_classes), class_counts))
    
    # Determine sampling strategy
    majority_class = unique_classes[np.argmax(class_counts)]
    majority_count = np.max(class_counts)
    
    if sampling_strategy == 'auto':
        # Oversample all minority classes to match majority class
        target_counts = {cls: majority_count for cls in unique_classes if cls != majority_class}
    elif isinstance(sampling_strategy, float):
        # Use ratio to determine target counts
        target_count = int(majority_count * sampling_strategy)
        target_counts = {cls: target_count for cls in unique_classes if cls != majority_class}
    elif isinstance(sampling_strategy, dict):
        # Convert string keys to encoded values if necessary
        target_counts = {}
        for key, value in sampling_strategy.items():
            if isinstance(key, str):
                encoded_key = label_encoder.transform([key])[0]
            else:
                encoded_key = key
            target_counts[encoded_key] = value
    else:
        raise ValueError("Invalid sampling_strategy")
    
    # Initialize arrays for resampled data
    X_resampled = X.copy()
    y_resampled = y_encoded.copy()
    synthetic_samples_per_class = {}
    
    # Generate synthetic samples for each minority class
    for target_class in target_counts:
        class_indices = np.where(y_encoded == target_class)[0]
        current_count = len(class_indices)
        target_count = target_counts[target_class]
        
        if target_count <= current_count:
            synthetic_samples_per_class[label_encoder.inverse_transform([target_class])[0]] = 0
            continue
            
        n_synthetic = target_count - current_count
        
        if current_count < k_neighbors:
            warnings.warn(f"Class {label_encoder.inverse_transform([target_class])[0]} has fewer samples "
                         f"({current_count}) than k_neighbors ({k_neighbors}). "
                         f"Using k_neighbors={current_count-1}")
            k_neighbors_adjusted = max(1, current_count - 1)
        else:
            k_neighbors_adjusted = k_neighbors
        
        # Get samples for current minority class
        X_class = X[class_indices]
        
        # Fit k-NN model on minority class samples
        nn_model = NearestNeighbors(n_neighbors=k_neighbors_adjusted + 1, 
                                   algorithm='auto').fit(X_class)
        
        # Generate synthetic samples
        synthetic_samples = []
        
        for _ in range(n_synthetic):
            # Randomly select a minority class sample
            sample_idx = np.random.randint(0, len(X_class))
            sample = X_class[sample_idx]
            
            # Find k nearest neighbors (excluding the sample itself)
            distances, indices = nn_model.kneighbors([sample])
            neighbor_indices = indices[0][1:]  # Exclude the sample itself
            
            # Randomly select one neighbor
            neighbor_idx = np.random.choice(neighbor_indices)
            neighbor = X_class[neighbor_idx]
            
            # Generate synthetic sample using linear interpolation
            # λ is random value between 0 and 1
            lambda_val = np.random.random()
            synthetic_sample = sample + lambda_val * (neighbor - sample)
            synthetic_samples.append(synthetic_sample)
        
        # Add synthetic samples to resampled data
        if synthetic_samples:
            synthetic_samples = np.array(synthetic_samples)
            X_resampled = np.vstack([X_resampled, synthetic_samples])
            y_resampled = np.hstack([y_resampled, 
                                   np.full(len(synthetic_samples), target_class)])
        
        synthetic_samples_per_class[label_encoder.inverse_transform([target_class])[0]] = n_synthetic
    
    # Calculate resampled distribution
    unique_resampled, counts_resampled = np.unique(y_resampled, return_counts=True)
    resampled_distribution = dict(zip(label_encoder.inverse_transform(unique_resampled), 
                                    counts_resampled))
    
    # Convert y_resampled back to original labels
    y_resampled_original = label_encoder.inverse_transform(y_resampled)
    
    return {
        'X_resampled': X_resampled,
        'y_resampled': y_resampled_original,
        'original_distribution': original_distribution,
        'resampled_distribution': resampled_distribution,
        'synthetic_samples_per_class': synthetic_samples_per_class,
        'sampling_strategy_used': {label_encoder.inverse_transform([k])[0]: v 
                                 for k, v in target_counts.items()},
        'original_shape': X.shape,
        'resampled_shape': X_resampled.shape
    }

if __name__ == "__main__":
    # Example usage with imbalanced dataset
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
    
    # Create imbalanced dataset
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=3,
                             n_informative=8, n_redundant=2, n_clusters_per_class=1,
                             weights=[0.7, 0.2, 0.1], random_state=42)
    
    print("SMOTE Example")
    print("=" * 50)
    
    # Apply SMOTE
    result = smote(X, y, k_neighbors=5, sampling_strategy='auto', random_state=42)
    
    print("Original class distribution:")
    for class_label, count in result['original_distribution'].items():
        print(f"  Class {class_label}: {count} samples")
    
    print(f"\nOriginal dataset shape: {result['original_shape']}")
    print(f"Resampled dataset shape: {result['resampled_shape']}")
    
    print("\nResampled class distribution:")
    for class_label, count in result['resampled_distribution'].items():
        print(f"  Class {class_label}: {count} samples")
    
    print("\nSynthetic samples generated per class:")
    for class_label, count in result['synthetic_samples_per_class'].items():
        print(f"  Class {class_label}: {count} synthetic samples")
    
    # Compare model performance before and after SMOTE
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                        random_state=42, stratify=y)
    
    # Train on original data
    clf_original = RandomForestClassifier(random_state=42)
    clf_original.fit(X_train, y_train)
    y_pred_original = clf_original.predict(X_test)
    
    # Apply SMOTE to training data only
    smote_result = smote(X_train, y_train, random_state=42)
    X_train_smote = smote_result['X_resampled']
    y_train_smote = smote_result['y_resampled']
    
    # Train on SMOTE-enhanced data
    clf_smote = RandomForestClassifier(random_state=42)
    clf_smote.fit(X_train_smote, y_train_smote)
    y_pred_smote = clf_smote.predict(X_test)
    
    print("\nClassification Report - Original Data:")
    print(classification_report(y_test, y_pred_original))
    
    print("\nClassification Report - After SMOTE:")
    print(classification_report(y_test, y_pred_smote))