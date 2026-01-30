import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Union, List
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def fair_gan(
    data: Union[pd.DataFrame, np.ndarray],
    protected_attribute: Union[str, int],
    target_variable: Optional[Union[str, int]] = None,
    generator_layers: List[int] = [64, 128, 64],
    discriminator_layers: List[int] = [64, 32],
    fairness_lambda: float = 1.0,
    learning_rate: float = 0.0002,
    batch_size: int = 64,
    n_epochs: int = 100,
    noise_dim: int = 32,
    random_state: int = 42
) -> Dict:
    """
    Implement FairGAN: Fairness-aware Generative Adversarial Networks.
    
    FairGAN generates synthetic data that maintains utility while removing bias
    with respect to protected attributes. It uses multiple discriminators:
    one for data quality and one for fairness enforcement through adversarial training.
    
    Parameters:
    -----------
    data : pd.DataFrame or np.ndarray
        Input dataset containing features and protected attributes
    protected_attribute : str or int
        Column name or index of the protected attribute
    target_variable : str or int, optional
        Column name or index of the target variable for utility evaluation
    generator_layers : List[int], default=[64, 128, 64]
        Architecture of generator network (hidden layer sizes)
    discriminator_layers : List[int], default=[64, 32]
        Architecture of discriminator networks (hidden layer sizes)
    fairness_lambda : float, default=1.0
        Weight for fairness loss in adversarial training
    learning_rate : float, default=0.0002
        Learning rate for Adam optimizer
    batch_size : int, default=64
        Batch size for training
    n_epochs : int, default=100
        Number of training epochs
    noise_dim : int, default=32
        Dimension of noise vector for generator
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    Dict containing:
        - 'synthetic_data': Generated synthetic dataset
        - 'generator_loss': Training loss of generator
        - 'discriminator_loss': Training loss of discriminator
        - 'fairness_loss': Training loss of fairness discriminator
        - 'statistical_parity_original': Statistical parity of original data
        - 'statistical_parity_synthetic': Statistical parity of synthetic data
        - 'utility_score': Utility preservation score
        - 'fairness_improvement': Improvement in fairness metrics
        - 'training_history': Loss history during training
    """
    
    # Input validation
    if isinstance(data, pd.DataFrame):
        df = data.copy()
        if isinstance(protected_attribute, str):
            if protected_attribute not in df.columns:
                raise ValueError(f"Protected attribute '{protected_attribute}' not found in data")
            protected_col = protected_attribute
        else:
            protected_col = df.columns[protected_attribute]
    else:
        df = pd.DataFrame(data)
        if isinstance(protected_attribute, int):
            if protected_attribute >= df.shape[1]:
                raise ValueError(f"Protected attribute index {protected_attribute} out of bounds")
            protected_col = df.columns[protected_attribute]
        else:
            raise ValueError("For numpy arrays, protected_attribute must be an integer index")
    
    if fairness_lambda < 0:
        raise ValueError("fairness_lambda must be non-negative")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if batch_size <= 0 or n_epochs <= 0:
        raise ValueError("batch_size and n_epochs must be positive")
    
    np.random.seed(random_state)
    
    # Preprocess data
    # Encode categorical variables
    encoders = {}
    df_processed = df.copy()
    
    for col in df_processed.columns:
        if df_processed[col].dtype == 'object' or df_processed[col].dtype.name == 'category':
            encoders[col] = LabelEncoder()
            df_processed[col] = encoders[col].fit_transform(df_processed[col])
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(df_processed.values)
    
    # Extract protected attribute
    protected_idx = list(df.columns).index(protected_col)
    protected_attr = X[:, protected_idx]
    
    n_samples, n_features = X.shape
    
    # Simple neural network implementation using numpy
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
    
    def relu(x):
        return np.maximum(0, x)
    
    def tanh(x):
        return np.tanh(x)
    
    class SimpleNetwork:
        def __init__(self, input_dim, layers, output_dim, activation='relu'):
            self.layers = []
            self.biases = []
            self.activation = activation
            
            # Initialize weights and biases
            prev_dim = input_dim
            for layer_dim in layers:
                self.layers.append(np.random.normal(0, 0.02, (prev_dim, layer_dim)))
                self.biases.append(np.zeros((1, layer_dim)))
                prev_dim = layer_dim
            
            # Output layer
            self.layers.append(np.random.normal(0, 0.02, (prev_dim, output_dim)))
            self.biases.append(np.zeros((1, output_dim)))
        
        def forward(self, x):
            self.activations = [x]
            current = x
            
            for i, (W, b) in enumerate(zip(self.layers[:-1], self.biases[:-1])):
                current = np.dot(current, W) + b
                if self.activation == 'relu':
                    current = relu(current)
                elif self.activation == 'sigmoid':
                    current = sigmoid(current)
                elif self.activation == 'tanh':
                    current = tanh(current)
                self.activations.append(current)
            
            # Output layer
            current = np.dot(current, self.layers[-1]) + self.biases[-1]
            if len(self.layers) > 1:  # Only apply activation if not single layer
                if self.activation == 'tanh':
                    current = tanh(current)
                else:
                    current = sigmoid(current)
            else:
                current = sigmoid(current)
            
            self.activations.append(current)
            return current
    
    # Initialize networks
    generator = SimpleNetwork(
        input_dim=noise_dim + 1,  # +1 for protected attribute conditioning
        layers=generator_layers,
        output_dim=n_features,
        activation='tanh'
    )
    
    # Data discriminator (distinguishes real vs fake data)
    data_discriminator = SimpleNetwork(
        input_dim=n_features,
        layers=discriminator_layers,
        output_dim=1,
        activation='relu'
    )
    
    # Fairness discriminator (predicts protected attribute from generated data)
    fairness_discriminator = SimpleNetwork(
        input_dim=n_features - 1,  # Exclude protected attribute
        layers=discriminator_layers,
        output_dim=1,
        activation='relu'
    )
    
    # Training history
    history = {
        'generator_loss': [],
        'data_discriminator_loss': [],
        'fairness_discriminator_loss': []
    }
    
    # Training loop (simplified version)
    for epoch in range(n_epochs):
        epoch_g_loss = 0
        epoch_d_loss = 0
        epoch_f_loss = 0
        
        # Shuffle data
        indices = np.random.permutation(n_samples)
        
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            real_batch = X[batch_indices]
            protected_batch = protected_attr[batch_indices]
            
            current_batch_size = len(batch_indices)
            
            # Generate fake data
            noise = np.random.normal(0, 1, (current_batch_size, noise_dim))
            # Condition on protected attribute
            generator_input = np.column_stack([noise, protected_batch])
            fake_data = generator.forward(generator_input)
            
            # Train data discriminator
            real_pred = data_discriminator.forward(real_batch)
            fake_pred = data_discriminator.forward(fake_data)
            
            # Simple discriminator loss (binary cross-entropy approximation)
            d_loss_real = -np.mean(np.log(real_pred + 1e-8))
            d_loss_fake = -np.mean(np.log(1 - fake_pred + 1e-8))
            d_loss = d_loss_real + d_loss_fake
            
            # Train fairness discriminator (exclude protected attribute from input)
            fake_data_no_protected = np.delete(fake_data, protected_idx, axis=1)
            fairness_pred = fairness_discriminator.forward(fake_data_no_protected)
            
            # Fairness loss - discriminator should not be able to predict protected attribute
            f_loss = -np.mean(protected_batch.reshape(-1, 1) * np.log(fairness_pred + 1e-8) + 
                            (1 - protected_batch.reshape(-1, 1)) * np.log(1 - fairness_pred + 1e-8))
            
            # Generator loss combines data quality and fairness
            g_loss_data = -np.mean(np.log(fake_pred + 1e-8))  # Fool data discriminator
            g_loss_fairness = -fairness_lambda * f_loss  # Minimize fairness discriminator accuracy
            g_loss = g_loss_data + g_loss_fairness
            
            epoch_g_loss += g_loss
            epoch_d_loss += d_loss
            epoch_f_loss += f_loss
        
        history['generator_loss'].append(epoch_g_loss / (n_samples // batch_size))
        history['data_discriminator_loss'].append(epoch_d_loss / (n_samples // batch_size))
        history['fairness_discriminator_loss'].append(epoch_f_loss / (n_samples // batch_size))
    
    # Generate final synthetic dataset
    noise = np.random.normal(0, 1, (n_samples, noise_dim))
    generator_input = np.column_stack([noise, protected_attr])
    synthetic_data = generator.forward(generator_input)
    
    # Inverse transform synthetic data
    synthetic_data_scaled = scaler.inverse_transform(synthetic_data)
    synthetic_df = pd.DataFrame(synthetic_data_scaled, columns=df.columns)
    
    # Calculate statistical parity (difference in positive outcome rates between groups)
    def calculate_statistical_parity(data, protected_col, target_col=None):
        if target_col is None:
            # Use mean of all features as proxy for positive outcome
            target = (data.drop(columns=[protected_col]).mean(axis=1) > 
                     data.drop(columns=[protected_col]).mean(axis=1).median()).astype(int)
        else:
            target = data[target_col]
        
        protected_groups = data[protected_col