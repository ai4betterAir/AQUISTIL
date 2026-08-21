import pandas as pd
import numpy as np
import logging
from spatial import prepare_spatial_temporal_data
from sklearn.preprocessing import StandardScaler
import math

# Optional torch import (provide sklearn fallback when unavailable)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.warning("PyTorch not available. TransformerImpute will use sklearn fallback implementation.")
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer

# Define the model name
MODEL_NAME = "TransformerImpute"

if HAS_TORCH:
    class PositionalEncoding(nn.Module):
        """Positional encoding for transformer"""
        def __init__(self, d_model, max_len=5000):
            super(PositionalEncoding, self).__init__()

            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)

            self.register_buffer('pe', pe)

        def forward(self, x):
            """
            Args:
                x:  Tensor of shape (batch, seq_len, d_model)
            """
            return x + self.pe[:, :x.size(1), :]


if HAS_TORCH:
    class TransformerImputationModel(nn.Module):
        """Transformer model for imputation with spatial embeddings"""
        def __init__(self, n_features, d_model=128, nhead=8, num_layers=4, dim_feedforward=512, dropout=0.1):
            super(TransformerImputationModel, self).__init__()

            self.n_features = n_features
            self.d_model = d_model

            # Input projection
            self.input_proj = nn.Linear(n_features * 2, d_model)  # *2 for data + mask

            # Positional encoding
            self.pos_encoder = PositionalEncoding(d_model)

            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

            # Output projection
            self.output_proj = nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, n_features)
            )

        def forward(self, x, mask):
            """
            Args:
                x: Input data (batch, seq_len, n_features)
                mask: Missing mask (batch, seq_len, n_features)
            """
            # Concatenate data and mask
            x_masked = torch.cat([x, mask], dim=-1)

            # Project to model dimension
            x_proj = self.input_proj(x_masked)

            # Add positional encoding
            x_pos = self.pos_encoder(x_proj)

            # Create attention mask (prevent attention to padding if needed)
            # For now, allow attention to all positions
            attn_mask = None

            # Apply transformer
            transformer_out = self.transformer_encoder(x_pos, mask=attn_mask)

            # Project to output
            output = self.output_proj(transformer_out)

            return output


def impute_mice(data, target_column, input_columns, max_iter=100, random_state=42, tol=0.01,
                custom_strategies=None, spatial_config=None):
    """
    Perform Transformer imputation on the given data with spatial-temporal features.

    Args:
        data (pd.DataFrame): Input data with missing values. 
        target_column (str): The target column to impute.
        input_columns (list): List of input columns for the imputation model. 
        max_iter (int): Maximum number of training epochs.
        random_state (int): Random seed for reproducibility. 
        tol (float): Tolerance for convergence. 
        custom_strategies (dict): Dictionary of column-specific imputation strategies.
        spatial_config (dict): Configuration for spatial-temporal features. 

    Returns:
        pd. DataFrame: Imputed data. 
    """
    logging.info("Starting Transformer imputation...")

    np.random.seed(random_state)
    if HAS_TORCH:
        torch.manual_seed(random_state)
    
    # Input validation
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")
    
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in the data.")
    
    # Ensure target column is not in input columns
    if target_column in input_columns:
        logging.warning(f"Target column '{target_column}' is in input_columns.  Removing it.")
        input_columns = [col for col in input_columns if col != target_column]
    
    # Prepare data with spatial-temporal features if configured
    if spatial_config and (spatial_config.get('use_spatial', False) or spatial_config.get('use_temporal', False)):
        logging.info("Preparing data with spatial-temporal features...")
        data_enhanced, enhanced_input_columns = prepare_spatial_temporal_data(
            data, target_column, input_columns, spatial_config
        )
        input_columns_to_use = enhanced_input_columns
        data_to_use = data_enhanced
    else:
        input_columns_to_use = input_columns
        data_to_use = data.copy()
    
    # Apply custom strategies if provided
    if custom_strategies:
        logging.info("Applying custom imputation strategies...")
        for col, strategy in custom_strategies.items():
            if col in data_to_use.columns:
                if strategy == "mean":
                    data_to_use[col] = data_to_use[col].fillna(data_to_use[col].mean())
                elif strategy == "median":
                    data_to_use[col] = data_to_use[col].fillna(data_to_use[col].median())
                elif isinstance(strategy, (int, float)):
                    data_to_use[col] = data_to_use[col].fillna(strategy)
    
    # Prepare data for Transformer
    columns_for_imputation = input_columns_to_use + [target_column]
    columns_for_imputation = [col for col in columns_for_imputation if col in data_to_use.columns]
    data_for_imputation = data_to_use[columns_for_imputation].copy()
    
    # If torch available, run transformer; otherwise fallback to IterativeImputer
    if HAS_TORCH:
        # Standardize data
        scaler = StandardScaler()
        data_values = data_for_imputation.values

        # Create mask for missing values (1 = observed, 0 = missing)
        mask = (~np.isnan(data_values)).astype(float)

        # Fill missing values temporarily with 0
        data_filled = np.nan_to_num(data_values, nan=0.0)
        data_scaled = scaler.fit_transform(data_filled)

        # Convert to tensors
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X = torch.FloatTensor(data_scaled).unsqueeze(0).to(device)  # (1, seq_len, features)
        M = torch.FloatTensor(mask).unsqueeze(0).to(device)

        # Initialize model
        n_features = data_scaled.shape[1]
        model = TransformerImputationModel(
            n_features=n_features,
            d_model=128,
            nhead=8,
            num_layers=4,
            dim_feedforward=512,
            dropout=0.1
        ).to(device)

        # Training
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        criterion = nn.MSELoss()

        logging.info("Training Transformer model...")
        model.train()

        best_loss = float('inf')
        patience = 15
        patience_counter = 0

        for epoch in range(max_iter):
            optimizer.zero_grad()

            # Forward pass
            output = model(X, M)

            # Calculate loss only on observed values
            loss = criterion(output * M, X * M)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                logging.info(f"Epoch {epoch + 1}/{max_iter}, Loss: {loss.item():.6f}")

            # Early stopping
            if loss.item() < best_loss - tol:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logging.info(f"Early stopping at epoch {epoch + 1}")
                    break

        # Inference
        model.eval()
        with torch.no_grad():
            imputed_output = model(X, M)
            imputed_output = imputed_output.cpu().numpy()[0]

        # Inverse transform
        imputed_values = scaler.inverse_transform(imputed_output)
        imputed_df = pd.DataFrame(
            imputed_values,
            columns=columns_for_imputation,
            index=data_for_imputation.index
        )

        # Create final output
        final_df = data.copy()
        for col in columns_for_imputation:
            if col in data.columns:
                final_df[col] = imputed_df[col]

        logging.info("Transformer imputation completed successfully.")
        return final_df
    else:
        logging.info("Transformer fallback: using sklearn IterativeImputer")
        imputer = IterativeImputer(random_state=random_state, max_iter=10)
        arr = data_for_imputation.values
        imputed = imputer.fit_transform(arr)
        imputed_df = pd.DataFrame(imputed, columns=columns_for_imputation, index=data_for_imputation.index)

        final_df = data.copy()
        for col in columns_for_imputation:
            if col in data.columns:
                final_df[col] = imputed_df[col]

        logging.info("Transformer fallback imputation completed successfully.")
        return final_df