import pandas as pd
import numpy as np
import logging
import warnings
from spatial import prepare_spatial_temporal_data
from sklearn.preprocessing import StandardScaler

# Optional torch import (provide sklearn fallback when unavailable)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.warning("PyTorch not available. GAIN will use sklearn fallback implementation.")
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    from sklearn.exceptions import ConvergenceWarning

# Define the model name
MODEL_NAME = "GAIN"


if HAS_TORCH:
    class Generator(nn.Module):
        """Generator network for GAIN"""
        def __init__(self, input_dim, hidden_dim=128):
            super(Generator, self).__init__()

            self.net = nn.Sequential(
                nn.Linear(input_dim * 2, hidden_dim),  # input_dim * 2 for data + mask
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, input_dim),
                nn.Sigmoid()
            )

        def forward(self, x, mask):
            """
            Args:
                x: Input data with missing values filled with random noise
                mask: Binary mask (1 = observed, 0 = missing)
            """
            inputs = torch.cat([x, mask], dim=-1)
            imputed = self.net(inputs)
            # Combine observed and imputed values
            output = mask * x + (1 - mask) * imputed
            return output


    class Discriminator(nn.Module):
        """Discriminator network for GAIN"""
        def __init__(self, input_dim, hidden_dim=128):
            super(Discriminator, self).__init__()

            self.net = nn.Sequential(
                nn.Linear(input_dim * 2, hidden_dim),  # input_dim * 2 for data + hint
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, input_dim),
                nn.Sigmoid()
            )

        def forward(self, x, hint):
            """
            Args:
                x:  Imputed data
                hint: Hint vector (helps discriminator)
            """
            inputs = torch.cat([x, hint], dim=-1)
            return self.net(inputs)


def sample_hint(mask, hint_rate=0.9):
    """
    Sample hint vector for discriminator. 
    
    Args:
        mask: Binary mask (1 = observed, 0 = missing)
        hint_rate: Probability of revealing true mask to discriminator
    """
    hint = torch.rand_like(mask)
    hint = (hint < hint_rate).float()
    hint = hint * mask + 0.5 * (1 - hint)
    return hint


def impute_mice(data, target_column, input_columns, max_iter=1000, random_state=42, tol=0.01,
                custom_strategies=None, spatial_config=None):
    """
    Perform GAIN imputation on the given data with spatial-temporal features.

    Args:
        data (pd.DataFrame): Input data with missing values.
        target_column (str): The target column to impute.
        input_columns (list): List of input columns for the imputation model.
        max_iter (int): Maximum number of training iterations.
        random_state (int): Random seed for reproducibility. 
        tol (float): Tolerance for convergence (not used in GAIN).
        custom_strategies (dict): Dictionary of column-specific imputation strategies.
        spatial_config (dict): Configuration for spatial-temporal features.

    Returns:
        pd.DataFrame: Imputed data.
    """
    logging.info("Starting GAIN imputation...")

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
        logging.warning(f"Target column '{target_column}' is in input_columns. Removing it.")
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
        data_to_use = data. copy()
    
    # Apply custom strategies if provided
    if custom_strategies:
        logging.info("Applying custom imputation strategies...")
        for col, strategy in custom_strategies.items():
            if col in data_to_use. columns:
                if strategy == "mean":
                    data_to_use[col] = data_to_use[col].fillna(data_to_use[col].mean())
                elif strategy == "median":
                    data_to_use[col] = data_to_use[col].fillna(data_to_use[col].median())
                elif isinstance(strategy, (int, float)):
                    data_to_use[col] = data_to_use[col].fillna(strategy)
    
    # Prepare data for GAIN
    columns_for_imputation = input_columns_to_use + [target_column]
    columns_for_imputation = [col for col in columns_for_imputation if col in data_to_use.columns]
    data_for_imputation = data_to_use[columns_for_imputation].copy()
    
    # If torch is available, run GAIN; otherwise use sklearn IterativeImputer fallback
    if HAS_TORCH:
        # Standardize data to [0, 1] range (GAIN works best with normalized data)
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        data_values = data_for_imputation.values

        # Create mask for missing values (1 = observed, 0 = missing)
        mask = (~np.isnan(data_values)).astype(float)

        # Fill missing values temporarily with random noise
        data_filled = data_values.copy()
        nan_mask = np.isnan(data_filled)
        data_filled[nan_mask] = np.random.uniform(0, 1, size=nan_mask.sum())

        # Scale data
        data_scaled = scaler.fit_transform(data_filled)

        # Convert to tensors
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X = torch.FloatTensor(data_scaled).to(device)
        M = torch.FloatTensor(mask).to(device)

        # Initialize models
        input_dim = data_scaled.shape[1]
        generator = Generator(input_dim, hidden_dim=128).to(device)
        discriminator = Discriminator(input_dim, hidden_dim=128).to(device)

        # Optimizers
        g_optimizer = optim.Adam(generator.parameters(), lr=0.001)
        d_optimizer = optim.Adam(discriminator.parameters(), lr=0.001)

        # Loss functions
        bce_loss = nn.BCELoss()
        mse_loss = nn.MSELoss()

        logging.info("Training GAIN model...")

        # Training loop
        for iteration in range(max_iter):
            # Sample random noise for missing values
            Z = torch.rand_like(X).to(device)
            X_noisy = M * X + (1 - M) * Z

            # ==================== Train Discriminator ====================
            d_optimizer.zero_grad()

            # Generate imputed data
            G_sample = generator(X_noisy, M)

            # Sample hint
            H = sample_hint(M, hint_rate=0.9)

            # Discriminator output
            D_prob = discriminator(G_sample, H)

            # Discriminator loss
            d_loss = -torch.mean(M * torch.log(D_prob + 1e-8) + (1 - M) * torch.log(1 - D_prob + 1e-8))

            d_loss.backward()
            d_optimizer.step()

            # ==================== Train Generator ====================
            g_optimizer.zero_grad()

            # Sample new noise
            Z = torch.rand_like(X).to(device)
            X_noisy = M * X + (1 - M) * Z

            # Generate imputed data
            G_sample = generator(X_noisy, M)

            # Sample hint
            H = sample_hint(M, hint_rate=0.9)

            # Discriminator output
            D_prob = discriminator(G_sample, H)

            # Generator loss (fool discriminator + reconstruction)
            g_loss_adv = -torch.mean((1 - M) * torch.log(D_prob + 1e-8))
            g_loss_mse = mse_loss(M * G_sample, M * X)
            g_loss = g_loss_adv + 10 * g_loss_mse

            g_loss.backward()
            g_optimizer.step()

            if (iteration + 1) % 100 == 0:
                logging.info(f"Iteration {iteration + 1}/{max_iter}, D Loss: {d_loss.item():.6f}, G Loss: {g_loss.item():.6f}")

        # Final imputation
        generator.eval()
        with torch.no_grad():
            Z = torch.rand_like(X).to(device)
            X_noisy = M * X + (1 - M) * Z
            imputed_output = generator(X_noisy, M)
            imputed_output = imputed_output.cpu().numpy()

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

        logging.info("GAIN imputation completed successfully.")
        return final_df
    else:
        # Fallback: use IterativeImputer (sklearn)
        logging.info("GAIN fallback: using sklearn IterativeImputer")
        imputer = IterativeImputer(random_state=random_state, max_iter=max_iter if max_iter is not None else 10, tol=tol)
        arr = data_for_imputation.values

        # Suppress ConvergenceWarning from IterativeImputer and perform imputation
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ConvergenceWarning)
            imputed = imputer.fit_transform(arr)

        imputed_df = pd.DataFrame(imputed, columns=columns_for_imputation, index=data_for_imputation.index)

        final_df = data.copy()
        for col in columns_for_imputation:
            if col in data.columns:
                final_df[col] = imputed_df[col]

        logging.info("GAIN fallback imputation completed successfully.")
        return final_df