import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    logging.warning('PyTorch not available. DLV2 will use sklearn fallback impute_mice.')
from spatial import prepare_spatial_temporal_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MODEL_NAME = "EnhancedAutoencoder"


if TORCH_AVAILABLE:
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    class EnhancedAutoencoder(nn.Module):
        """Enhanced Autoencoder with skip connections and batch normalization"""
        def __init__(self, input_dim, encoding_dims=None, dropout=0.2):
            super(EnhancedAutoencoder, self).__init__()
            if encoding_dims is None:
                encoding_dims = [256, 128, 64]

            self.input_dim = input_dim
            self.encoding_dims = encoding_dims

            encoder_layers = []
            prev_dim = input_dim
            for dim in encoding_dims:
                encoder_layers.extend([
                    nn.Linear(prev_dim, dim),
                    nn.BatchNorm1d(dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
                prev_dim = dim

            self.encoder = nn.Sequential(*encoder_layers)

            # Decoder (symmetric to encoder)
            decoding_dims = encoding_dims[::-1][1:] + [input_dim]
            decoder_layers = []
            prev_dim = encoding_dims[-1]
            for i, dim in enumerate(decoding_dims):
                if i < len(decoding_dims) - 1:
                    decoder_layers.extend([
                        nn.Linear(prev_dim, dim),
                        nn.BatchNorm1d(dim),
                        nn.ReLU(),
                        nn.Dropout(dropout)
                    ])
                else:
                    decoder_layers.append(nn.Linear(prev_dim, dim))
                prev_dim = dim

            self.decoder = nn.Sequential(*decoder_layers)

        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded


    def impute_mice(data, target_column, input_columns, max_iter=100, random_state=42, tol=0.01,
                    custom_strategies=None, spatial_config=None):
        """
        DLV2 imputation using PyTorch autoencoder when available.
        Falls back to sklearn IterativeImputer when torch not available.
        """
        logging.info(f"Starting {MODEL_NAME} imputation...")
        np.random.seed(random_state)
        torch.manual_seed(random_state)

        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")

        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in the data.")

        if target_column in input_columns:
            input_columns = [col for col in input_columns if col != target_column]

        if spatial_config and (spatial_config.get('use_spatial', False) or spatial_config.get('use_temporal', False)):
            data_enhanced, enhanced_input_columns = prepare_spatial_temporal_data(
                data, target_column, input_columns, spatial_config
            )
            input_columns_to_use = enhanced_input_columns
            data_to_use = data_enhanced
        else:
            input_columns_to_use = input_columns
            data_to_use = data.copy()

        if custom_strategies:
            for col, strategy in custom_strategies.items():
                if col in data_to_use.columns:
                    if strategy == "mean":
                        data_to_use[col] = data_to_use[col].fillna(data_to_use[col].mean())
                    elif strategy == "median":
                        data_to_use[col] = data_to_use[col].fillna(data_to_use[col].median())
                    elif isinstance(strategy, (int, float)):
                        data_to_use[col] = data_to_use[col].fillna(strategy)

        columns_for_imputation = input_columns_to_use + [target_column]
        columns_for_imputation = [col for col in columns_for_imputation if col in data_to_use.columns]
        data_for_imputation = data_to_use[columns_for_imputation].copy()

        # Use torch autoencoder if TORCH_AVAILABLE else IterativeImputer
        if TORCH_AVAILABLE:
            scaler = StandardScaler()
            data_filled = data_for_imputation.fillna(data_for_imputation.mean())
            data_scaled = scaler.fit_transform(data_filled)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            X_tensor = torch.FloatTensor(data_scaled).to(device)

            dataset = TensorDataset(X_tensor)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

            input_dim = data_scaled.shape[1]
            model = EnhancedAutoencoder(input_dim=input_dim, encoding_dims=[256,128,64], dropout=0.2).to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()

            model.train()
            for epoch in range(max_iter):
                epoch_loss = 0
                for batch in dataloader:
                    X_batch = batch[0]
                    optimizer.zero_grad()
                    predictions = model(X_batch)
                    loss = criterion(predictions, X_batch)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

            model.eval()
            with torch.no_grad():
                reconstructed = model(torch.FloatTensor(scaler.transform(data_for_imputation.fillna(0).values)).to(device))
                reconstructed = reconstructed.cpu().numpy()
                imputed = scaler.inverse_transform(reconstructed)

            imputed_df = pd.DataFrame(imputed, columns=columns_for_imputation, index=data_for_imputation.index)
            final_df = data.copy()
            for col in columns_for_imputation:
                if col in data.columns:
                    final_df[col] = imputed_df[col]
            return final_df

# If torch is not available we must provide an impute_mice entrypoint
if not TORCH_AVAILABLE:
    def impute_mice(data, target_column, input_columns, max_iter=100, random_state=42, tol=0.01,
                    custom_strategies=None, spatial_config=None):
        """
        Fallback DLV2 imputation using sklearn IterativeImputer when PyTorch is absent.
        """
        logging.info(f"PyTorch not available. Running DLV2 sklearn fallback impute_mice...")

        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")

        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in the data.")

        if target_column in input_columns:
            input_columns = [col for col in input_columns if col != target_column]

        if spatial_config and (spatial_config.get('use_spatial', False) or spatial_config.get('use_temporal', False)):
            data_enhanced, enhanced_input_columns = prepare_spatial_temporal_data(
                data, target_column, input_columns, spatial_config
            )
            input_columns_to_use = enhanced_input_columns
            data_to_use = data_enhanced
        else:
            input_columns_to_use = input_columns
            data_to_use = data.copy()

        if custom_strategies:
            for col, strategy in custom_strategies.items():
                if col in data_to_use.columns:
                    if strategy == "mean":
                        data_to_use[col] = data_to_use[col].fillna(data_to_use[col].mean())
                    elif strategy == "median":
                        data_to_use[col] = data_to_use[col].fillna(data_to_use[col].median())
                    elif isinstance(strategy, (int, float)):
                        data_to_use[col] = data_to_use[col].fillna(strategy)

        columns_for_imputation = input_columns_to_use + [target_column]
        columns_for_imputation = [col for col in columns_for_imputation if col in data_to_use.columns]
        data_for_imputation = data_to_use[columns_for_imputation].copy()

        # Use sklearn IterativeImputer as fallback
        try:
            imputer = IterativeImputer(random_state=random_state, max_iter=max_iter, tol=tol)
            filled = imputer.fit_transform(data_for_imputation)
            imputed_df = pd.DataFrame(filled, columns=columns_for_imputation, index=data_for_imputation.index)

            final_df = data.copy()
            for col in columns_for_imputation:
                if col in final_df.columns:
                    final_df[col] = imputed_df[col]
            return final_df
        except Exception as e:
            logging.error(f"DLV2 sklearn fallback imputation failed: {e}")
            # Last resort: simple mean fill
            final_df = data.copy()
            for col in columns_for_imputation:
                if col in final_df.columns:
                    final_df[col] = final_df[col].fillna(final_df[col].mean())
            return final_df