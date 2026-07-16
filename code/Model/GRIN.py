import pandas as pd
import numpy as np
import logging
import warnings
from sklearn.exceptions import ConvergenceWarning
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
    logging.warning("PyTorch not available. GRIN will use sklearn fallback implementation.")
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer

# Define the model name
MODEL_NAME = "GRIN"


if HAS_TORCH:
    class GraphConvolution(nn.Module):
        """Graph Convolution Layer"""
        def __init__(self, in_features, out_features):
            super(GraphConvolution, self).__init__()
            self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
            self.reset_parameters()

        def reset_parameters(self):
            nn.init.xavier_uniform_(self.weight)
            nn.init.zeros_(self.bias)

        def forward(self, x, adj):
            """
            Args:
                x: Node features (batch, nodes, features)
                adj: Adjacency matrix (nodes, nodes)
            """
            support = torch.matmul(x, self.weight)
            output = torch.matmul(adj, support) + self.bias
            return output
else:
    pass
    """Graph Convolution Layer"""
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch. FloatTensor(out_features))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, x, adj):
        """
        Args:
            x: Node features (batch, nodes, features)
            adj: Adjacency matrix (nodes, nodes)
        """
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support) + self.bias
        return output


if HAS_TORCH:
    class GRINModel(nn.Module):
        """Graph Recurrent Imputation Network"""
        def __init__(self, n_features, n_nodes, hidden_dim=64, gnn_layers=2, rnn_layers=2, dropout=0.2):
            super(GRINModel, self).__init__()

            self.n_features = n_features
            self.n_nodes = n_nodes
            self.hidden_dim = hidden_dim

            # Graph Convolutional Layers
            self.gcn_layers = nn.ModuleList()
            self.gcn_layers.append(GraphConvolution(n_features, hidden_dim))
            for _ in range(gnn_layers - 1):
                self.gcn_layers.append(GraphConvolution(hidden_dim, hidden_dim))

            # Bidirectional GRU for temporal patterns
            self.gru = nn.GRU(
                hidden_dim * n_nodes,
                hidden_dim,
                num_layers=rnn_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if rnn_layers > 1 else 0
            )

            # Output layers
            self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, n_features * n_nodes)
            self.dropout = nn.Dropout(dropout)
            self.relu = nn.ReLU()

        def forward(self, x, adj, mask):
            """
            Args:
                x: Input features (batch, time, nodes, features)
                adj: Adjacency matrix (nodes, nodes)
                mask: Missing value mask (batch, time, nodes, features)
            """
            batch_size, time_steps, n_nodes, n_features = x.shape

            # Apply graph convolutions at each time step
            gcn_out = []
            for t in range(time_steps):
                x_t = x[:, t, :, :]  # (batch, nodes, features)

                # Apply GCN layers
                h = x_t
                for gcn in self.gcn_layers:
                    h = self.relu(gcn(h, adj))
                    h = self.dropout(h)

                gcn_out.append(h)

            # Stack temporal features
            gcn_out = torch.stack(gcn_out, dim=1)  # (batch, time, nodes, hidden)
            gcn_out = gcn_out.reshape(batch_size, time_steps, -1)  # (batch, time, nodes*hidden)

            # Apply bidirectional GRU
            rnn_out, _ = self.gru(gcn_out)  # (batch, time, hidden*2)

            # Decode
            out = self.relu(self.fc1(rnn_out))
            out = self.dropout(out)
            out = self.fc2(out)  # (batch, time, nodes*features)

            # Reshape to original dimensions
            out = out.reshape(batch_size, time_steps, n_nodes, n_features)

            return out
else:
    pass


def create_adjacency_matrix(spatial_features, threshold=0.5):
    """
    Create adjacency matrix from spatial features based on correlation. 
    
    Args:
        spatial_features: DataFrame with spatial features
        threshold:  Correlation threshold for edge creation
    
    Returns: 
        Adjacency matrix as torch tensor
    """
    if spatial_features.shape[1] == 0:
        # No spatial features, return identity matrix (numpy)
        return np.eye(1)

    # Calculate correlation between sites
    corr = spatial_features.corr().abs()

    # Create adjacency matrix
    adj = (corr > threshold).astype(float).values

    # Add self-loops
    np.fill_diagonal(adj, 1.0)

    # Normalize adjacency matrix (symmetric normalization)
    rowsum = adj.sum(1)
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    adj_normalized = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

    return adj_normalized


def impute_mice(data, target_column, input_columns, max_iter=50, random_state=42, tol=0.01, 
                custom_strategies=None, spatial_config=None):
    """
    Perform GRIN imputation on the given data with spatial-temporal features. 

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
        pd.DataFrame: Imputed data. 
    """
    logging.info("Starting GRIN imputation...")

    # Set random seed
    if HAS_TORCH:
        try:
            torch.manual_seed(random_state)
        except Exception:
            pass
    np.random.seed(random_state)
    
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
            if col in data_to_use.columns:
                if strategy == "mean":
                    data_to_use[col] = data_to_use[col].fillna(data_to_use[col].mean())
                elif strategy == "median":
                    data_to_use[col] = data_to_use[col].fillna(data_to_use[col].median())
                elif isinstance(strategy, (int, float)):
                    data_to_use[col] = data_to_use[col].fillna(strategy)
    
    # Prepare data for GRIN
    columns_for_imputation = input_columns_to_use + [target_column]
    columns_for_imputation = [col for col in columns_for_imputation if col in data_to_use.columns]
    data_for_imputation = data_to_use[columns_for_imputation]. copy()

    # If PyTorch is not available, use sklearn IterativeImputer fallback
    if not HAS_TORCH:
        logging.info("PyTorch not available. GRIN fallback: using sklearn IterativeImputer")
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer

        imputer = IterativeImputer(random_state=random_state, max_iter=max_iter if max_iter is not None else 10, tol=tol)
        arr = data_for_imputation.values
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ConvergenceWarning)
            imputed = imputer.fit_transform(arr)

        imputed_df = pd.DataFrame(imputed, columns=columns_for_imputation, index=data_for_imputation.index)
        final_df = data.copy()
        for col in columns_for_imputation:
            if col in data.columns:
                final_df[col] = imputed_df[col]

        logging.info("GRIN fallback imputation completed successfully.")
        return final_df
    
    # Identify spatial features
    spatial_cols = [col for col in columns_for_imputation if col.startswith('spatial_')]
    
    # Create adjacency matrix from spatial features
    if spatial_cols:
        spatial_data = data_for_imputation[spatial_cols].fillna(0)
        adj_matrix = create_adjacency_matrix(spatial_data)
        n_nodes = len(spatial_cols) + 1  # +1 for target site
    else:
        adj_matrix = torch.eye(1)
        n_nodes = 1
    
    # Standardize data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_for_imputation)
    data_scaled = pd.DataFrame(data_scaled, columns=columns_for_imputation, index=data_for_imputation.index)
    
    # Create mask for missing values
    mask = data_for_imputation.isna().values.astype(float)
    
    # Fill missing values temporarily with 0
    data_filled = np.nan_to_num(data_scaled.values, nan=0.0)
    
    # Prepare sequences (use sliding window for time series)
    sequence_length = min(24, len(data_filled))  # 24-hour window
    n_features = data_filled.shape[1]
    
    # Reshape data for GRIN:  (batch, time, nodes, features)
    # For simplicity, treat each feature as a node
    X = torch.FloatTensor(data_filled).unsqueeze(0).unsqueeze(2)  # (1, time, 1, features)
    X = X.repeat(1, 1, n_nodes, 1)  # Replicate for nodes
    mask_tensor = torch.FloatTensor(mask).unsqueeze(0).unsqueeze(2).repeat(1, 1, n_nodes, 1)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GRINModel(
        n_features=n_features,
        n_nodes=n_nodes,
        hidden_dim=64,
        gnn_layers=2,
        rnn_layers=2,
        dropout=0.2
    ).to(device)
    
    X = X.to(device)
    mask_tensor = mask_tensor.to(device)
    adj_matrix = adj_matrix.to(device)
    
    # Training
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn. MSELoss()
    
    logging.info("Training GRIN model...")
    model.train()
    
    for epoch in range(max_iter):
        optimizer.zero_grad()
        
        # Forward pass
        output = model(X, adj_matrix, mask_tensor)
        
        # Calculate loss only on observed values
        observed_mask = 1 - mask_tensor
        loss = criterion(output * observed_mask, X * observed_mask)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            logging.info(f"Epoch {epoch + 1}/{max_iter}, Loss: {loss.item():.6f}")
    
    # Inference
    model.eval()
    with torch.no_grad():
        imputed_output = model(X, adj_matrix, mask_tensor)
        imputed_output = imputed_output.cpu().numpy()
    
    # Extract imputed values (take first node)
    imputed_values = imputed_output[0, : , 0, :]
    
    # Inverse transform
    imputed_df = pd.DataFrame(
        scaler.inverse_transform(imputed_values),
        columns=columns_for_imputation,
        index=data_for_imputation.index
    )
    
    # Create final output
    final_df = data. copy()
    for col in columns_for_imputation:
        if col in data. columns:
            final_df[col] = imputed_df[col]
    
    logging.info("GRIN imputation completed successfully.")
    return final_df