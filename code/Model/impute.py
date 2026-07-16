"""
Advanced Imputation Methods Collection
Includes:  Matrix Factorization, Tree-based, Bayesian, and Spatial methods

Author: Dr.  Masrur
Last Updated: 2026-01-02
"""

import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Try importing optional dependencies
try:
    from fancyimpute import SoftImpute as FancySoftImpute
    FANCYIMPUTE_AVAILABLE = True
except ImportError:
    FANCYIMPUTE_AVAILABLE = False
    logging.warning("fancyimpute not available. SoftImpute will use alternative implementation.")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("xgboost not available. XGBoost imputation disabled.")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logging.warning("lightgbm not available. LightGBM imputation disabled.")

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
    GP_AVAILABLE = True
except ImportError:
    GP_AVAILABLE = False
    logging.warning("Gaussian Process not available.")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging. warning("PyTorch not available.  SPIN will be disabled.")

from spatial import prepare_spatial_temporal_data


# ============================================================================
# MATRIX FACTORIZATION METHODS
# ============================================================================

class SoftImputeCustom:
    """
    Custom SoftImpute implementation using SVD with nuclear norm regularization.
    """
    def __init__(self, max_rank=None, lambda_reg=0.1, max_iter=100, tol=1e-5):
        self.max_rank = max_rank
        self.lambda_reg = lambda_reg
        self.max_iter = max_iter
        self.tol = tol
    
    def fit_transform(self, X):
        """
        Impute missing values using soft-thresholded SVD.
        
        Args:
            X:  numpy array with NaN for missing values
        
        Returns:
            Imputed matrix
        """
        # Create mask
        mask = ~np.isnan(X)
        
        # Initialize with mean
        X_filled = X.copy()
        col_means = np.nanmean(X, axis=0)
        for j in range(X.shape[1]):
            X_filled[np.isnan(X[: , j]), j] = col_means[j]
        
        for iteration in range(self.max_iter):
            X_old = X_filled.copy()
            
            # SVD
            U, s, Vt = np.linalg.svd(X_filled, full_matrices=False)
            
            # Soft thresholding
            s_thresh = np.maximum(s - self.lambda_reg, 0)
            
            # Determine rank
            if self.max_rank is not None: 
                rank = min(self.max_rank, np.sum(s_thresh > 0))
            else:
                rank = np.sum(s_thresh > 0)
            
            # Reconstruct
            if rank > 0:
                X_filled = U[:, :rank] @ np.diag(s_thresh[:rank]) @ Vt[:rank, :]
            
            # Keep observed values
            X_filled[mask] = X[mask]
            
            # Check convergence
            change = np.linalg.norm(X_filled - X_old) / np.linalg.norm(X_old)
            if change < self.tol:
                logging.info(f"SoftImpute converged at iteration {iteration + 1}")
                break
        
        return X_filled


class OptSpace:
    """
    OptSpace:  Matrix completion via optimization on manifold. 
    Based on "A Singular Value Thresholding Algorithm for Matrix Completion" (Cai et al., 2010)
    """
    def __init__(self, rank=10, max_iter=100, tol=1e-4):
        self.rank = rank
        self.max_iter = max_iter
        self.tol = tol
    
    def fit_transform(self, X):
        """
        Impute missing values using OptSpace algorithm.
        
        Args:
            X: numpy array with NaN for missing values
        
        Returns:
            Imputed matrix
        """
        m, n = X.shape
        mask = ~np.isnan(X)
        
        # Initialize with mean imputation
        X_filled = X.copy()
        col_means = np.nanmean(X, axis=0)
        for j in range(n):
            X_filled[np.isnan(X[:, j]), j] = col_means[j]
        
        # Trim to specified rank
        U, s, Vt = np.linalg.svd(X_filled, full_matrices=False)
        rank = min(self.rank, len(s))
        
        U = U[:, :rank]
        s = s[:rank]
        Vt = Vt[:rank, :]
        
        for iteration in range(self.max_iter):
            X_old = X_filled.copy()
            
            # Update U, S, V using gradient descent on observed entries
            X_approx = U @ np.diag(s) @ Vt
            
            # Compute residual on observed entries
            residual = np.zeros_like(X)
            residual[mask] = X[mask] - X_approx[mask]
            
            # Update factors
            # U update
            U_grad = -2 * residual @ Vt. T @ np.diag(s)
            U = U - 0.01 * U_grad
            
            # Orthogonalize U
            U, _ = np.linalg.qr(U)
            
            # V update
            V_grad = -2 * residual. T @ U @ np.diag(s)
            V = Vt. T - 0.01 * V_grad
            V, _ = np.linalg. qr(V)
            Vt = V. T
            
            # S update
            s = np.diag(U. T @ (X_filled * mask) @ Vt.T) / np.maximum(np.diag(U.T @ mask @ Vt.T), 1e-10)
            s = np.maximum(s, 0)
            
            # Reconstruct
            X_filled = U @ np.diag(s) @ Vt
            X_filled[mask] = X[mask]
            
            # Check convergence
            change = np. linalg.norm(X_filled - X_old) / np.linalg.norm(X_old)
            if change < self.tol:
                logging.info(f"OptSpace converged at iteration {iteration + 1}")
                break
        
        return X_filled


class SVT:
    """
    Singular Value Thresholding for matrix completion.
    Based on "A Singular Value Thresholding Algorithm for Matrix Completion" (Cai et al., 2010)
    """
    def __init__(self, tau=None, delta=None, max_iter=500, tol=1e-4):
        self.tau = tau
        self.delta = delta
        self.max_iter = max_iter
        self.tol = tol
    
    def fit_transform(self, X):
        """
        Impute missing values using Singular Value Thresholding. 
        
        Args:
            X: numpy array with NaN for missing values
        
        Returns: 
            Imputed matrix
        """
        m, n = X.shape
        mask = ~np.isnan(X)
        
        # Auto-set parameters if not provided
        if self. tau is None:
            self. tau = 5 * np.sqrt(m * n)
        
        if self.delta is None:
            self.delta = 1.2 * m * n / np.sum(mask)
        
        # Initialize
        Y = np.zeros_like(X)
        X_filled = np.zeros_like(X)
        
        for iteration in range(self.max_iter):
            X_old = X_filled.copy()
            
            # SVD
            U, s, Vt = np.linalg. svd(Y, full_matrices=False)
            
            # Soft thresholding
            s_thresh = np.maximum(s - self.tau, 0)
            
            # Reconstruct
            rank = np.sum(s_thresh > 0)
            if rank > 0:
                X_filled = U[:, :rank] @ np.diag(s_thresh[:rank]) @ Vt[:rank, :]
            
            # Update Y
            residual = np.zeros_like(X)
            residual[mask] = X[mask] - X_filled[mask]
            Y = Y + self.delta * residual
            
            # Check convergence
            rel_change = np.linalg.norm(X_filled[mask] - X_old[mask]) / (np.linalg.norm(X_old[mask]) + 1e-10)
            
            if (iteration + 1) % 50 == 0:
                logging.info(f"SVT iteration {iteration + 1}, relative change: {rel_change:.6f}")
            
            if rel_change < self.tol:
                logging.info(f"SVT converged at iteration {iteration + 1}")
                break
        
        return X_filled


# ============================================================================
# TREE-BASED METHODS
# ============================================================================

class MissForest:
    """
    missForest: Nonparametric missing value imputation using Random Forest.
    Based on Stekhoven & Bühlmann (2012)
    """
    def __init__(self, max_iter=10, n_estimators=100, random_state=42):
        self.max_iter = max_iter
        self.n_estimators = n_estimators
        self.random_state = random_state
    
    def fit_transform(self, X, cat_vars=None):
        """
        Impute missing values using Random Forest.
        
        Args:
            X: numpy array or DataFrame with NaN for missing values
            cat_vars: list of categorical variable indices (not used in this implementation)
        
        Returns:
            Imputed matrix
        """
        from sklearn.ensemble import RandomForestRegressor
        
        if isinstance(X, pd.DataFrame):
            columns = X.columns
            index = X.index
            X = X.values
        else:
            columns = None
            index = None
        
        mask = np.isnan(X)
        
        # Initial imputation with mean
        X_filled = X. copy()
        col_means = np.nanmean(X, axis=0)
        for j in range(X.shape[1]):
            X_filled[mask[: , j], j] = col_means[j]
        
        # Iterate
        for iteration in range(self. max_iter):
            X_old = X_filled.copy()
            
            # For each column with missing values
            for j in range(X.shape[1]):
                if not np.any(mask[:, j]):
                    continue
                
                # Rows with observed values in column j
                obs_idx = ~mask[:, j]
                miss_idx = mask[:, j]
                
                # Features (all other columns)
                features = [i for i in range(X.shape[1]) if i != j]
                
                if len(features) == 0:
                    continue
                
                X_train = X_filled[obs_idx][: , features]
                y_train = X_filled[obs_idx, j]
                X_test = X_filled[miss_idx][:, features]
                
                # Train Random Forest
                rf = RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    random_state=self.random_state,
                    n_jobs=-1,
                    max_depth=10
                )
                rf.fit(X_train, y_train)
                
                # Predict
                y_pred = rf.predict(X_test)
                X_filled[miss_idx, j] = y_pred
            
            # Check convergence
            change = np.sum((X_filled - X_old) ** 2) / np.sum(X_old ** 2)
            
            if (iteration + 1) % 2 == 0:
                logging.info(f"missForest iteration {iteration + 1}/{self.max_iter}, change: {change:.6f}")
            
            if change < 1e-4:
                logging.info(f"missForest converged at iteration {iteration + 1}")
                break
        
        if columns is not None:
            return pd.DataFrame(X_filled, columns=columns, index=index)
        return X_filled


class XGBoostImputer:
    """
    XGBoost-based iterative imputation.
    """
    def __init__(self, max_iter=10, n_estimators=100, random_state=42):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed. Install with: pip install xgboost")
        
        self.max_iter = max_iter
        self. n_estimators = n_estimators
        self.random_state = random_state
    
    def fit_transform(self, X):
        """
        Impute missing values using XGBoost. 
        
        Args:
            X: numpy array or DataFrame with NaN for missing values
        
        Returns:
            Imputed matrix
        """
        if isinstance(X, pd.DataFrame):
            columns = X.columns
            index = X.index
            X = X.values
        else:
            columns = None
            index = None
        
        mask = np.isnan(X)
        
        # Initial imputation with mean
        X_filled = X.copy()
        col_means = np.nanmean(X, axis=0)
        for j in range(X.shape[1]):
            X_filled[mask[:, j], j] = col_means[j]
        
        # Iterate
        for iteration in range(self.max_iter):
            X_old = X_filled.copy()
            
            # For each column with missing values
            for j in range(X.shape[1]):
                if not np. any(mask[:, j]):
                    continue
                
                obs_idx = ~mask[:, j]
                miss_idx = mask[:, j]
                
                features = [i for i in range(X.shape[1]) if i != j]
                
                if len(features) == 0:
                    continue
                
                X_train = X_filled[obs_idx][:, features]
                y_train = X_filled[obs_idx, j]
                X_test = X_filled[miss_idx][:, features]
                
                # Train XGBoost
                model = xgb.XGBRegressor(
                    n_estimators=self.n_estimators,
                    random_state=self.random_state,
                    max_depth=6,
                    learning_rate=0.1,
                    n_jobs=-1,
                    verbosity=0
                )
                model.fit(X_train, y_train)
                
                # Predict
                y_pred = model.predict(X_test)
                X_filled[miss_idx, j] = y_pred
            
            # Check convergence
            change = np.sum((X_filled - X_old) ** 2) / np.sum(X_old ** 2)
            
            if (iteration + 1) % 2 == 0:
                logging. info(f"XGBoost iteration {iteration + 1}/{self.max_iter}, change: {change:.6f}")
            
            if change < 1e-4:
                logging.info(f"XGBoost converged at iteration {iteration + 1}")
                break
        
        if columns is not None:
            return pd.DataFrame(X_filled, columns=columns, index=index)
        return X_filled


class LightGBMImputer:
    """
    LightGBM-based iterative imputation.
    """
    def __init__(self, max_iter=10, n_estimators=100, random_state=42):
        # If LightGBM not available, we'll fall back to sklearn RandomForest
        self.use_sklearn_fallback = not LIGHTGBM_AVAILABLE
        if self.use_sklearn_fallback:
            logging.warning("LightGBM not available; LightGBMImputer will fall back to RandomForestRegressor.")

        self.max_iter = max_iter
        self.n_estimators = n_estimators
        self.random_state = random_state
    
    def fit_transform(self, X):
        """
        Impute missing values using LightGBM.
        
        Args:
            X: numpy array or DataFrame with NaN for missing values
        
        Returns:
            Imputed matrix
        """
        if isinstance(X, pd. DataFrame):
            columns = X. columns
            index = X.index
            X = X.values
        else:
            columns = None
            index = None
        
        mask = np.isnan(X)
        
        # Initial imputation with mean
        X_filled = X.copy()
        col_means = np.nanmean(X, axis=0)
        for j in range(X.shape[1]):
            X_filled[mask[:, j], j] = col_means[j]
        
        # Iterate
        for iteration in range(self.max_iter):
            X_old = X_filled.copy()
            
            # For each column with missing values
            for j in range(X.shape[1]):
                if not np.any(mask[:, j]):
                    continue
                
                obs_idx = ~mask[:, j]
                miss_idx = mask[: , j]
                
                features = [i for i in range(X.shape[1]) if i != j]
                
                if len(features) == 0:
                    continue
                
                X_train = X_filled[obs_idx][:, features]
                y_train = X_filled[obs_idx, j]
                X_test = X_filled[miss_idx][:, features]
                
                # Train model: LightGBM if available, otherwise RandomForest fallback
                if not self.use_sklearn_fallback:
                    model = lgb.LGBMRegressor(
                        n_estimators=self.n_estimators,
                        random_state=self.random_state,
                        max_depth=6,
                        learning_rate=0.1,
                        n_jobs=-1,
                        verbosity=-1
                    )
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                else:
                    from sklearn.ensemble import RandomForestRegressor
                    model = RandomForestRegressor(
                        n_estimators=self.n_estimators,
                        random_state=self.random_state,
                        n_jobs=-1,
                        max_depth=10
                    )
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                X_filled[miss_idx, j] = y_pred
            
            # Check convergence
            change = np.sum((X_filled - X_old) ** 2) / np.sum(X_old ** 2)
            
            if (iteration + 1) % 2 == 0:
                logging.info(f"LightGBM iteration {iteration + 1}/{self. max_iter}, change: {change:.6f}")
            
            if change < 1e-4:
                logging.info(f"LightGBM converged at iteration {iteration + 1}")
                break
        
        if columns is not None:
            return pd.DataFrame(X_filled, columns=columns, index=index)
        return X_filled


# ============================================================================
# BAYESIAN METHODS
# ============================================================================

class GaussianProcessImputer: 
    """
    Gaussian Process-based imputation for spatial data.
    """
    def __init__(self, kernel=None, random_state=42, max_train_size=1000,
                 prediction_batch_size=2000):
        if not GP_AVAILABLE:
            raise ImportError("Gaussian Process not available. Check scikit-learn installation.")
        
        self.random_state = random_state
        self.max_train_size = max(1, int(max_train_size))
        self.prediction_batch_size = max(1, int(prediction_batch_size))
        
        if kernel is None:
            # Default kernel: RBF + White noise
            self.kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
        else:
            self.kernel = kernel
    
    def fit_transform(self, X):
        """
        Impute missing values using Gaussian Process. 
        
        Args:
            X: numpy array or DataFrame with NaN for missing values
        
        Returns:
            Imputed matrix
        """
        if isinstance(X, pd.DataFrame):
            columns = X.columns
            index = X.index
            X = X.values
        else:
            columns = None
            index = None
        
        mask = np.isnan(X)
        X_filled = X.copy()
        
        # For each column with missing values
        for j in range(X.shape[1]):
            if not np. any(mask[:, j]):
                continue
            
            obs_idx = ~mask[:, j]
            miss_idx = mask[: , j]
            
            # Use row indices as spatial coordinates
            X_train = np.arange(X.shape[0])[obs_idx].reshape(-1, 1)
            y_train = X[obs_idx, j]
            X_test = np.arange(X.shape[0])[miss_idx].reshape(-1, 1)

            # Exact Gaussian processes allocate an O(n^2) covariance matrix
            # and perform O(n^3) fitting. Regional inputs contain well over
            # 100,000 observations, which previously caused SLURM OOM kills.
            # Retain an evenly distributed, deterministic training subset and
            # predict missing rows in bounded batches.
            if len(X_train) > self.max_train_size:
                keep = np.linspace(
                    0, len(X_train) - 1, self.max_train_size, dtype=int
                )
                X_train = X_train[keep]
                y_train = y_train[keep]
            
            # Train Gaussian Process
            gp = GaussianProcessRegressor(
                kernel=self.kernel,
                random_state=self.random_state,
                optimizer=None,
                normalize_y=True
            )
            
            gp.fit(X_train, y_train)
            
            # Predict
            prediction_parts = []
            sigma_parts = []
            for start in range(0, len(X_test), self.prediction_batch_size):
                stop = start + self.prediction_batch_size
                batch_prediction, batch_sigma = gp.predict(
                    X_test[start:stop], return_std=True
                )
                prediction_parts.append(batch_prediction)
                sigma_parts.append(batch_sigma)
            y_pred = np.concatenate(prediction_parts)
            sigma = np.concatenate(sigma_parts)
            X_filled[miss_idx, j] = y_pred
            
            logging.info(f"GP imputed column {j}, mean uncertainty: {np.mean(sigma):.4f}")
        
        if columns is not None:
            return pd.DataFrame(X_filled, columns=columns, index=index)
        return X_filled


# ============================================================================
# SPATIAL NEURAL NETWORK METHOD (SPIN)
# ============================================================================

if TORCH_AVAILABLE:
    class SPINModel(nn.Module):
        """
        Spatial Imputation Network (SPIN)
        Uses spatial relationships between sites for imputation. 
        """
        def __init__(self, n_features, n_spatial_features, hidden_dim=64):
            super(SPINModel, self).__init__()

            self.n_features = n_features
            self.n_spatial_features = n_spatial_features

            # Spatial encoding
            self.spatial_encoder = nn.Sequential(
                nn.Linear(n_spatial_features, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )

            # Feature encoding
            self.feature_encoder = nn.Sequential(
                nn.Linear(n_features + 1, hidden_dim),  # +1 for mask
                nn.ReLU(),
                nn.Dropout(0.2)
            )

            # Combined decoder
            self.decoder = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, n_features)
            )

        def forward(self, x, mask, spatial_features):
            """
            Args:
                x: Target site features (batch, n_features)
                mask: Missing mask (batch, n_features)
                spatial_features: Features from nearby sites (batch, n_spatial_features)
            """
            # Encode spatial information
            spatial_encoded = self.spatial_encoder(spatial_features)

            # Encode target features with mask
            mask_float = mask.float().unsqueeze(-1)
            x_masked = torch.cat([x, mask_float.squeeze(-1)], dim=-1)
            feature_encoded = self.feature_encoder(x_masked)

            # Combine and decode
            combined = torch.cat([spatial_encoded, feature_encoded], dim=-1)
            output = self.decoder(combined)

            return output


    class SPIN:
        """
        Spatial Imputation Network wrapper.
        """
        def __init__(self, max_iter=100, hidden_dim=64, random_state=42):
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch is not installed. Install with: pip install torch")

            self.max_iter = max_iter
            self.hidden_dim = hidden_dim
            self.random_state = random_state

        def fit_transform(self, X, spatial_features=None):
            """
            Impute missing values using SPIN.

            Args:
                X: numpy array with NaN for missing values
                spatial_features: numpy array with spatial information from other sites

            Returns:
                Imputed matrix
            """
            torch.manual_seed(self.random_state)

            mask = ~np.isnan(X)

            # Fill missing with mean
            X_filled = X.copy()
            col_means = np.nanmean(X, axis=0)
            for j in range(X.shape[1]):
                X_filled[np.isnan(X[:, j]), j] = col_means[j]

            # If no spatial features provided, use zero padding
            if spatial_features is None:
                spatial_features = np.zeros((X.shape[0], 10))

            # Standardize
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_filled)

            # Convert to tensors
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            X_tensor = torch.FloatTensor(X_scaled).to(device)
            mask_tensor = torch.FloatTensor(mask).to(device)
            spatial_tensor = torch.FloatTensor(spatial_features).to(device)

            # Initialize model
            model = SPINModel(
                n_features=X.shape[1],
                n_spatial_features=spatial_features.shape[1],
                hidden_dim=self.hidden_dim
            ).to(device)

            # Training
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()

            model.train()
            for epoch in range(self.max_iter):
                optimizer.zero_grad()

                output = model(X_tensor, mask_tensor, spatial_tensor)

                # Loss only on observed values
                loss = criterion(output * mask_tensor, X_tensor * mask_tensor)

                loss.backward()
                optimizer.step()

                if (epoch + 1) % 20 == 0:
                    logging.info(f"SPIN Epoch {epoch + 1}/{self.max_iter}, Loss: {loss.item():.6f}")

            # Inference
            model.eval()
            with torch.no_grad():
                imputed = model(X_tensor, mask_tensor, spatial_tensor)
                imputed = imputed.cpu().numpy()

            # Inverse transform
            imputed = scaler.inverse_transform(imputed)

            return imputed
else:
    # Provide a sklearn-based fallback SPIN that does not require PyTorch
    class SPIN:
        """
        Fallback SPIN using IterativeImputer when PyTorch is not available.
        This provides a reasonable imputation alternative without GPU dependency.
        """
        def __init__(self, max_iter=10, hidden_dim=64, random_state=42):
            self.max_iter = max_iter
            self.hidden_dim = hidden_dim
            self.random_state = random_state
            from sklearn.experimental import enable_iterative_imputer  # noqa: F401
            from sklearn.impute import IterativeImputer
            self._imputer = IterativeImputer(random_state=self.random_state, max_iter=self.max_iter)

        def fit_transform(self, X, spatial_features=None):
            # If spatial features provided, concatenate them as additional columns
            if spatial_features is not None:
                # Ensure shapes align on rows
                if spatial_features.shape[0] != X.shape[0]:
                    # Fall back to ignoring spatial features
                    spatial_features = None

            if spatial_features is not None:
                X_cat = np.concatenate([X, np.nan_to_num(spatial_features, nan=0.0)], axis=1)
            else:
                X_cat = X

            imputed = self._imputer.fit_transform(X_cat)

            # Return only original columns
            return imputed[:, :X.shape[1]]


# ============================================================================
# UNIFIED IMPUTATION INTERFACE
# ============================================================================

def impute_with_method(data, target_column, input_columns, method='softimpute', 
                      custom_strategies=None, spatial_config=None, **kwargs):
    """
    Unified imputation interface for all methods.
    
    Args:
        data: Input DataFrame
        target_column: Target column to impute
        input_columns:  List of input columns
        method:  Imputation method name
        custom_strategies: Custom imputation strategies
        spatial_config: Spatial-temporal configuration
        **kwargs: Additional method-specific parameters
    
    Returns:
        Imputed DataFrame
    """
    logging.info(f"Starting {method. upper()} imputation...")
    
    # ✅ ADD THIS SAFETY CHECK:
    if spatial_config and spatial_config.get('use_spatial', False):
        # Check if target is in input columns (from spatial features)
        spatial_cols = [col for col in data.columns if col.startswith('spatial_')]
        target_spatial = [col for col in spatial_cols if target_column in col]

        if target_spatial:
            logging.error(
                f"⚠️  DATA LEAKAGE DETECTED!\n"
                f"Found spatial features of target:  {target_spatial[:3]}...\n"
                f"R² will be artificially inflated to ~1.0\n"
                f"FIX: Set USE_SPATIAL_FEATURES = False in config_spatial.py"
            )
            logging.warning("Auto-removing spatial target features...")
            data = data.drop(columns=target_spatial)

    # Prepare data
    if target_column in input_columns:
        input_columns = [col for col in input_columns if col != target_column]
    
    # Apply spatial-temporal features if configured
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
    
    # Apply custom strategies
    if custom_strategies:
        for col, strategy in custom_strategies.items():
            if col in data_to_use.columns:
                if strategy == "mean":
                    data_to_use[col] = data_to_use[col].fillna(data_to_use[col].mean())
                elif strategy == "median":
                    data_to_use[col] = data_to_use[col]. fillna(data_to_use[col].median())
                elif isinstance(strategy, (int, float)):
                    data_to_use[col] = data_to_use[col].fillna(strategy)

    # Optional: Pre-impute input columns using MissForest to fill input missing values
    try:
        import config_spatial as config
        if getattr(config, 'APPLY_MISSFOREST_PREIMPUTE', False):
            input_cols_prefill = [col for col in input_columns_to_use if col in data_to_use.columns]
            if input_cols_prefill:
                sub = data_to_use[input_cols_prefill]
                if sub.isna().any().any():
                    logging.info("Applying MissForest pre-imputation to input columns...")
                    mf = MissForest(max_iter=getattr(config, 'MISSFOREST_PREIMPUTE_MAX_ITER', 5),
                                    n_estimators=getattr(config, 'MISSFOREST_PREIMPUTE_N_ESTIMATORS', 100),
                                    random_state=kwargs.get('random_state', 42))
                    imputed_sub = mf.fit_transform(sub)
                    if isinstance(imputed_sub, pd.DataFrame):
                        data_to_use[input_cols_prefill] = imputed_sub
                    else:
                        data_to_use[input_cols_prefill] = pd.DataFrame(imputed_sub, columns=input_cols_prefill, index=data_to_use.index)
                    logging.info("MissForest pre-imputation completed")
    except Exception:
        # Fail silently to avoid breaking pipelines if config or MissForest unavailable
        logging.debug("MissForest pre-imputation skipped (config or implementation missing)")
    
    # Prepare columns for imputation
    columns_for_imputation = input_columns_to_use + [target_column]
    columns_for_imputation = [col for col in columns_for_imputation if col in data_to_use.columns]
    data_for_imputation = data_to_use[columns_for_imputation]. copy()
    
    # Select and apply method
    method = method.lower()
    
    if method == 'softimpute':
        imputer = SoftImputeCustom(max_iter=kwargs.get('max_iter', 100))
        imputed_values = imputer.fit_transform(data_for_imputation. values)
    
    elif method == 'optspace':
        imputer = OptSpace(rank=kwargs.get('rank', 10), max_iter=kwargs.get('max_iter', 100))
        imputed_values = imputer. fit_transform(data_for_imputation.values)
    
    elif method == 'svt':
        imputer = SVT(max_iter=kwargs.get('max_iter', 500))
        imputed_values = imputer.fit_transform(data_for_imputation.values)
    
    elif method == 'missforest':
        imputer = MissForest(
            max_iter=kwargs.get('max_iter', 10),
            n_estimators=kwargs.get('n_estimators', 100),
            random_state=kwargs.get('random_state', 42)
        )
        imputed_values = imputer.fit_transform(data_for_imputation)
        if isinstance(imputed_values, pd. DataFrame):
            imputed_values = imputed_values.values
    
    elif method == 'xgboost':
        imputer = XGBoostImputer(
            max_iter=kwargs.get('max_iter', 10),
            n_estimators=kwargs.get('n_estimators', 100),
            random_state=kwargs.get('random_state', 42)
        )
        imputed_values = imputer.fit_transform(data_for_imputation)
        if isinstance(imputed_values, pd.DataFrame):
            imputed_values = imputed_values.values
    
    elif method == 'lightgbm':
        imputer = LightGBMImputer(
            max_iter=kwargs.get('max_iter', 10),
            n_estimators=kwargs.get('n_estimators', 100),
            random_state=kwargs. get('random_state', 42)
        )
        imputed_values = imputer.fit_transform(data_for_imputation)
        if isinstance(imputed_values, pd.DataFrame):
            imputed_values = imputed_values.values
    
    elif method == 'gp' or method == 'gaussian_process':
        imputer = GaussianProcessImputer(random_state=kwargs.get('random_state', 42))
        imputed_values = imputer. fit_transform(data_for_imputation)
        if isinstance(imputed_values, pd.DataFrame):
            imputed_values = imputed_values.values

    elif method == 'mice' or method == 'iterative':
        # MICE: use sklearn's IterativeImputer (default estimator: BayesianRidge)
        try:
            from sklearn.linear_model import BayesianRidge
            estimator = kwargs.get('estimator', BayesianRidge())
        except Exception:
            estimator = None

        imputer_kwargs = {
            'random_state': kwargs.get('random_state', 42),
            'max_iter': kwargs.get('max_iter', 10)
        }
        if estimator is not None:
            imputer = IterativeImputer(estimator=estimator, **imputer_kwargs)
        else:
            imputer = IterativeImputer(**imputer_kwargs)

        imputed_values = imputer.fit_transform(data_for_imputation.values)
        if isinstance(imputed_values, pd.DataFrame):
            imputed_values = imputed_values.values
    
    elif method == 'spin':
        # Extract spatial features for SPIN
        spatial_cols = [col for col in columns_for_imputation if col.startswith('spatial_')]
        if spatial_cols:
            spatial_features = data_for_imputation[spatial_cols].fillna(0).values
        else:
            spatial_features = None
        
        imputer = SPIN(
            max_iter=kwargs.get('max_iter', 100),
            hidden_dim=kwargs. get('hidden_dim', 64),
            random_state=kwargs.get('random_state', 42)
        )
        imputed_values = imputer.fit_transform(data_for_imputation. values, spatial_features)
    
    else:
        raise ValueError(f"Unknown imputation method: {method}")
    
    # Create output DataFrame
    imputed_df = pd.DataFrame(
        imputed_values,
        columns=columns_for_imputation,
        index=data_for_imputation.index
    )
    
    # Merge with original data
    final_df = data. copy()
    for col in columns_for_imputation:
        if col in data.columns:
            final_df[col] = imputed_df[col]
    
    logging.info(f"{method.upper()} imputation completed successfully.")
    return final_df


# ============================================================================
# MODEL WRAPPERS FOR EACH METHOD (Compatible with main.py)
# ============================================================================

# Each method gets its own file-like interface

# SoftImpute
MODEL_NAME = "SoftImpute"

def impute_mice(data, target_column, input_columns, max_iter=100, random_state=42, 
                tol=0.01, custom_strategies=None, spatial_config=None):
    """SoftImpute imputation compatible with main.py"""
    return impute_with_method(
        data, target_column, input_columns, 
        method='softimpute',
        custom_strategies=custom_strategies,
        spatial_config=spatial_config,
        max_iter=max_iter
    )


# Example usage for creating separate model files: 
# You can copy this template for each method

def create_model_file(method_name, display_name):
    """
    Template for creating individual model files.
    Copy this to create:  SoftImpute. py, OptSpace.py, SVT.py, etc.
    """
    template = f'''
from impute import impute_with_method
import logging

logging.basicConfig(level=logging. INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MODEL_NAME = "{display_name}"

def impute_mice(data, target_column, input_columns, max_iter=100, random_state=42, 
                tol=0.01, custom_strategies=None, spatial_config=None):
    """
    {display_name} imputation compatible with main.py
    
    Args:
        data: Input DataFrame
        target_column: Target column to impute
        input_columns: List of input columns
        max_iter: Maximum iterations
        random_state: Random seed
        tol: Tolerance (not used in all methods)
        custom_strategies:  Custom imputation strategies
        spatial_config: Spatial-temporal configuration
    
    Returns:
        Imputed DataFrame
    """
    return impute_with_method(
        data, target_column, input_columns,
        method='{method_name}',
        custom_strategies=custom_strategies,
        spatial_config=spatial_config,
        max_iter=max_iter,
        random_state=random_state
    )
'''
    return template


if __name__ == "__main__": 
    # Example:  Create individual model files
    methods = {
        'softimpute': 'SoftImpute',
        'optspace': 'OptSpace',
        'svt': 'SVT',
        'missforest': 'MissForest',
        'xgboost': 'XGBoost',
        'lightgbm': 'LightGBM',
        'gp': 'GaussianProcess',
        'spin': 'SPIN'
    }
    
    print("Creating individual model files...")
    for method_key, method_name in methods.items():
        filename = f"{method_name}.py"
        content = create_model_file(method_key, method_name)
        
        with open(filename, 'w') as f:
            f.write(content)
        
        print(f"✅ Created {filename}")
    
    print("\n✨ All model files created successfully!")
    print("\nAdd these to config_spatial. py MODELS_TO_RUN:")
    print(f"MODELS_TO_RUN = {list(methods.values())}")
