import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from fancyimpute import SoftImpute
from sklearn.base import BaseEstimator, TransformerMixin

# Define a custom wrapper to integrate SoftImpute with scikit-learn's GridSearchCV
class SoftImputeWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, shrinkage_value=0.1, max_iters=100, max_rank=None):
        self.shrinkage_value = shrinkage_value
        self.max_iters = max_iters
        self.max_rank = max_rank
    
    def fit(self, X, y=None):
        self.model = SoftImpute(shrinkage_value=self.shrinkage_value,
                                max_iters=self.max_iters,
                                max_rank=self.max_rank)
        self.model.fit(X)
        return self
    
    def transform(self, X):
        return self.model.transform(X)

# Define the path
folder_path = '/mnt/scratch_lustre/barthelx/Masrur/Projects/Data_imputation/Input_Data_with_Missing/Run1_data'
output_folder = '/mnt/scratch_lustre/barthelx/Masrur/Projects/Data_imputation/Imputed_Data'
os.makedirs(output_folder, exist_ok=True)

# Hyperparameter grid for GridSearchCV
param_grid = {
    'shrinkage_value': [0.1, 0.5, 1.0],
    'max_iters': [50, 100, 200],
    'max_rank': [None, 10, 20]
}

# Iterate through CSV files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path)

        # Extract relevant columns
        datetime_col = df['datetime']
        pm25_col = df['PM2.5']
        missing_info = df['missing_info']
        removed_values = df['removed_values']

        # Drop 'missing_info' and 'removed_values' columns for modeling
        df.drop(columns=['missing_info', 'removed_values'], inplace=True)

        # Select only numeric columns for scaling and imputation
        numeric_df = df.select_dtypes(include=[np.number])

        # Normalize the numeric data
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(numeric_df)

        # Apply GridSearchCV to find the best hyperparameters for SoftImpute
        soft_impute = SoftImputeWrapper()
        grid_search = GridSearchCV(soft_impute, param_grid, cv=3, n_jobs=-1)
        grid_search.fit(df_scaled)

        # Use the best model for imputation
        best_model = grid_search.best_estimator_
        df_imputed = best_model.transform(df_scaled)

        # Reverse normalization
        df_imputed = scaler.inverse_transform(df_imputed)

        # Replace the missing values in the original dataframe with the imputed values
        df.loc[:, numeric_df.columns] = df_imputed

        # Combine the datetime, predicted PM2.5, missing_info, and removed_values
        result_df = pd.DataFrame({
            'datetime': datetime_col,
            'Predicted_PM2.5': df['PM2.5'],
            'missing_info': missing_info,
            'removed_values': removed_values
        })

        # Save the results with model name in the file name
        model_name = 'SoftImpute'
        output_file_name = f"{os.path.basename(folder_path)}_{model_name}_{file_name}"
        result_df.to_csv(os.path.join(output_folder, output_file_name), index=False)

        print(f"Imputation complete for {file_name}. Results saved as {output_file_name}.")

print("All files processed successfully.")
