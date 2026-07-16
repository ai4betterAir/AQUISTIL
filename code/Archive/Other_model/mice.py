import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from fancyimpute import IterativeImputer

# Define the path
folder_path = '/mnt/scratch_lustre/barthelx/Masrur/Projects/Data_imputation/Input_Data_with_Missing/Run1_data'
output_folder = '/mnt/scratch_lustre/barthelx/Masrur/Projects/Data_imputation/Imputed_Data'
os.makedirs(output_folder, exist_ok=True)

# Define a custom wrapper to integrate MICE (IterativeImputer) with scikit-learn's GridSearchCV
class MICEWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, max_iter=10, random_state=None):
        self.max_iter = max_iter
        self.random_state = random_state
    
    def fit(self, X, y=None):
        self.model = IterativeImputer(max_iter=self.max_iter, random_state=self.random_state)
        self.model.fit(X)
        return self
    
    def transform(self, X):
        return self.model.transform(X)

# Hyperparameter grid for GridSearchCV
param_grid = {
    'max_iter': [5, 10, 20],
    'random_state': [0, 42]
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

        # Apply GridSearchCV to find the best hyperparameters for MICE
        mice_imputer = MICEWrapper()
        grid_search = GridSearchCV(mice_imputer, param_grid, cv=3, n_jobs=-1)
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

        # Save the results as 'folder_name_file_name.csv'
        output_file_name = f"{os.path.basename(folder_path)}_{file_name}"
        result_df.to_csv(os.path.join(output_folder, output_file_name), index=False)

        print(f"Imputation complete for {file_name}. Results saved as {output_file_name}.")

print("All files processed successfully.")
