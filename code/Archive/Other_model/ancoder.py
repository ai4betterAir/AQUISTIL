import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

# Define the path
folder_path = '/mnt/scratch_lustre/barthelx/Masrur/Projects/Data_imputation/Input_Data_with_Missing/Run1_data'
output_folder = '/mnt/scratch_lustre/barthelx/Masrur/Projects/Data_imputation/Imputed_Data'
os.makedirs(output_folder, exist_ok=True)

# Define the model architecture
def create_autoencoder(input_dim, encoding_dim=16, optimizer='adam'):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(64, activation='relu')(input_layer)
    encoded = Dense(32, activation='relu')(encoded)
    bottleneck = Dense(encoding_dim, activation='relu')(encoded)
    decoded = Dense(32, activation='relu')(bottleneck)
    decoded = Dense(64, activation='relu')(decoded)
    output_layer = Dense(input_dim, activation='sigmoid')(decoded)

    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer=optimizer, loss='mse')
    return autoencoder

# Hyperparameter grid for GridSearchCV
param_grid = {
    'encoding_dim': [16, 32, 64],
    'optimizer': ['adam', 'rmsprop'],
    'batch_size': [64, 128],
    'epochs': [50, 100]
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

        # Normalize the data
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df)

        # Apply GridSearchCV to find the best hyperparameters for Autoencoder
        input_dim = df_scaled.shape[1]
        model = KerasRegressor(build_fn=create_autoencoder, input_dim=input_dim)

        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1)
        grid_search.fit(df_scaled, df_scaled)  # Autoencoders are trained to reconstruct the input

        # Use the best model for imputation
        best_model = grid_search.best_estimator_.model
        df_imputed = best_model.predict(df_scaled)

        # Reverse normalization
        df_imputed = scaler.inverse_transform(df_imputed)

        # Replace missing PM2.5 with imputed values
        df['PM2.5'] = np.where(pm25_col.isna(), df_imputed[:, df.columns.get_loc('PM2.5')], pm25_col)

        # Combine the datetime, predicted PM2.5, missing_info, and removed_values
        result_df = pd.DataFrame({
            'datetime': datetime_col,
            'Predicted_PM2.5': df['PM2.5'],
            'missing_info': missing_info,
            'removed_values': removed_values
        })

        # Save the results with model name in the file name
        model_name = 'Autoencoder'
        output_file_name = f"{os.path.basename(folder_path)}_{model_name}_{file_name}"
        result_df.to_csv(os.path.join(output_folder, output_file_name), index=False)

        print(f"Imputation complete for {file_name}. Results saved as {output_file_name}.")

print("All files processed successfully.")
