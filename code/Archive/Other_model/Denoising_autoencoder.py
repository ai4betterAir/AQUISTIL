import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.model_selection import cross_val_score
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
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

# Define the search space for Bayesian Optimization
search_space = {
    'encoding_dim': Integer(16, 64),
    'optimizer': Categorical(['adam', 'rmsprop']),
    'batch_size': Integer(64, 256),
    'epochs': Integer(50, 150)
}

# Function to wrap the Keras model for scikit-learn compatibility
def build_keras_regressor(encoding_dim, optimizer):
    return KerasRegressor(build_fn=create_autoencoder, input_dim=input_dim, encoding_dim=encoding_dim, optimizer=optimizer, epochs=100, batch_size=32, verbose=0)

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
        numeric_df = df.select_dtypes(include=[np.number])
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(numeric_df)

        # Add noise to input data (denoising autoencoder training)
        noise_factor = 0.2
        df_noisy = df_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=df_scaled.shape)
        df_noisy = np.clip(df_noisy, 0., 1.)

        # Apply Bayesian Optimization to find the best hyperparameters
        input_dim = df_scaled.shape[1]
        model = build_keras_regressor(encoding_dim=32, optimizer='adam')

        opt = BayesSearchCV(estimator=model, search_spaces=search_space, n_iter=10, cv=3, n_jobs=-1)
        opt_result = opt.fit(df_noisy, df_scaled)

        # Use the best model for imputation
        best_model = opt_result.best_estimator_.model
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

        # Save the results as 'folder_name_file_name.csv'
        model_name = 'DenoisingAutoencoder'
        output_file_name = f"{os.path.basename(folder_path)}_{model_name}_{file_name}"
        result_df.to_csv(os.path.join(output_folder, output_file_name), index=False)

        print(f"Imputation complete for {file_name}. Results saved as {output_file_name}.")

print("All files processed successfully.")