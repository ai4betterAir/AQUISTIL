import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional, Masking
from tensorflow.keras.callbacks import EarlyStopping

# Define the path
folder_path = '/mnt/scratch_lustre/barthelx/Masrur/Projects/Data_imputation/Input_Data_with_Missing/Run1_data'
output_directory_base = '/mnt/scratch_lustre/barthelx/Masrur/Projects/Data_imputation/0Imputed_Data'
method = 'BRITS'  # Model name for BRITS

# Extract run_number from the folder_path (assuming it's the first word, separated by underscores)
run_number = os.path.basename(folder_path).split('_')[0]

# Ensure the output directory exists
os.makedirs(output_directory_base, exist_ok=True)

# Function to create BRITS model
def create_brits(input_dim, time_steps, encoding_dim=16, optimizer='adam'):
    input_layer = Input(shape=(time_steps, input_dim))

    # Masking for handling missing values
    masked = Masking(mask_value=0.0)(input_layer)

    # Bidirectional LSTM layers
    encoded = Bidirectional(LSTM(64, return_sequences=True))(masked)
    encoded = Bidirectional(LSTM(32, return_sequences=True))(encoded)

    # Dense layer for imputation
    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(32, activation='relu')(decoded)
    output_layer = Dense(input_dim, activation='sigmoid')(decoded)

    brits = Model(inputs=input_layer, outputs=output_layer)
    brits.compile(optimizer=optimizer, loss='mse')
    return brits

# Function to create sequences for time series forecasting
def create_sequences(data, timesteps):
    sequences = []
    for i in range(len(data) - timesteps):
        sequences.append(data[i:i+timesteps])
    return np.array(sequences)

# Parameters for sequence creation
timesteps = 10  # Number of timesteps to look back

# Iterate through CSV files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        file_path = os.path.join(folder_path, file_name)

        # Load the CSV file
        df = pd.read_csv(file_path)

        if 'PM2.5' not in df.columns:
            print(f"File {file_name} does not have 'PM2.5' column. Skipping.")
            continue

        # Extract study site from the file name (second word of the CSV file name)
        study_site = file_name.split('_')[1]  # Extracting the second word as study site

        # Extract relevant columns
        datetime_col = df['datetime']
        pm25_col_original = df['PM2.5']  # Original PM2.5
        missing_info = df['missing_info'] if 'missing_info' in df.columns else np.nan
        removed_values = df['removed_values'] if 'removed_values' in df.columns else np.nan

        # Keep 'missing_info' and 'removed_values' in the DataFrame but not in the imputation process
        non_imputation_cols = ['PM2.5', 'missing_info', 'removed_values', 'datetime']

        # Normalize the data (excluding 'datetime', 'PM2.5', 'missing_info', and 'removed_values')
        numeric_df = df.select_dtypes(include=[np.number]).drop(columns=non_imputation_cols, errors='ignore')

        # Ensure we are working with some numeric data
        if numeric_df.empty:
            print(f"No numeric columns to process in {file_name}. Skipping.")
            continue

        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(numeric_df)

        # Prepare the data for BRITS model (convert to sequences)
        input_dim = df_scaled.shape[1]  # Number of features
        X = create_sequences(df_scaled, timesteps)
        y = pm25_col_original[timesteps:].fillna(0).values  # Replace NaNs for the model's training

        # Reshape y to match the batch size
        y = y.reshape(-1, 1)

        # Build and train the BRITS model
        model = create_brits(input_dim=input_dim, time_steps=timesteps)

        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

        # Train the model
        model.fit(X, X, epochs=20, batch_size=32, verbose=1, callbacks=[early_stopping])

        # Now use the model to predict the missing values
        imputed_values = model.predict(X).reshape(-1, input_dim)

        # Add the predicted values back to the DataFrame
        df['PM2.5(original)'] = pm25_col_original

        # Ensure that the length of imputed_values matches the length of the slice timesteps: in the DataFrame
        imputed_values_length = len(imputed_values)
        rows_to_fill = len(df) - timesteps

        # If imputed_values has more rows than can be placed, adjust the slicing accordingly
        if imputed_values_length > rows_to_fill:
            imputed_values = imputed_values[:rows_to_fill, :]

        # Add the imputed PM2.5 values (assuming the first column corresponds to PM2.5)
        df['PM2.5(only_imputed_data)'] = np.nan
        df.loc[timesteps:timesteps + imputed_values_length, 'PM2.5(only_imputed_data)'] = imputed_values[:, 0]

        # Replace original non-missing values in 'PM2.5(only_imputed_data)' with NaN
        df.loc[~df['PM2.5'].isna(), 'PM2.5(only_imputed_data)'] = np.nan

        # Merge the original and imputed values into 'full PM2.5 (after imputation)'
        df['full_PM2.5(after_imputation)'] = df['PM2.5(original)'].combine_first(df['PM2.5(only_imputed_data)'])

        # Define the output directory and create it if it doesn't exist
        output_directory = os.path.join(output_directory_base, method)
        os.makedirs(output_directory, exist_ok=True)

        # Save the final result with the file name format '{run_number}_{study_site}_{Model_name}.csv'
        output_file_name = f"{run_number}_{study_site}_{method}.csv"
        output_file_path = os.path.join(output_directory, output_file_name)

        # Save the final DataFrame
        output_df = df[['datetime', 'missing_info', 'removed_values', 'PM2.5(original)', 'PM2.5(only_imputed_data)', 'full_PM2.5(after_imputation)']]
        output_df.to_csv(output_file_path, index=False)

        print(f"Processed {file_name} and saved results to {output_file_path}")

print("All files processed and saved successfully.")