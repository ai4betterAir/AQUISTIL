import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, LeakyReLU, BatchNormalization, Dropout
from sklearn.preprocessing import StandardScaler

# Define the path
folder_path = '/mnt/scratch_lustre/barthelx/Masrur/Projects/Data_imputation/Input_Data_with_Missing/Run1_data'
output_folder = '/mnt/scratch_lustre/barthelx/Masrur/Projects/Data_imputation/Imputed_Data'
os.makedirs(output_folder, exist_ok=True)

# Define GAN architecture
def build_generator(input_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(input_dim, activation='sigmoid'))
    return model

def build_discriminator(input_dim):
    model = Sequential()
    model.add(Dense(512, input_dim=input_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model

# GAN Class
class GAN:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.generator = build_generator(input_dim)
        self.discriminator = build_discriminator(input_dim)
        self.discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Combined model
        noise = Input(shape=(input_dim,))
        generated_data = self.generator(noise)
        self.discriminator.trainable = False
        validity = self.discriminator(generated_data)
        self.combined = Model(noise, validity)
        self.combined.compile(optimizer='adam', loss='binary_crossentropy')

    def train(self, X_train, epochs=1000, batch_size=32):
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            real_data = X_train[idx]

            noise = np.random.normal(0, 1, (batch_size, self.input_dim))
            generated_data = self.generator.predict(noise)

            d_loss_real = self.discriminator.train_on_batch(real_data, valid)
            d_loss_fake = self.discriminator.train_on_batch(generated_data, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            noise = np.random.normal(0, 1, (batch_size, self.input_dim))
            g_loss = self.combined.train_on_batch(noise, valid)

            if epoch % 100 == 0:
                print(f"{epoch} [D loss: {d_loss[0]}, acc.: {100*d_loss[1]}] [G loss: {g_loss}]")

# Data Preprocessing
def preprocess_data(df):
    # Select only numeric columns for scaling
    numeric_df = df.select_dtypes(include=[np.number])
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(numeric_df)
    return df_scaled, scaler, numeric_df.columns

# Impute missing values
def impute_missing_values(df_imputed, original_df, columns):
    for col in columns:
        original_col = original_df[col]
        imputed_col = df_imputed[col]
        original_df[col] = np.where(original_col.isna(), imputed_col, original_col)
    return original_df

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

        # Preprocess data
        df_scaled, scaler, numeric_columns = preprocess_data(df)

        # Build and train GAN
        gan = GAN(input_dim=df_scaled.shape[1])
        gan.train(X_train=df_scaled, epochs=5000, batch_size=32)

        # Generate data with the trained generator
        generated_data = gan.generator.predict(df_scaled)

        # Reverse scaling to get back to original data range
        generated_data = scaler.inverse_transform(generated_data)

        # Create a DataFrame from the imputed data
        df_imputed = pd.DataFrame(generated_data, columns=numeric_columns)

        # Impute missing values back into the original DataFrame
        df_final = impute_missing_values(df_imputed, df, numeric_columns)

        # Combine the datetime, predicted PM2.5, missing_info, removed_values
        result_df = pd.DataFrame({
            'datetime': datetime_col,
            'Predicted_PM2.5': df_final['PM2.5'],
            'missing_info': missing_info,
            'removed_values': removed_values
        })

        # Save the results with model name in the file name
        model_name = 'GAN_Imputation'
        output_file_name = f"{os.path.basename(folder_path)}_{model_name}_{file_name}"
        result_df.to_csv(os.path.join(output_folder, output_file_name), index=False)

        print(f"Imputation complete for {file_name}. Results saved as {output_file_name}.")

print("All files processed successfully.")