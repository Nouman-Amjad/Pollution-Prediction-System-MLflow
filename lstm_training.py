import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from itertools import product
import mlflow
import mlflow.keras
from mlflow.models.signature import infer_signature

# Suppress TensorFlow debug logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load processed data
data = np.load("Data/processed_data.npz")
X_train_seq = data['X_train_seq']
y_train_seq = data['y_train_seq']
X_test_seq = data['X_test_seq']
y_test_seq = data['y_test_seq']

# Set or create an MLflow experiment
experiment_name = "LSTM Pollution Prediction"
mlflow.set_experiment(experiment_name)

# Hyperparameter grid
n_units = [50, 100]
dropout_rates = [0.2, 0.3]
batch_sizes = [32, 64]
epochs = [20, 30]

# Initialize best score tracking
best_rmse = float("inf")
best_params = None

# Perform grid search for hyperparameter tuning
for units, dropout, batch_size, epoch in product(n_units, dropout_rates, batch_sizes, epochs):
    with mlflow.start_run():
        # Build the LSTM model
        model = Sequential([
            LSTM(units, activation='relu', input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
            Dropout(dropout),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        # Train the model
        model.fit(X_train_seq, y_train_seq, validation_data=(X_test_seq, y_test_seq), epochs=epoch, batch_size=batch_size)

        # Make predictions and calculate metrics
        y_pred = model.predict(X_test_seq)
        rmse = np.sqrt(mean_squared_error(y_test_seq, y_pred))
        mae = mean_absolute_error(y_test_seq, y_pred)
        r2 = r2_score(y_test_seq, y_pred)

        # Log parameters and metrics
        mlflow.log_param("n_units", units)
        mlflow.log_param("dropout_rate", dropout)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epoch)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2_Score", r2)

        # Log the model
        signature = infer_signature(X_train_seq, model.predict(X_train_seq))
        mlflow.keras.log_model(model, "lstm_model", signature=signature)

        # Update the best model if RMSE improves
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = {"n_units": units, "dropout_rate": dropout, "batch_size": batch_size, "epochs": epoch}
            # Save the best model locally
            os.makedirs("models", exist_ok=True)
            model.save("models/best_lstm_model.h5")

        print(f"Trained with params: Units={units}, Dropout={dropout}, Batch Size={batch_size}, Epochs={epoch}")
        print(f"Metrics: RMSE={rmse}, MAE={mae}, R2_Score={r2}")

# Print the best parameters and score
print(f"Best RMSE: {best_rmse} with parameters {best_params}")
