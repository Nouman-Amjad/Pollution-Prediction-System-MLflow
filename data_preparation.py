import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load the combined dataset
data = pd.read_csv("Data/environmental_data.csv")

# Check for missing values and handle them
# Separate numeric and non-numeric columns
numeric_columns = data.select_dtypes(include=[np.number]).columns
non_numeric_columns = data.select_dtypes(exclude=[np.number]).columns

# Fill missing values for numeric columns with the mean
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

# For non-numeric columns, fill missing values with a placeholder or drop them
data[non_numeric_columns] = data[non_numeric_columns].fillna("Unknown")

# Select features and target variable
features = ['Temperature', 'Humidity', 'Wind Speed', 'PM10', 'Ozone']
target = 'PM2.5'

X = data[features]
y = data[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape data for LSTM
def create_sequences(X, y, time_steps):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X.iloc[i:(i + time_steps)].values)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 5
X_train_seq, y_train_seq = create_sequences(X_train, y_train, time_steps)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, time_steps)

# Save processed data for training
np.savez("Data/processed_data.npz", X_train_seq=X_train_seq, y_train_seq=y_train_seq, X_test_seq=X_test_seq, y_test_seq=y_test_seq)
print("Data preparation completed.")
