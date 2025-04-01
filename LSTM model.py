# ====== Import Libraries ======
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

# ====== Load Dataset ======
file_path = r"D:\cyber security\major project\real time\Road_Accident_Data_Reduced.csv"
df = pd.read_csv(file_path)

# ====== Preprocessing ======
# 1. Select numeric columns
X_numeric = df.select_dtypes(include=['float64', 'int64'])

# 2. Handle Categorical Columns with Label Encoding
cat_cols = df.select_dtypes(include=['object']).columns

if len(cat_cols) > 0:
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))  # Encode categorical values into integers

# 3. Combine Numeric + Encoded Categorical Features
X_final = df.drop('Accident_Severity_Num', axis=1)
y = df['Accident_Severity_Num']

# 4. Scale the Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_final)

# Save the scaler for future use
scaler_path = r"D:\cyber security\major project\real time\lstm_scaler.pkl"
joblib.dump(scaler, scaler_path)

# ====== Prepare LSTM Input ======
# LSTM expects 3D input: (samples, timesteps, features)
timesteps = 5  # Define how many previous steps to consider
num_features = X_scaled.shape[1]

# Create sequences for LSTM input
def create_sequences(X, y, timesteps):
    Xs, ys = [], []
    for i in range(len(X) - timesteps):
        Xs.append(X[i:i + timesteps])
        ys.append(y[i + timesteps])
    return np.array(Xs), np.array(ys)

# Prepare LSTM sequences
X_lstm, y_lstm = create_sequences(X_scaled, y, timesteps)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)

# ====== Build LSTM Model ======
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(128, activation='relu', return_sequences=True, input_shape=(timesteps, num_features)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(len(np.unique(y)), activation='softmax')  # Multi-class classification
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# ====== Train Model ======
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# ====== Save Model ======
model_path = r"D:\cyber security\major project\real time\lstm_model.h5"
model.save(model_path)

# ====== Evaluate Model ======
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# ====== Display Training History ======
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
plt.title('LSTM Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss', color='red')
plt.plot(history.history['val_loss'], label='Validation Loss', color='purple')
plt.title('LSTM Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
