# ====== Import Libraries ======
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns

# ====== Load Models and Scaler ======
model_paths = {
    "Logistic Regression": r"D:\cyber security\major project\real time\logistic_model.pkl",
    "Decision Tree": r"D:\cyber security\major project\real time\decision_tree_model.pkl",
    "Random Forest": r"D:\cyber security\major project\real time\random_forest_model.pkl",
    "XGBoost": r"D:\cyber security\major project\real time\xgboost_model_fixed.pkl",
    "LSTM": r"D:\cyber security\major project\real time\lstm_model.h5"
}

scaler_path = r"D:\cyber security\major project\real time\new_scaler_logistic.pkl"
scaler = joblib.load(scaler_path)

# ====== Load and Preprocess Dataset ======
file_path = r"D:\cyber security\major project\real time\Road_Accident_Data_Reduced.csv"
df = pd.read_csv(file_path)

# Use only the features that the scaler was trained on
original_features = scaler.feature_names_in_

# Scale the dataset
X = df[original_features]
X_scaled = scaler.transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=original_features)

# Target labels
y = df['Accident_Severity_Num']

# ====== Model Integration and Predictions ======
predictions = {}


# ✅ Fixed align_features function
def align_features(model, X_df, model_name):
    """Align dataset features with model features."""

    if model_name == "XGBoost":
        # Use only the specific features used during XGBoost training
        xgboost_features = [
            'Day_of_Week', 'Light_Conditions', 'Weather_Conditions',
            'Road_Type', 'Vehicle_Type', 'Accident_Severity_Num'
        ]

        # Filter the dataset for XGBoost
        X_df = X_df[xgboost_features]

    elif hasattr(model, "feature_names_in_"):
        # Get model's expected features
        model_features = model.feature_names_in_
        missing_features = set(model_features) - set(X_df.columns)

        # Add missing features with zeros
        for feature in missing_features:
            X_df[feature] = 0

        # Ensure consistent feature order
        X_df = X_df[model_features]

    return X_df


# Iterate through models except LSTM
for model_name, model_path in model_paths.items():
    if model_name != "LSTM":
        print(f"\n🔥 Loading: {model_name}")

        # Load the model
        model = joblib.load(model_path)

        # Align features
        try:
            X_test_matched = align_features(model, X_scaled_df, model_name)
            y_pred = model.predict(X_test_matched)
            predictions[model_name] = y_pred
        except Exception as e:
            print(f"❌ Error predicting with {model_name}: {e}")
            predictions[model_name] = None

# ====== LSTM Model Predictions ======
print("\n🔥 Loading: LSTM")

# Get the LSTM model input shape
lstm_model = load_model(model_paths['LSTM'])
lstm_input_shape = lstm_model.input_shape[2]  # Extract input feature count

# Handle LSTM feature mismatch by padding or reducing
if X_scaled.shape[1] < lstm_input_shape:
    # Add padding with zeros
    padding = np.zeros((X_scaled.shape[0], lstm_input_shape - X_scaled.shape[1]))
    X_lstm = np.hstack((X_scaled, padding))
elif X_scaled.shape[1] > lstm_input_shape:
    # Reduce to match LSTM shape
    X_lstm = X_scaled[:, :lstm_input_shape]
else:
    X_lstm = X_scaled

# Reshape for LSTM (3D: [samples, timesteps, features])
X_lstm = np.expand_dims(X_lstm, axis=1)

# Predict with LSTM
lstm_y_pred = np.argmax(lstm_model.predict(X_lstm), axis=1)
predictions["LSTM"] = lstm_y_pred

# ====== Display the Predictions ======
print("\n✅ Model Predictions:")
for model, preds in predictions.items():
    print(f"\n{model} Predictions:")
    print(preds)

# ====== Save Predictions ======
output_path = r"D:\cyber security\major project\real time\model_predictions.csv"
prediction_df = pd.DataFrame(predictions)
prediction_df.to_csv(output_path, index=False)
print(f"\n✅ Predictions saved to: {output_path}")

# ====== Graphical Visualization ======

# Convert predictions into a DataFrame for visualization
visual_df = pd.DataFrame(predictions)

# Add ground truth values to the DataFrame
visual_df['Ground Truth'] = y.values

# Plot the predictions
plt.figure(figsize=(16, 8))
sns.set(style="whitegrid")

# Line plot for comparison
for model_name in predictions.keys():
    plt.plot(visual_df[model_name][:100], label=f"{model_name}", alpha=0.7, linewidth=1.5)

plt.plot(visual_df['Ground Truth'][:100], label="Ground Truth", color='black', linestyle='dashed', linewidth=2)

# Title and Labels
plt.title("Model Predictions vs. Ground Truth", fontsize=18)
plt.xlabel("Sample Index", fontsize=14)
plt.ylabel("Accident Severity", fontsize=14)
plt.legend()
plt.grid(True)
plt.show()

# ====== Heatmap for Predictions ======
plt.figure(figsize=(12, 6))
sns.heatmap(visual_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=1, linecolor='black')
plt.title("Correlation Heatmap of Model Predictions", fontsize=16)
plt.show()

