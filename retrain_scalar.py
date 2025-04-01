# ====== Import Libraries ======
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# ✅ Load the dataset
file_path = r"D:\cyber security\major project\real time\Road_Accident_Data_Preprocessed (1).csv"
df = pd.read_csv(file_path)

# ✅ Separate features (X) and target (y)
y = df['Accident_Severity_Num']  # Target variable
X = df.drop('Accident_Severity_Num', axis=1)  # Features only

# ✅ Select only numerical features
X_numerical = X.select_dtypes(include=['number', 'float64', 'int64'])

# ✅ Re-train the scaler on numerical features only
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numerical)

# ✅ Save the new scaler
new_scaler_filepath = r"D:\cyber security\major project\real time\new_scaler_logistic.pkl"
joblib.dump(scaler, new_scaler_filepath)

print(f"✅ New scaler saved successfully at: {new_scaler_filepath}")
