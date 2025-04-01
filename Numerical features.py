# ====== Import Libraries ======
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# ====== Load Dataset ======
file_path = r"D:\cyber security\major project\real time\Road_Accident_Data_Reduced.csv"  # PyCharm path
df = pd.read_csv(file_path)

# ✅ Select only numerical features
X_numerical = df.select_dtypes(include=['number', 'float64', 'int64'])

# ✅ Re-train the scaler on numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numerical)

# ✅ Save the new scaler
new_scaler_filepath = r"D:\cyber security\major project\real time\scaler_logistic.pkl"
joblib.dump(scaler, new_scaler_filepath)

print(f"✅ Numerical features scaled successfully and saved as '{new_scaler_filepath}'!")
