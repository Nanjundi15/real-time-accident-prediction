# ====== Import Libraries ======
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# ====== Load Dataset ======
# Corrected Pycharm file path
file_path = r"D:\cyber security\major project\real time\Road_Accident_Data_Reduced.csv"
df = pd.read_csv(file_path)

# ====== Separate Numeric and Non-Numeric Columns ======
# Identify numeric and non-numeric columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
non_numeric_cols = df.select_dtypes(exclude=['int64', 'float64']).columns

# ====== Apply StandardScaler only on numeric columns ======
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[numeric_cols])

# Convert scaled features back to DataFrame with correct column names
scaled_df = pd.DataFrame(scaled_features, columns=numeric_cols)

# **Combine both scaled numeric and non-numeric columns**
# This ensures the original column order is preserved
final_df = pd.concat([scaled_df, df[non_numeric_cols].reset_index(drop=True)], axis=1)

# ✅ Save the scaler to a file
scaler_path = r"D:\cyber security\major project\real time\scaler.pkl"
joblib.dump(scaler, scaler_path)

print(f"✅ Scaler saved successfully at: {scaler_path}")
print("\n🔹 Final DataFrame Shape:", final_df.shape)
print("\n🔹 Sample Data:")
print(final_df.head())
