# ====== Import Libraries ======
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ✅ Load the Dataset
file_path = r"D:\cyber security\major project\real time\Road_Accident_Data_Reduced.csv"
df = pd.read_csv(file_path)

# ✅ Prepare Features and Target
y = df['Accident_Severity_Num']

# ✅ Load the Scaler
scaler_path = r"D:\cyber security\major project\real time\new_scaler_logistic.pkl"
scaler = joblib.load(scaler_path)

# ✅ Match the Feature Names
# Extract the original features used during scaler fitting
original_features = scaler.feature_names_in_

# Select only the matching columns
X = df[original_features]

# ✅ Scale the matched features
X_scaled = scaler.transform(X)

# ✅ Split the Data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ✅ Retrain the XGBoost Model
xgboost_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgboost_model.fit(X_train, y_train)

# ✅ Save the retrained model
xgboost_output_path = r"D:\cyber security\major project\real time\xgboost_model_fixed.pkl"
joblib.dump(xgboost_model, xgboost_output_path)
print(f"\n✅ XGBoost model retrained and saved as '{xgboost_output_path}'!")
