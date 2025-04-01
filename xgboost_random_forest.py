# ====== Import Libraries ======
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

# ====== Load Preprocessed Dataset ======
file_path = r"D:\cyber security\major project\real time\Road_Accident_Data_Reduced.csv"  # Your PyCharm local path
df = pd.read_csv(file_path)
print("✅ File loaded successfully!")

# ====== Data Preprocessing ======
# Drop irrelevant columns if present
if 'Accident_Index' in df.columns:
    df.drop(['Accident_Index'], axis=1, inplace=True)

# ====== Encode Categorical Columns Using Label Encoding (Memory Efficient) ======
label_encoder = LabelEncoder()

# Apply label encoding to categorical columns
for col in df.select_dtypes(include='object').columns:
    df[col] = label_encoder.fit_transform(df[col])

print("✅ Label encoding applied!")

# ====== Split Features and Target ======
X = df.drop(['Accident_Severity'], axis=1)  # Features
y = df['Accident_Severity']                 # Target

# ====== Split into Train and Test Sets ======
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ====== Train Random Forest Model ======
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate Random Forest
rf_y_pred = rf_model.predict(X_test)
print("\n✅ Random Forest Performance:")
print(f"Accuracy: {accuracy_score(y_test, rf_y_pred)}")
print(classification_report(y_test, rf_y_pred))

# ====== Train XGBoost Model ======
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# Evaluate XGBoost
xgb_y_pred = xgb_model.predict(X_test)
print("\n✅ XGBoost Performance:")
print(f"Accuracy: {accuracy_score(y_test, xgb_y_pred)}")
print(classification_report(y_test, xgb_y_pred))

# ====== Save Both Models Locally in PyCharm ======
joblib.dump(rf_model, r"D:\cyber security\major project\real time\random_forest_model.pkl")  # Save Random Forest
joblib.dump(xgb_model, r"D:\cyber security\major project\real time\xgboost_model.pkl")        # Save XGBoost

print("\n✅ Both models saved successfully in PyCharm!")
