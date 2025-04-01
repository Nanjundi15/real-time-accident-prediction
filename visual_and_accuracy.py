# ====== Import Libraries ======
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ====== Load the Dataset ======
file_path = r"D:\cyber security\major project\real time\Road_Accident_Data_Reduced.csv"  # Update with your file path
data = pd.read_csv(file_path)

# ====== Set Seaborn Theme ======
sns.set_theme(style="whitegrid")

# 🔥 Downsample data for faster visualization (random 5,000 samples)
sample_data = data.sample(n=5000, random_state=42) if len(data) > 5000 else data

# 🎯 Accident Severity Distribution
plt.figure(figsize=(12, 6))
sns.countplot(data=sample_data, x='Accident_Severity_Num', palette='coolwarm')
plt.title('Accident Severity Distribution', fontsize=18, fontweight='bold')
plt.xlabel('Accident Severity', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.tight_layout()   # Ensures proper spacing in PyCharm
plt.show()

# 🚦 Weather Conditions Impact
plt.figure(figsize=(12, 6))
sns.boxplot(x='Weather_Conditions_Num', y='Accident_Severity_Num', data=sample_data, palette='muted')
plt.title('Weather Conditions vs Severity', fontsize=18, fontweight='bold')
plt.xlabel('Weather Conditions', fontsize=14)
plt.ylabel('Severity', fontsize=14)
plt.tight_layout()
plt.show()

# 🚗 Vehicle Type Distribution
plt.figure(figsize=(12, 6))
sns.countplot(data=sample_data, x='Vehicle_Type', palette='viridis')
plt.title('Vehicle Type Distribution', fontsize=18, fontweight='bold')
plt.xlabel('Vehicle Type', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 📈 Correlation Heatmap (limited to numeric columns for faster execution)
plt.figure(figsize=(12, 8))
corr = sample_data.select_dtypes(include=['float64', 'int64']).corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=1, linecolor='white')
plt.title('Correlation Heatmap', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.show()
