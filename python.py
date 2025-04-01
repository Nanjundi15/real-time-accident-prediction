import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ====== Load the Preprocessed Dataset ======
# Use the correct file path
file_path = r"D:\cyber security\major project\real time\Road_Accident_Data_Preprocessed (1).csv"

# Check if the file exists before loading
if os.path.exists(file_path):
    accident_df = pd.read_csv(file_path)
    print("File loaded successfully!")
else:
    print(f"File not found: {file_path}")
    exit()

# ====== Display Column Names to Identify Issues ======
print("Columns in the dataset:")
print(accident_df.columns)

# ====== Strip Spaces from Column Names (if any) ======
accident_df.columns = accident_df.columns.str.strip()

# ====== Convert DateTime Column to Datetime Format ======
# Ensure correct date format (DD-MM-YYYY HH:MM)
try:
    accident_df['Datetime'] = pd.to_datetime(
        accident_df['Accident Date'] + ' ' + accident_df['Time'],
        format="%d-%m-%Y %H:%M",  # Match the format in your dataset
        errors='coerce'  # Invalid dates become NaT
    )
    print("Datetime column created successfully!")
except Exception as e:
    print(f"Error parsing dates: {e}")
    exit()

# ====== Set Plot Style ======
sns.set_style("whitegrid")

# ====== 1. Accident Severity vs. Weather Conditions ======
plt.figure(figsize=(12, 6))

# Check and use correct column names
if 'Weather_Conditions_Num' in accident_df.columns and 'Accident_Severity_Num' in accident_df.columns:
    severity_weather = accident_df.groupby(['Weather_Conditions_Num', 'Accident_Severity_Num']).size().unstack(fill_value=0)

    # Plotting
    severity_weather.plot(kind='bar', stacked=True, colormap='viridis', figsize=(14, 7))
    plt.title('Accident Severity by Weather Conditions')
    plt.xlabel('Weather Conditions (0=Clear, 1=Rain, 2=High Winds, 3=Snow, 4=Fog, 5=Other, 6=Unknown)')
    plt.ylabel('Number of Accidents')
    plt.legend(title='Severity (3=Fatal, 2=Serious, 1=Slight)')
    plt.show()
else:
    print("Weather or Severity columns not found!")

# ====== 2. Scatter Plot of Accident Locations ======
plt.figure(figsize=(12, 8))

# Check if the location columns exist
if 'Longitude' in accident_df.columns and 'Latitude' in accident_df.columns:
    sns.scatterplot(
        x='Longitude', y='Latitude', hue='Accident_Severity_Num', size='Speed_limit',
        data=accident_df, palette='coolwarm', sizes=(20, 200), alpha=0.7
    )
    plt.title('Scatter Plot of Accident Locations')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend(title='Severity')
    plt.show()
else:
    print("Longitude or Latitude columns not found!")

# ====== 3. Accident Frequency by Time ======
plt.figure(figsize=(14, 7))

# Check if the Datetime column exists
if 'Datetime' in accident_df.columns:
    accident_df['Datetime'].dt.date.value_counts().sort_index().plot(kind='line', color='blue', marker='o')
    plt.title('Accident Frequency Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Accidents')
    plt.grid(True)
    plt.show()
else:
    print("Datetime column not found!")

