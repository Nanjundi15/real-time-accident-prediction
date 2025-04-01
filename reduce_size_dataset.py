import pandas as pd

# Load the dataset in chunks
chunk_size = 100_000
file_path = r"D:\cyber security\major project\real time\Road_Accident_Data_Preprocessed (1).csv"

# Read only the first chunk (or sample more chunks)
chunk = next(pd.read_csv(file_path, chunksize=chunk_size))

# Save the smaller chunk
reduced_file = r"D:\cyber security\major project\real time\Road_Accident_Data_Reduced.csv"
chunk.to_csv(reduced_file, index=False)

print(f"✅ Reduced dataset saved at: {reduced_file}")
