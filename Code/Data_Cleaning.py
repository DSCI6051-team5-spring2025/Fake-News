import pandas as pd 
# Load the dataset
df = pd.read_csv(r"C:\Users\rajar\OneDrive\Desktop\New folder\Fake-News\politifact_final_dataset.csv")

# Normalize Rating values (convert to lowercase, remove spaces)
df["Rating"] = df["Rating"].str.strip().str.lower()

# Debug: Print the cleaned unique values
print("Unique Rating Values After Cleaning:")
print(df["Rating"].unique())

# Correcting the True/False mapping
true_labels = ["true", "half-true", "mostly-true"]
false_labels = ["false", "pants-fire", "barely-true", "full-flop"]

# Apply correct mapping
df["Label"] = df["Rating"].apply(lambda x: "True" if x in true_labels else "False")

# Debug: Check label distribution
print("Final Label Distribution:")
print(df["Label"].value_counts())

# Drop the 'Source' column since it's not needed
df_cleaned = df.drop(columns=["Source"], errors="ignore")

# Remove rows with missing values or "N/A" entries
df_cleaned = df_cleaned.replace("N/A", pd.NA).dropna()

# Save the cleaned dataset
cleaned_file_path = "politifact_cleaned_dataset.csv"
df_cleaned.to_csv(cleaned_file_path, index=False)


# Return the path of the cleaned file
cleaned_file_path
