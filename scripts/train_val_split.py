import pandas as pd
from sklearn.model_selection import train_test_split

# Load your CSV
df = pd.read_csv("../data/LUNA25_Public_Training_Development_Data.csv")

# Get unique patient IDs
unique_patients = df['PatientID'].unique()

# Split patient IDs into train and validation (e.g., 80-20 split)
train_patients, val_patients = train_test_split(
    unique_patients, test_size=0.2, random_state=42
)

# Create train and validation sets by filtering on PatientID
train_df = df[df['PatientID'].isin(train_patients)].reset_index(drop=True)
val_df = df[df['PatientID'].isin(val_patients)].reset_index(drop=True)

train_df.to_csv("../data/train.csv", index=False)
val_df.to_csv("../data/valid.csv", index=False)
