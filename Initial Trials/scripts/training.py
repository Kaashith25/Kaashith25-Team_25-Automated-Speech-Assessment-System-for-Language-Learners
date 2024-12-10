import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
dataset_csv = r"D:\NLP_project\svarah_with_cefr_features.csv"
data = pd.read_csv(dataset_csv)

# Check and print all column names
print("Columns in dataset:", data.columns)

# Use 'average_cefr' as proficiency level (adjust according to your needs)
fluency_column = "average_cefr"  # Assuming 'average_cefr' is used for fluency scores
proficiency_column = "average_cefr"  # Assuming 'average_cefr' is used for proficiency levels

# Features and target variables
features = data.drop(columns=[fluency_column, 'audio_file', 'transcription'])
targets_regression = data[fluency_column]  # Fluency scores (based on 'average_cefr')
targets_classification = data[fluency_column]  # Proficiency levels (based on 'average_cefr')

# Split into train, validation, and test sets
X_train, X_temp, y_train_reg, y_temp_reg = train_test_split(features, targets_regression, test_size=0.2, random_state=42)
X_val, X_test, y_val_reg, y_test_reg = train_test_split(X_temp, y_temp_reg, test_size=0.5, random_state=42)

y_train_class, y_temp_class = train_test_split(targets_classification, test_size=0.2, random_state=42)
y_val_class, y_test_class = train_test_split(y_temp_class, test_size=0.5, random_state=42)

# Standardize the features (optional)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Save preprocessed data for model training
pd.DataFrame(X_train_scaled).to_csv("X_train.csv", index=False)
pd.DataFrame(X_val_scaled).to_csv("X_val.csv", index=False)
pd.DataFrame(X_test_scaled).to_csv("X_test.csv", index=False)

pd.DataFrame(y_train_reg).to_csv("y_train_reg.csv", index=False)
pd.DataFrame(y_val_reg).to_csv("y_val_reg.csv", index=False)
pd.DataFrame(y_test_reg).to_csv("y_test_reg.csv", index=False)

pd.DataFrame(y_train_class).to_csv("y_train_class.csv", index=False)
pd.DataFrame(y_val_class).to_csv("y_val_class.csv", index=False)
pd.DataFrame(y_test_class).to_csv("y_test_class.csv", index=False)

print("Dataset prepared and saved!")
