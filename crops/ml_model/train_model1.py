import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np

file_path = "/Users/bharath.yakkala/Library/CloudStorage/OneDrive-DAZN/Desktop/farmerfriend/farmerfriend/crops/DataSets/crop_production.csv" 

# Define batch size
chunk_size = 1000

# Step 1: Collect unique category values (Ensures consistent encoding)
unique_categories = {col: set() for col in ["State_Name", "District_Name", "Season", "Crop"]}

for chunk in pd.read_csv(file_path, chunksize=chunk_size):
    for col in unique_categories:
        unique_categories[col].update(chunk[col].dropna().unique())  # Collect unique values

# Initialize LabelEncoders with a fixed category list
label_encoders = {col: LabelEncoder() for col in unique_categories}
for col in unique_categories:
    label_encoders[col].fit(list(unique_categories[col]))  # Set fixed categories

# Step 2: Initialize Model for Incremental Learning
model = RandomForestClassifier(n_estimators=100, random_state=42)

first_batch = True

# Step 3: Train the model in batches
for chunk in pd.read_csv(file_path, chunksize=chunk_size):
    print(f"Processing chunk of size {len(chunk)}")

    # Drop missing values
    chunk.dropna(inplace=True)

    # Encode categorical columns using fixed LabelEncoders
    for col in ["State_Name", "District_Name", "Season", "Crop"]:
        chunk[col] = label_encoders[col].transform(chunk[col])

    # Features and Target Variable
    X = chunk[["State_Name", "District_Name", "Season", "Area"]]
    y = chunk["Crop"]

    # Train-Test Split (10% test data for evaluation)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Ensure consistent number of classes by providing `classes_` manually
    if first_batch:
        model.fit(X_train, y_train)
        first_batch = False
    else:
        model.fit(X_train, y_train)

    # Evaluate on test batch
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Batch Accuracy: {accuracy * 100:.2f}%")

# Step 4: Save Model & Label Encoders
joblib.dump(model, 'crop_predict_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')

print("Model training complete! Saved model & encoders.")
