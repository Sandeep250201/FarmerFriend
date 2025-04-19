import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv('DataSets/Crop_recommendation.csv')

# Encode the target variable
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])  # Convert crop names to numeric labels

# Feature selection (Exclude the 'label' column from X)
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the trained model and label encoder
joblib.dump(model, 'crop_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl') 
