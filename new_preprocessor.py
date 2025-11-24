import joblib
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

print("Creating new preprocessor...")

# Create a new preprocessor
numeric_features = ['LB', 'LT', 'KT', 'KM']
categorical_features = ['Provinsi', 'Kota/Kab', 'Type']

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

print("Created preprocessor successfully!")
print("\nSaving preprocessor...")
joblib.dump(preprocessor, 'models/trained/barupreprocessor.pkl')
print("Preprocessor saved successfully!")