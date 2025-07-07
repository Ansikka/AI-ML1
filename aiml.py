import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Simulated dataset with advanced feature
np.random.seed(42)
n_samples = 500

df = pd.DataFrame({
    'area': np.random.randint(600, 4500, n_samples),
    'bedrooms': np.random.randint(1, 6, n_samples),
    'bathrooms': np.random.randint(1, 4, n_samples),
    'stories': np.random.randint(1, 4, n_samples),
    'parking': np.random.randint(0, 4, n_samples),
    'age': np.random.randint(0, 50, n_samples),
    'mainroad': np.random.choice([0, 1], n_samples),
    'furnishing_status': np.random.choice([0, 1, 2], n_samples),
    'location': np.random.choice(['Urban', 'Suburban', 'Rural'], n_samples),
    'basement': np.random.choice([0, 1], n_samples),
    'air_conditioning': np.random.choice([0, 1], n_samples),
    'preferred_area_score': np.random.randint(1, 6, n_samples)
})

# Price formula simulation for realism
df['price'] = (
    df['area'] * 70 +
    df['bedrooms'] * 50000 +
    df['bathrooms'] * 30000 +
    df['parking'] * 20000 +
    df['preferred_area_score'] * 25000 +
    df['air_conditioning'] * 40000 +
    df['basement'] * 35000 +
    df['mainroad'] * 20000 +
    np.random.normal(0, 50000, n_samples)
)

# Define features and target
X = df.drop('price', axis=1)
y = df['price']


numeric_features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking', 'age', 'preferred_area_score']
categorical_features = ['location', 'furnishing_status', 'mainroad', 'basement', 'air_conditioning']

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first')

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Model pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=150, random_state=42))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

r2, mae, rmse
