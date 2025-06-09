import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

# Step 1: Expanded Dataset with more features
data = {
    'area': [1500, 1800, 2400, 3000, 3500],
    'bedrooms': [2, 3, 3, 4, 4],
    'bathrooms': [1, 2, 2, 3, 3],
    'stories': [1, 2, 2, 3, 2],
    'parking': [1, 1, 2, 2, 3],
    'age': [5, 10, 8, 3, 15],
    'mainroad': [1, 0, 1, 1, 0],
    'furnishing_status': [0, 1, 1, 2, 0],  # 0: Unfurnished, 1: Semi, 2: Furnished
    'location': ['Urban', 'Suburban', 'Urban', 'Urban', 'Rural'],
    'basement': [1, 0, 1, 1, 0],
    'air_conditioning': [1, 1, 0, 1, 0],
    'preferred_area_score': [5, 4, 4, 5, 3],
    'price': [200000, 250000, 300000, 400000, 350000]
}

df = pd.DataFrame(data)

# Step 2: Features and labels
X = df.drop('price', axis=1)
y = df['price']

# Step 3: Preprocessing
numeric_features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking', 'age', 'preferred_area_score']
categorical_features = ['location', 'furnishing_status', 'mainroad', 'basement', 'air_conditioning']

# Pipelines for preprocessing
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first')

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Step 4: Model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Step 5: Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train model
model.fit(X_train, y_train)

# Step 7: Predict and evaluate
y_pred = model.predict(X_test)
print("R² Score on Test Data:", r2_score(y_test, y_pred))

# Step 8: Cross-validation score
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print("Cross-Validated R² Score:", round(np.mean(cv_scores), 2))

# Step 9: Predict new house
new_house = pd.DataFrame([{
    'area': 2500,
    'bedrooms': 3,
    'bathrooms': 2,
    'stories': 2,
    'parking': 2,
    'age': 7,
    'mainroad': 1,
    'furnishing_status': 1,
    'location': 'Urban',
    'basement': 1,
    'air_conditioning': 1,
    'preferred_area_score': 5
}])

predicted_price = model.predict(new_house)
print("Predicted Price for New House:", round(predicted_price[0], 2))
