import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Step 1: Create the dataset
data = {
    'area': [1500, 1800, 2400, 3000, 3500],
    'bedrooms': [2, 3, 3, 4, 4],
    'bathrooms': [1, 2, 2, 3, 3],
    'stories': [1, 2, 2, 3, 2],
    'parking': [1, 1, 2, 2, 3],
    'age': [5, 10, 8, 3, 15],
    'mainroad': [1, 0, 1, 1, 0],
    'furnishing_status': [0, 1, 1, 2, 0],
    'price': [200000, 250000, 300000, 400000, 350000]
}

df = pd.DataFrame(data)

# Step 2: Features and labels
X = df.drop('price', axis=1)
y = df['price']

# Step 3: Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Predict & evaluate
y_pred = model.predict(X_test)
print("R² Score on Test Data:", r2_score(y_test, y_pred))

# Step 6: Predict new house price
new_house = pd.DataFrame([[2500, 3, 2, 2, 2, 7, 1, 1]], 
                         columns=['area', 'bedrooms', 'bathrooms', 'stories', 'parking', 'age', 'mainroad', 'furnishing_status'])
predicted_price = model.predict(new_house)
print("Predicted Price for new house:", round(predicted_price[0], 2))
