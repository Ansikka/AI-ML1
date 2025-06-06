import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Step 1: Create dataset
data = {
    'area': [1000, 1500, 1800, 2400, 3000, 3500, 4000],
    'bedrooms': [2, 3, 3, 4, 4, 5, 5],
    'bathrooms': [1, 2, 2, 2, 3, 3, 4],
    'price': [150000, 200000, 220000, 280000, 350000, 400000, 450000]
}

df = pd.DataFrame(data)

# Step 2: Prepare features and labels
X = df[['area', 'bedrooms', 'bathrooms']]
y = df['price']

# Step 3: Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Evaluate model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Step 6: Make prediction
test_house_df = pd.DataFrame([[2500, 3, 2]], columns=['area', 'bedrooms', 'bathrooms'])
predicted_price = model.predict(test_house_df)
print("Predicted Price for 2500 sqft, 3BHK, 2 Bath:", predicted_price[0])
