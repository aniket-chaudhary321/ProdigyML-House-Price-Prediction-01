import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np


import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv('kc_house_data.csv')



     

# Explore the dataset (optional)
print(df.head())

print(df.info())


# Selecting features (X) and target variable (y)
features = ['sqft_living', 'bedrooms', 'bathrooms']
target = 'price'

X = df[features]
y = df[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize the linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Calculate and print the mean squared error
mse = mean_squared_error(y_test, predictions)
print('Mean Squared Error:', mse)



# Example: Making predictions on new data
new_data = pd.DataFrame({'sqft_living': [1500], 'bedrooms': [3], 'bathrooms': [2]})
predicted_price = model.predict(new_data)
print('Predicted Price:', predicted_price[0])


# Example: Visualizing predictions vs. actual values based on the selected features
plt.figure(figsize=(18, 5))

# Visualize histograms for square footage
plt.subplot(1, 3, 1)
plt.hist(X_test['sqft_living'], bins=30, color='skyblue', alpha=0.7, label='Square Footage')
plt.xlabel('Square Footage')
plt.ylabel('Frequency')
plt.title('Square Footage Distribution')
plt.legend()


# Visualize histograms for number of bedrooms
plt.subplot(1, 3, 2)
plt.hist(X_test['bedrooms'], bins=10, color='salmon', alpha=0.7, label='Bedrooms')
plt.xlabel('Number of Bedrooms')
plt.ylabel('Frequency')
plt.title('Number of Bedrooms Distribution')
plt.legend()
     
     
     
# Visualize histograms for number of bathrooms
plt.subplot(1, 3, 3)
plt.hist(X_test['bathrooms'], bins=10, color='green', alpha=0.7, label='Bathrooms')
plt.xlabel('Number of Bathrooms')
plt.ylabel('Frequency')
plt.title('Number of Bathrooms Distribution')
plt.legend()


#3d plot
fig=plt.figure(figsize=(20,13))
ax=fig.add_subplot(2,2,1,projection='3d')
ax.scatter(df['floors'],df['bedrooms'],df['bathrooms'],c='darkgreen',alpha=0.5)
ax.set(xlabel='\nfloors',ylabel='\nbedrooms',zlabel='\nbathrooms')
ax.set(ylim=[0,12])

ax=fig.add_subplot(2,2,2,projection='3d')
ax.scatter(df['floors'],df['bedrooms'],df['sqft_living'],c='darkgreen',alpha=0.5)
ax.set(xlabel='\nfloors',ylabel='\nbedrooms',zlabel='\nsqft living')
ax.set(ylim=[0,12])

ax=fig.add_subplot(2,2,3,projection='3d')
ax.scatter(df['sqft_living'],df['sqft_lot'],df['bathrooms'],c='darkgreen',alpha=0.5)
ax.set(xlabel='\nsqft living',ylabel='\nsqft lot',zlabel='\nbathrooms')
ax.set(ylim=[0,250000])

ax=fig.add_subplot(2,2,4,projection='3d')
ax.scatter(df['sqft_living'],df['sqft_lot'],df['bedrooms'],c='darkgreen',alpha=0.5)
ax.set(xlabel='\nsqft living',ylabel='\nsqft lot',zlabel='\nbedrooms')
ax.set(ylim=[0,250000])


# Scatter plot for predictions vs. actual prices
plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions, color='blue', alpha=0.5, label='Actual vs. Predicted Prices')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', linewidth=2, label='Perfect Prediction')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual Prices vs. Predicted Prices')
plt.legend()
plt.show()


# Print coefficients and intercept
coefficients = model.coef_
intercept = model.intercept_
print('Coefficients:', coefficients)
print('Intercept:', intercept)


from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# R-squared score
r2 = r2_score(y_test, predictions)
print('R-squared:', r2)

# Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, predictions)
print('Mean Absolute Error:', mae)

# Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print('Root Mean Squared Error:', rmse)
     
     