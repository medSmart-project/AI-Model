import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


data = pd.read_csv('med_dataset.csv')

# features (demand and stock levels) , target variable (Difference Between Dates)
X = data[['demand', 'quantity']]
y = data['Difference Between Dates']

# Split the data into training (70%), validation (15%), and testing (15%) sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Apply MinMaxScaler to scale the features in each set
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# Initialize the Linear Regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train_scaled, y_train)

# ask model to predict the output on train data
# This will produce the highest accuracy.This will produce the highest accuracy
# I Do this just to make sure the model read the data correctly
predictions_train = model.predict(X_train_scaled)


# Calculate MSE score on the train set
mse_train = mean_squared_error(y_train, predictions_train)
print(f'Mean Squared Error on Train Set: {mse_train}')

# Calculate R-squared score on the test set
r2_train = r2_score(y_train, predictions_train)
print(f'R-squared on train Set: {r2_train}')



# Make predictions on the validation data
predictions = model.predict(X_val_scaled)

# Evaluate the model using Mean Squared Error on the validation set
mse = mean_squared_error(y_val, predictions)
print(f'Mean Squared Error on Validation Set: {mse}')

# Calculate R-squared score on the validation set
r2_val = r2_score(y_val, predictions)
print(f'R-squared on Validation Set: {r2_val}')



# predict out of stock status on the test set
predictions_test = model.predict(X_test_scaled)

# Evaluate the model using Mean Squared Error on the test set
mse_test = mean_squared_error(y_test, predictions_test)
print(f'Mean Squared Error on Test Set: {mse_test}')

# Calculate R-squared score on the test set
r2_test = r2_score(y_test, predictions_test)
print(f'R-squared on Test Set: {r2_test}')

"""
Mean Squared Error on Train Set: 80.94242945270766
R-squared on train Set: 0.8828434081680618
Mean Squared Error on Validation Set: 83.77399582226143
R-squared on Validation Set: 0.8766721913577102
Mean Squared Error on Test Set: 79.47034261050811
R-squared on Test Set: 0.8846726881508655

finally we can say the accuracy of this model is almost 0.9 ---> 90% 
"""

