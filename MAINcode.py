# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv(r"C:\Users\Gupta\Downloads\btp_data.csv")

# Convert the categorical variables to numerical using one-hot encoding
data = pd.get_dummies(data, columns=['Compensating_Element', 'Recess_Shape'])
y = data["Load_Capacity"]
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('Load_Capacity', axis=1), data['Load_Capacity'], test_size=0.2, random_state=42)

# Random Forest Regression
n_estimators=100
random_state=42
rf_model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
rf_model.fit(X_train, y_train)

# make predictions on the testing set
y_pred = rf_model.predict(X_test)

# evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("FOR n_estimators=",n_estimators," and random_state=",random_state,": \n")
print("Random Forest Regression: ")
print("Mean Squared Error: ", mse)
print("R-Squared: ", r2," \n ")

# Gradient Boosting Regressor
gb_model = GradientBoostingRegressor(n_estimators=n_estimators, random_state=random_state)
gb_model.fit(X_train, y_train)

# make predictions on the testing set
y_pred = gb_model.predict(X_test)

# evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Gradient Booster Model: ")
print("Mean Squared Error: ", mse)
print("R-Squared: ", r2, "\n")

# XGBoost Regressor
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=n_estimators, random_state=random_state)
xgb_model.fit(X_train, y_train)

# make predictions on the testing set
y_pred = xgb_model.predict(X_test)

# evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)



print("XGBoost Regression Model: ")
print("min Mean Squared Error: ", mse)
print("max R-Squared: ", r2,"\n")
print(" n_estimators=",n_estimators," random_state= ",random_state)

plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([0, max(y_test)], [0, max(y_pred)], 'r--', label='Perfect Prediction')
plt.xlabel('Actual Load Capacity')
plt.ylabel('Predicted Load Capacity')
plt.title('XGBoost Regression (R^2 = {:.2f})'.format(r2))
plt.legend()
plt.show()
