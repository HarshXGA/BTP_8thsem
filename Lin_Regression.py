import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


# load the dataset
dataframe = pd.read_csv(r"C:\Users\Gupta\Downloads\btp_data.csv")

# split the dataset into input and target variables
X = dataframe[["Compensating_Element", "Recess_Shape", "Electric_Field"]]
y = dataframe["Load_Capacity"]

# encode categorical variables using one-hot encoding
X = pd.get_dummies(X, columns=["Compensating_Element", "Recess_Shape"])
X= X.astype(int)

# Encoding ends

# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# make predictions on the testing set
y_pred = model.predict(X_test)

# evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# create a 3D scatter plot

plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([0, max(y)], [0, max(y_pred)], 'r--', label='Perfect Prediction')
plt.xlabel('Actual Load Capacity')
plt.ylabel('Predicted Load Capacity')
plt.title('Linear Regression Model (R^2 = {:.2f})'.format(r2))
plt.legend()
plt.show()

print("Mean Squared Error: ", mse)
print("R-Squared: ", r2)
