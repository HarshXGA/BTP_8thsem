from sklearn.svm import SVR
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv(r"C:\Users\Gupta\Downloads\btp_data.csv")

# Convert the categorical variables to numerical using one-hot encoding
data = pd.get_dummies(data, columns=['Compensating_Element', 'Recess_Shape'])
y = data["Load_Capacity"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('Load_Capacity', axis=1), data['Load_Capacity'], test_size=0.2, random_state=42)
# Create SVM model
svm_model = SVR(kernel='rbf')

# Train the model
svm_model.fit(X_train, y_train)

# Predict the test set results
y_pred_svm = svm_model.predict(X_test)

# Calculate performance metrics
mse_svm = mean_squared_error(y_test, y_pred_svm)
r2_svm = r2_score(y_test, y_pred_svm)

# Print performance metrics
print("SVM Mean Squared Error:", mse_svm)
print("SVM R^2 Score:", r2_svm)

plt.scatter(y_test, y_pred_svm, alpha=0.5)
plt.plot([0, max(y_test)], [0, max(y_pred_svm)], 'r--', label='Perfect Prediction')
plt.xlabel('Actual Load Capacity')
plt.ylabel('Predicted Load Capacity')
plt.title('Support Vector Machine (SVM) (R^2 = {:.2f})'.format(r2_svm))
plt.legend()
plt.show()
