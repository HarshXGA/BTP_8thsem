from sklearn.svm import SVR
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers


# Load the dataset
data = pd.read_csv(r"C:\Users\Gupta\Downloads\btp_data.csv")

# Convert the categorical variables to numerical using one-hot encoding
data = pd.get_dummies(data, columns=['Compensating_Element', 'Recess_Shape'])
y = data["Load_Capacity"]
data = np.asarray(data).astype(np.float32)
y = np.asarray(y).astype(np.float32)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[0:-2][:223], y[:223], test_size=0.2, random_state=42)

# Define the ANN model architecture
model = keras.Sequential()
model.add(layers.Dense(10, activation='linear', input_dim=X_train.shape[1]))
model.add(layers.Dense(10, activation='linear'))
model.add(layers.Dense(1, activation='linear'))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Predict the test set results
y_pred_ann = model.predict(X_test)

# Calculate performance metrics
mse_ann = mean_squared_error(y_test, y_pred_ann)
r2_ann = r2_score(y_test, y_pred_ann)

# Print performance metrics
print("ANN Mean Squared Error:", mse_ann)
print("ANN R^2 Score:", r2_ann)


plt.scatter(y_test, y_pred_ann, alpha=0.5)
plt.plot([0, max(y_test)], [0, max(y_pred_ann)], 'r--', label='Perfect Prediction')
plt.xlabel('Actual Load Capacity')
plt.ylabel('Predicted Load Capacity')
plt.title('Aritifical Neural Network (R^2 = {:.2f})'.format(r2_ann))
plt.legend()
plt.show()
