# import necessary libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

# load the data into a pandas dataframe
df = pd.read_csv('your_dataset.csv')

# encode categorical columns
le = LabelEncoder()
df['Compensating_Element'] = le.fit_transform(df['Compensating_Element'])
df['Recess_Shape'] = le.fit_transform(df['Recess_Shape'])

# split data into input and output variables
X = df.drop('Load_Capacity', axis=1)
y = df['Load_Capacity']

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create and fit linear regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# calculate and print cross-validation score for linear regression
lr_cv_score = cross_val_score(lr, X, y, cv=5, scoring='neg_mean_squared_error')
print("Linear Regression Cross-Validation Score:", lr_cv_score.mean())

# create and fit decision tree regressor model
dtr = DecisionTreeRegressor(random_state=42)
dtr.fit(X_train, y_train)

# calculate and print cross-validation score for decision tree regressor
dtr_cv_score = cross_val_score(dtr, X, y, cv=5, scoring='neg_mean_squared_error')
print("Decision Tree Regressor Cross-Validation Score:", dtr_cv_score.mean())

# create and fit random forest regressor model
rfr = RandomForestRegressor(n_estimators=100, random_state=42)
rfr.fit(X_train, y_train)

# calculate and print cross-validation score for random forest regressor
rfr_cv_score = cross_val_score(rfr, X, y, cv=5, scoring='neg_mean_squared_error')
print("Random Forest Regressor Cross-Validation Score:", rfr_cv_score.mean())

# create and fit gradient boosting regressor model
gbr = GradientBoostingRegressor(n_estimators=100, random_state=42)
gbr.fit(X_train, y_train)

# calculate and print cross-validation score for gradient boosting regressor
gbr_cv_score = cross_val_score(gbr, X, y, cv=5, scoring='neg_mean_squared_error')
print("Gradient Boosting Regressor Cross-Validation Score:", gbr_cv_score.mean())

# create and fit XGBoost regressor model
xgb = XGBRegressor(n_estimators=100, random_state=42)
xgb.fit(X_train, y_train)

# calculate and print cross-validation score for XGBoost regressor
xgb_cv_score = cross_val_score(xgb, X, y, cv=5, scoring='neg_mean_squared_error')
print("XGBoost Regressor Cross-Validation Score:", xgb_cv_score.mean())
