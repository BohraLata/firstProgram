import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

# from sklearn.linear_model import LinearRegression
print("Libraries Imported")  # all the libraries are imported

s_data = pd.read_csv("http://bit.ly/w-data")  # accessing the csv file from web
print("Data Imported")
print(s_data)  # print csv file

s_data.head()

# Analysing the data
s_data.plot(x="Hours", y="Scores", kind="scatter")
plt.title("Hours Vs Percentage")
plt.xlabel("Hours Studies")
plt.ylabel("Percentage Scored")
plt.show()

# Data Preparation
x = np.asanyarray(s_data[["Hours"]])
y = np.asanyarray(s_data["Scores"])
# splitting the train and test data

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Training the Model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
print("Coefficient:-", regressor.coef_)
print("Intercept:-", regressor.intercept_)
print("Training Complete")

# Plotting the fit line
fitline = regressor.coef_[0]*x_train + regressor.intercept_
s_data.plot(x="Hours", y="Scores", kind="scatter", figsize=(7, 4), color="b")
plt.plot(x_train, fitline)
plt.title("Hours Vs Percentage")
plt.xlabel("Hours Studies")
plt.ylabel("Percentage Score")
plt.show()

# making prediction to check accuracy
# model Evolution
from sklearn import metrics
from sklearn.metrics import r2_score
y_pre = regressor.predict(x_test)
print("Mean Absolute Error", metrics.mean_absolute_error(y_pre, y_test))
print("r2 Score:-", r2_score(y_test,y_pre))
r = r2_score(y_test,y_pre)

# Comparing Actual and Predictive value
df = pd.DataFrame({"Actual:-", y_test, "Predict:-", y_pre})
print(df)

# Predicting Score value for student who studies 9.25 Hours
hours = 9.25
predicted_score=regressor.predict([[hours]])
print(f"Number of Hours={hours}")
print(f"Predicted Score= {predicted_score}")
print("R2 Score is", r)
