# BLENDED_LEARNING
# Implementation-of-Linear-Regression-for-Predicting-Car-Prices
## AIM:
To write a program to predict car prices using a linear regression model and test the assumptions for linear regression.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import Libraries: Bring in essential libraries such as pandas, numpy, matplotlib, and sklearn.

2. Load Dataset: Import the dataset containing car prices along with relevant features.

3. Data Preprocessing: Manage missing data and select key features for the model, if required.

4. Split Data: Divide the dataset into training and testing subsets.

5. Train Model: Build a linear regression model and train it using the training data.

6. Make Predictions: Apply the model to predict outcomes for the test set.
   
7. Evaluate Model: Measure the model's performance using metrics like R² score, Mean Absolute Error (MAE), etc.

8. Check Assumptions: Plot residuals to verify assumptions like homoscedasticity, normality, and linearity.

9. Output Results: Present the predictions and evaluation metrics.


## Program:
```
/*
 Program to implement linear regression model for predicting car prices and test assumptions.
Developed by: Nandhika P
RegisterNumber: 212223040125  
*/
```
```
import pandas as pd
import numpy as np
df=pd.read_csv("CarPrice_Assignment.csv")

X=df[["enginesize","horsepower","citympg","highwaympg"]]
Y=df["price"]

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2, random_state = 42)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train_scaled,Y_train)

Y_pred=model.predict(X_test_scaled)
Y_pred

print("MODEL COEFFICIENTS:")
for feature, coef in zip(X.columns,model.coef_):
    print(f" {feature:>12}: {coef:>10.2f}")
print(f"{'Intercept': >12}: {model.intercept_:>10.2f}")

from sklearn.metrics import mean_squared_error,r2_score
print("\nMODEL PERFORMANCE:")
print(f"{'MSE':}: {mean_squared_error(Y_test, Y_pred):}")
print(f"{'RMSE':}: {np.sqrt(mean_squared_error(Y_test,Y_pred)):}")
print(f"{'R-squared':}: {r2_score(Y_test, Y_pred):}")

import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

#1. Linearity check
plt.figure(figsize=(10, 5))
plt.scatter(Y_test, Y_pred, alpha=0.6)
plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'r--')
plt.title("Linearity Check: Actual vs Predicted Prices")
plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.grid(True)
plt.show()

#2. Independence (Durbin-Watson)

residuals= Y_test - Y_pred
dw_test = sm.stats.durbin_watson(residuals)
print(f"\nDurbin-Watson Statistic: {dw_test:.2f}", "\n(Values close to 2 indicate no autocorrelation)")

#3. Homoscedasticity
plt.figure(figsize=(10, 5))
sns.residplot(x=Y_pred, y=residuals, lowess=True, line_kws={'color': 'red'})
plt.title("Homoscedasticity Check: Residuals vs Predicted")
plt.xlabel("Predicted Price ($)")
plt.ylabel("Residuals ($)")
plt.grid(True)
plt.show()

#4. Normality of residuals
fig, (ax1, ax2)=plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(residuals, kde=True, ax=ax1)
ax1.set_title("Residuals Distribution")
sm.qqplot(residuals, line='45', fit=True, ax=ax2)
ax2.set_title("Q-Q Plot")
plt.tight_layout()
plt.show()

```
## Output:

<img width="673" height="194" alt="image" src="https://github.com/user-attachments/assets/5ce96b87-19cd-4c01-8c9e-8730ec0de23d" />

<img width="283" height="122" alt="image" src="https://github.com/user-attachments/assets/853e7d24-222d-4e3a-b4f8-dae4c4c1f971" />

<img width="329" height="93" alt="image" src="https://github.com/user-attachments/assets/dbf2f543-b727-4628-991a-34d7e268d46b" />

<img width="907" height="451" alt="image" src="https://github.com/user-attachments/assets/c8dcba60-2ce9-41b3-b360-39513491306d" />

<img width="918" height="442" alt="image" src="https://github.com/user-attachments/assets/9f62fc81-81cb-483f-9288-33ceb3fecaf3" />

<img width="973" height="418" alt="image" src="https://github.com/user-attachments/assets/3f702eef-b5f0-413d-a1d0-6f9cde121a31" />

## Result:

   Thus, the program to implement a linear regression model for predicting car prices is written and verified using Python programming, along with the testing of key assumptions for linear regression.
