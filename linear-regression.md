Great, let us start with **Linear Regression** 📈

# 1) Linear Regression — Theory

## What it does

Linear Regression is used to **predict a continuous number**.

Examples:

* house price
* salary
* marks
* sales
* temperature

So unlike classification, where the answer is a category, here the answer is a **value**.

## Basic idea

It tries to find a straight-line relationship between input and output.

For one feature:

[
y = mx + b
]

Where:

* **y** = predicted output
* **x** = input feature
* **m** = slope
* **b** = intercept

In machine learning form, we write:

[
y = w_0 + w_1x_1 + w_2x_2 + \dots + w_nx_n
]

Where:

* **w0** = bias/intercept
* **w1, w2, ...** = weights
* **x1, x2, ...** = features

## Example intuition

Suppose you want to predict marks from study hours.

* more hours studied → usually more marks
* less hours studied → usually fewer marks

Linear regression tries to draw the **best fitting line** through the data points.

## How it learns

The model makes predictions, compares them with actual values, and tries to reduce the error.

The most common error measure is:

[
\text{MSE} = \frac{1}{n}\sum (y_{true} - y_{pred})^2
]

This is called **Mean Squared Error**.

Why square?

* to make negative errors positive
* to punish large mistakes more

## Goal of training

The model adjusts weights so that the line fits the data as well as possible.

## Important assumptions

In viva, teacher may ask this. Say:

* Relationship is roughly linear
* Features are not highly correlated
* Errors are independent
* Error variance is roughly constant

## Where it is used

* price prediction
* demand forecasting
* trend estimation

---

# 2) Practical Implementation in Python

We will use a built-in dataset from `sklearn`, so it is easy to run and explain.

We will use the **Diabetes dataset**:

* input features: medical measurements
* target: disease progression score

This is a nice dataset because it is already available inside Python.

## Code

```python
# Linear Regression using the diabetes dataset

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load the dataset
diabetes = load_diabetes()

# 2. Convert data into a DataFrame for easier viewing
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = pd.Series(diabetes.target, name="target")

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Create the model
model = LinearRegression()

# 5. Train the model
model.fit(X_train, y_train)

# 6. Make predictions
y_pred = model.predict(X_test)

# 7. Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

# 8. Print results
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R2 Score:", r2)

# 9. Show coefficients
print("\nIntercept:", model.intercept_)
print("Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.4f}")

# 10. Plot actual vs predicted values
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")
plt.grid(True)
plt.show()
```

---

# 3) Explanation of the Code, End to End

## Importing libraries

```python
import pandas as pd
import matplotlib.pyplot as plt
```

* `pandas` helps us store and view data in table form.
* `matplotlib.pyplot` helps us draw a graph.

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```

These are from `scikit-learn`:

* `load_diabetes()` gives the dataset
* `train_test_split()` splits data into train and test parts
* `LinearRegression()` creates the model
* metric functions help us evaluate the model

---

## Loading the dataset

```python
diabetes = load_diabetes()
```

This loads the diabetes dataset into memory.

It contains:

* input features
* target values

---

## Creating `X` and `y`

```python
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = pd.Series(diabetes.target, name="target")
```

Here:

* `X` is the input data
* `y` is the output we want to predict

### Why do we separate them?

Because the model learns from:

* **features** → input
* **target** → answer

### What is in `X`?

Columns like:

* age
* sex
* bmi
* bp
* and others

### What is in `y`?

A numeric progression score.

---

## Splitting training and testing data

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

This means:

* 80% data goes to training
* 20% data goes to testing

### Why split?

To check if the model works on new unseen data.

If we test on the same data we trained on, the result can be misleading.

### `random_state=42`

This makes the split the same every time you run the code.

---

## Creating the model

```python
model = LinearRegression()
```

This creates a linear regression model object.

At this point, the model is empty. It has not learned anything yet.

---

## Training the model

```python
model.fit(X_train, y_train)
```

This is the learning step.

The model:

* looks at training features
* compares them with training targets
* finds the best weights and intercept

So after this line, the model is trained.

---

## Making predictions

```python
y_pred = model.predict(X_test)
```

Now the model uses test features to predict output values.

These predictions are stored in `y_pred`.

---

## Evaluating the model

```python
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)
```

### MAE

Mean Absolute Error:

* average absolute difference between actual and predicted values
* easier to understand

### MSE

Mean Squared Error:

* penalizes bigger errors more

### RMSE

Root Mean Squared Error:

* square root of MSE
* same unit as the target

### R² score

* tells how well the model explains the variation in the target
* closer to 1 is better

A rough interpretation:

* **1** = perfect prediction
* **0** = no better than simple average
* negative = very poor model

---

## Printing results

```python
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R2 Score:", r2)
```

This shows the final performance numbers.

In viva, you can say:

> “These metrics tell us how far our predictions are from the actual values.”

---

## Intercept and coefficients

```python
print("\nIntercept:", model.intercept_)
print("Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.4f}")
```

### Intercept

This is the starting value of the line.

### Coefficients

These tell how much each feature affects the prediction.

Example:

* positive coefficient → increases prediction
* negative coefficient → decreases prediction

So if `bmi` has a high positive weight, it means BMI strongly increases the predicted score.

---

## Plotting actual vs predicted

```python
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")
plt.grid(True)
plt.show()
```

This creates a scatter plot.

### How to interpret it

* If points are close to a diagonal line, the model is good
* If points are scattered a lot, the model is not very accurate

---

# 4) Very Important Viva Points

You can say these confidently:

### What is Linear Regression?

It is a supervised machine learning algorithm used to predict continuous values.

### Why is it called linear?

Because it assumes a linear relationship between input and output.

### What does the model learn?

It learns the best line by finding weights and intercept.

### What is the loss function?

Usually Mean Squared Error.

### Why split train and test data?

To check whether the model generalizes to unseen data.

### What does R² mean?

It measures how much of the target variation is explained by the model.

---

# 5) Simple one-line explanation for teacher

You can say:

> “Linear regression finds the best straight line that fits the training data and uses it to predict continuous values.”

---

# 6) Mini recap

* **Theory:** predicts continuous values
* **Equation:** ( y = w_0 + w_1x_1 + ... + w_nx_n )
* **Training:** minimizes error
* **Evaluation:** MAE, MSE, RMSE, R²
* **Use case:** price, marks, sales prediction

Now I can do the next part in the same style: **Logistic Regression**.
