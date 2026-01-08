# Gradient Boosting Regression (Scikit-Learn)

This project demonstrates how to use **Gradient Boosting Regressor** with a built-in **scikit-learn dataset** to predict house prices.

The goal is to understand:
- How Gradient Boosting works
- How to train a regression model
- How to evaluate model performance

---

## 📌 Dataset Used

**California Housing Dataset (sklearn)**

- Provided by `sklearn.datasets`
- No manual download required
- Target variable: `Median House Value`

---

## 📌 Libraries Used

- **NumPy** – numerical computations  
- **Pandas** – data handling  
- **Scikit-learn** – machine learning models and utilities  

---

## 📌 How the Code Works (Step-by-Step)

### 1️⃣ Import Required Libraries

```python
import numpy as np
import pandas as pd
````

Used for numerical operations and data handling.

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
```

Used for:

* Loading dataset
* Splitting data
* Building the model
* Evaluating performance

---

### 2️⃣ Load the Dataset

```python
housing = fetch_california_housing()
```

Loads the California housing dataset into memory.

```python
X = housing.data
y = housing.target
```

* `X` → input features (house details)
* `y` → target values (house prices)

---

### 3️⃣ Split Data into Train and Test Sets

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

* 80% data for training
* 20% data for testing
* Prevents overfitting
* Ensures fair evaluation

---

### 4️⃣ Create Gradient Boosting Model

```python
gboost = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
```

**Key Parameters:**

* `n_estimators`: number of decision trees
* `learning_rate`: controls contribution of each tree
* `max_depth`: limits tree complexity
* `random_state`: ensures reproducible results

---

### 5️⃣ Train the Model

```python
gboost.fit(X_train, y_train)
```

The model learns patterns between input features and target values.

---

### 6️⃣ Make Predictions

```python
y_pred = gboost.predict(X_test)
```

Predicts house prices for unseen test data.

---

### 7️⃣ Evaluate the Model

```python
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
```

* **R² Score** → how well the model explains variance
* **MSE** → average squared prediction error

```python
print("R2 Score:", r2)
print("Mean Squared Error:", mse)
```

Displays model performance.

---

## 📌 What is Gradient Boosting?

Gradient Boosting is an **ensemble learning method** that:

* Builds multiple decision trees
* Trains trees sequentially
* Each tree corrects errors of the previous one
* Produces high accuracy on tabular data

---

## 📌 Key Takeaways

* Gradient Boosting reduces bias and improves accuracy
* Learning rate and number of trees control performance
* Strong model for regression problems
* Works well on structured datasets

---

## 📌 Future Improvements

* Hyperparameter tuning with GridSearchCV
* Feature importance analysis
* Comparison with Random Forest or XGBoost
* Cross-validation

---
