# House Price Prediction using Linear Regression

This project is a **Machine Learning program** that predicts house prices based on various features using **Linear Regression**. It is a beginner-friendly project that demonstrates **data preprocessing, model training, and evaluation**.

---

## Dataset

The dataset `Housing.csv` contains the following columns:

| Column | Description |
|--------|-------------|
| price | Price of the house |
| area | Total area of the house (in sqft) |
| bedrooms | Number of bedrooms |
| bathrooms | Number of bathrooms |
| stories | Number of stories |
| mainroad | Whether the house is on the main road (yes/no) |
| guestroom | Presence of guest room (yes/no) |
| basement | Presence of basement (yes/no) |
| hotwaterheating | Hot water heating available (yes/no) |
| airconditioning | Air conditioning available (yes/no) |
| parking | Number of parking spaces |
| prefarea | Located in a preferred area (yes/no) |
| furnishingstatus | Furnishing status (furnished/semi-furnished/unfurnished) |

---

## Data Preprocessing

1. Convert **binary categorical columns** (`yes/no`) to numeric (1/0).
```
   binary_cols = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]
   for col in binary_cols:
       housing[col] = housing[col].map({"yes": 1, "no": 0})
```

2. Apply **one-hot encoding** for the `furnishingstatus` column:
```
   housing = pd.get_dummies(housing, columns=["furnishingstatus"])
```


3. Save the preprocessed dataset:

   ```
   housing.to_csv("housing_encoded.csv", index=False)
   ```


---

## Model Training

* Split the dataset into **train and test sets**:

  ```python
  from sklearn.model_selection import train_test_split
  train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
  X_train = train_set.drop("price", axis=1)
  y_train = train_set["price"]
  X_test = test_set.drop("price", axis=1)
  y_test = test_set["price"]
  ```

* Train a **Linear Regression** model:

  ```python
  from sklearn.linear_model import LinearRegression
  model = LinearRegression()
  model.fit(X_train, y_train)
  ```

* Make predictions and evaluate the model:

  ```python
  from sklearn.metrics import r2_score
  y_pred = model.predict(X_test)
  r2 = r2_score(y_test, y_pred)
  print(f"R^2 Score: {r2:.4f}")
  ```

* **Result:**

  ```
  R^2 Score: 0.6529
  ```

---

## Key Learnings

* Preprocessing is crucial for categorical data.
* Linear Regression can provide **insights into relationships between features and target**.
* Evaluation using **R² score** helps understand model performance.

---

## Next Steps

* Experiment with **feature selection** to improve performance.
* Try **advanced models** like Decision Trees, Random Forests, or Gradient Boosting.
* Deploy the model to a web app for interactive predictions.

---
---

### Tech Stack

* Python 3
* Pandas & NumPy
* scikit-learn (sklearn)
