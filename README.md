# Predicting Car Prices with Machine Learning Models
This project focuses on building regression models to predict car prices based on various features in a dataset. Below, we describe the dataset, techniques, and models used.

## Dataset
The dataset (data.csv) contains various features about cars, including their specifications and price. Here are key steps performed on the data:

## Data Loading and Exploration

Imported the dataset using Pandas.

Explored the structure using .head(), .info(), and .isnull().sum().
## Handling Categorical Data

Encoded text columns using LabelEncoder from scikit-learn. Columns encoded include:

CarName

fueltype

aspiration

doornumber

carbody

drivewheel

enginelocation

fuelsystem

enginetype

cylindernumber
## Feature and Target Separation

Features (X) were separated by dropping the price column.

Target variable (y) was assigned as the price column.
## Splitting Data

Used train_test_split with a 90/10 split to divide the data into training and validation sets.
## Models and Techniques
### 1. Decision Tree Regressor
A tree-based model that splits data recursively to predict the target.

Strength: Easy to interpret and handles non-linear data well.
Limitation: Prone to overfitting.

### 2. Random Forest Regressor
An ensemble method combining multiple decision trees for better generalization.

Strength: Reduces overfitting and improves accuracy.

Limitation: Computationally intensive.
### 3. Gradient Boosting Regressor
Sequentially builds models that correct errors made by previous models.

Strength: High accuracy and efficient handling of skewed data.

Limitation: Requires careful tuning of hyperparameters.
### 4. Support Vector Regressor (SVR)
Uses support vector machines to create a hyperplane that predicts the target.

Strength: Effective in high-dimensional spaces.

Limitation: Sensitive to parameter selection.
### 5. Linear Regression
A simple linear approach to modeling the relationship between features and target.

Strength: Fast and interpretable.

Limitation: Assumes linear relationships.
## Evaluation
Each model was evaluated using the Mean Absolute Error (MAE) metric, which calculates the average absolute differences between predicted and actual values.

## Results:
Best Model: Gradient Boosting Regressor (lowest MAE).
## Final Model
The Gradient Boosting Regressor was retrained on the training set and evaluated on the validation set. The predictions and differences between actual and predicted values were stored in a dataframe.

## Outputs
Model Results: results DataFrame contains actual vs. predicted prices and their differences.

Performance Metric: MAE for the Gradient Boosting model is reported as the final performance measure.

## Dependencies
Ensure the following Python libraries are installed:

pandas

scikit-learn

Install them using:

pip install pandas scikit-learn

## Usage
Load the dataset into data.csv.

Run the Python script to train models and evaluate results.

Examine the results DataFrame for predicted car prices.
