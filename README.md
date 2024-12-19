
Predict Audience Ratings with Machine Learning

This project predicts audience ratings for movies using machine learning. The pipeline includes data preprocessing, model training, evaluation, and saving the results. The dataset used is "Rotten Tomatoes Movies."

Features

Load data from an Excel file.

Preprocess the dataset by handling missing values and encoding categorical data.

Build and train a linear regression model.

Evaluate the model using metrics like Mean Squared Error (MSE) and R2 Score.

Save the trained model and scaler for future use.

Generate predictions for all rows in the dataset and save the results.

Tools and Libraries Used

Python

pandas, numpy

scikit-learn

matplotlib, seaborn

joblib

How to Use

Place the dataset in the specified path.

Run the script to:

Preprocess the data.

Train the linear regression model.

Evaluate the model's performance.

Save predictions and the trained model.

Check the output files for predictions and model artifacts.

Validation

The model is validated using:

Mean Squared Error (MSE): Measures the average squared difference between predicted and actual values.

R2 Score: Indicates how well the model explains the variance in the target variable.

This project demonstrates building a machine learning pipeline to predict audience ratings efficiently.

