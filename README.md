# Medicine Stock Prediction Model
A model that predicts the number of days in which each type of medicine will run out of stock in the pharmacy.
I added detailed comments in the code file (medicine_stock_prediction_model.py) explaining each step I took to create the model.

This repository contains a machine learning model for predicting medicine stock levels based on demand and quantity information. The model uses a linear regression approach to estimate the difference between dates as a proxy for stock levels.

Dataset
The model is trained on a dataset (med_dataset.csv) that includes information on demand, quantity, and the difference between dates.

Dependencies
pandas
scikit-learn
NumPy
Usage
Clone the repository:
bash

Copy
git clone https://github.com/your-username/medicine-stock-prediction.git
Install the required libraries:
bash

Copy
pip install -r requirements.txt
Run the medicine_stock_prediction_model.py script to train and evaluate the model.
Model Evaluation
The model is evaluated using Mean Squared Error (MSE) and R-squared scores on the training, validation, and test sets. Here are the evaluation metrics for the model:

Training Set:
Mean Squared Error: 80.94
R-squared: 0.88
Validation Set:
Mean Squared Error: 83.77
R-squared: 0.88
Test Set:
Mean Squared Error: 79.47
R-squared: 0.88
Conclusion
The model shows promising performance with an R-squared value of approximately 0.88 on both the training and test sets, indicating that it can predict medicine stock levels with a high degree of accuracy.
