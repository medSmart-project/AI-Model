# Medicine Stock Prediction Model

This repository contains a machine learning model for predicting medicine stock levels based on demand and quantity information. The model uses a linear regression approach to estimate the difference between dates as a proxy for stock levels.

## Dataset
The model is trained on a dataset (`med_dataset.csv`) that includes information on demand, quantity, and the difference between dates.

## Dependencies
- pandas
- scikit-learn
- NumPy

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/medicine-stock-prediction.git
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the `medicine_stock_prediction_model.py` script to train and evaluate the model.

## Model Evaluation
The model is evaluated using Mean Squared Error (MSE) and R-squared scores on the training, validation, and test sets. Here are the evaluation metrics for the model:
- **Training Set:**
  - Mean Squared Error: 80.94
  - R-squared: 0.88
- **Validation Set:**
  - Mean Squared Error: 83.77
  - R-squared: 0.88
- **Test Set:**
  - Mean Squared Error: 79.47
  - R-squared: 0.88

## Conclusion
The model shows promising performance with an R-squared value of approximately 0.88 on both the training and test sets, indicating that it can predict medicine stock levels with a high degree of accuracy.

---

You can modify and expand this template to include more detailed information about the model, the dataset, the preprocessing steps, and any other relevant details about the project.
