# Online Payments Fraud Detection

## Project Overview
This project aims to detect fraudulent online payment transactions using a **Decision Tree Classifier**. The dataset includes transaction details, and the target variable (`isFraud`) indicates whether a transaction is fraudulent (1) or not (0). The goal is to build an accurate model to assist in identifying suspicious transactions effectively.

---

## Dataset Features
The dataset contains the following key columns:

- **`step`**: A unit of time (1 step = 1 hour).
- **`type`**: The type of online transaction (e.g., CASH-IN, CASH-OUT, TRANSFER).
- **`amount`**: The transaction amount.
- **`oldbalanceOrg`**: The initial balance of the sender's account.
- **`newbalanceOrig`**: The remaining balance of the sender's account after the transaction.
- **`oldbalanceDest`**: The initial balance of the receiver's account.
- **`newbalanceDest`**: The updated balance of the receiver's account after the transaction.
- **`isFraud`**: The target variable (1 = fraud, 0 = not fraud).

---

## Results
- **Model Accuracy**: Achieved an accuracy of **{insert_accuracy:.2f}** using the Decision Tree Classifier.
- **Confusion Matrix**: Provides a visual understanding of correct and incorrect predictions for both fraudulent and non-fraudulent transactions.

---

## Example Prediction
An example prediction result for a random transaction:
- **Predicted Outcome**: Fraud or Not Fraud (based on model prediction)

---

## Visualization
The project includes insightful visualizations:
- **Fraudulent Transaction Distribution**: Displays the balance of fraud vs. non-fraud transactions.
- **Feature Correlation Heatmap**: Highlights relationships between features to understand their influence on fraud detection.
- **Confusion Matrix**: Visualizes the performance of the model's predictions.

---

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Install the required Python libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn joblib
   ```

3. Place the dataset file (`online_payments_fraud.csv`) in the project directory.

---

## Usage

1. Run the script:
   ```bash
   python fraud_detection.py
   ```

2. Outputs:
   - **Visualizations**: Fraud distribution, correlation heatmap, and confusion matrix.
   - **Model Accuracy**: Displays the evaluation metrics.
   - **Example Prediction**: Generates a prediction for a random transaction.
   - **Model Saving**: Trains and saves the model as `fraud_detection_model.pkl`.

---

## File Structure

- **fraud_detection.py**: Main script containing the preprocessing, model training, evaluation, and visualizations.
- **online_payments_fraud.csv**: Dataset file (place this in the project directory).
- **README.md**: Project documentation.
- **fraud_detection_model.pkl**: Saved machine learning model after training.

---

## Libraries Used
- **Pandas**: Data manipulation and preprocessing.
- **NumPy**: Numerical computations.
- **Matplotlib**: Data visualization.
- **Seaborn**: Statistical data visualization.
- **Scikit-learn**: Machine learning algorithms and evaluation metrics.
- **Joblib**: Model saving and loading.

---

## Future Enhancements
- Incorporate advanced machine learning models (e.g., Random Forest, Gradient Boosting) for improved accuracy.
- Add hyperparameter tuning for model optimization.
- Implement a web interface to make predictions on new transactions.

---

## Acknowledgments
This project is built using publicly available datasets and libraries. It serves as an educational tool for fraud detection techniques.
