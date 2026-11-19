# Loan Approval Prediction System

A beginner-friendly machine-learning classification project with a Streamlit interface.

## Important dataset note
`data/loan_data.csv` is a **synthetic demonstration dataset** generated for this project. It is not a real bank dataset and the model's scores are not evidence of real-world lending performance. For a portfolio project, you can replace it with a properly licensed public loan prediction dataset (for example, the commonly used Loan Prediction dataset) after checking its terms and matching the column names.

## Features
- Handles missing numerical and categorical values with imputers.
- One-hot encodes categorical columns.
- Trains a Random Forest classifier.
- Reports accuracy, precision, recall, F1-score and confusion matrix.
- Provides an interactive Streamlit prediction form.

## Project structure
```text
loan_approval_prediction/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   └── loan_data.csv
└── models/
    └── loan_approval_model.pkl
```

## Setup (Windows)
1. Install Python 3.10 or newer.
2. Extract the ZIP and open this folder in VS Code.
3. Open the terminal and create a virtual environment:

   ```bash
   python -m venv venv
   venv\\Scripts\\activate
   ```

4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Train the model (optional; a demo model is already included):

   ```bash
   python train_model.py
   ```

6. Start the web app:

   ```bash
   streamlit run app.py
   ```

The app opens in your browser, usually at `http://localhost:8501`.

## How it works
1. Read the CSV dataset with Pandas.
2. Separate input features from `Loan_Status`.
3. Split data into training and testing sets (80:20).
4. Impute missing values and encode categorical features in a preprocessing pipeline.
5. Train a Random Forest classifier.
6. Evaluate predictions on the held-out test set.
7. Save the complete preprocessing + model pipeline with Joblib.
8. Load the pipeline in Streamlit and predict for user-entered applicant details.

## Interview explanation
"I built a binary classification project that estimates whether a loan application is likely to be approved from applicant and loan details. I used Pandas for data handling and Scikit-learn Pipelines for missing-value imputation, one-hot encoding and Random Forest classification. I evaluated it on a held-out test set using precision, recall, F1-score and a confusion matrix, then created a Streamlit interface for interactive predictions."

## Limitations and responsible use
- This bundled dataset is synthetic and intended only to demonstrate the workflow.
- The model can learn biases from data and may produce inaccurate predictions.
- Do not use this project to make actual credit, eligibility or financial decisions.
- For a serious project, use a documented, representative, legally usable dataset and evaluate fairness, calibration and performance on external data.
