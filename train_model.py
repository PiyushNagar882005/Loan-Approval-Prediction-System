from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "loan_data.csv"
MODEL_PATH = ROOT / "models" / "loan_approval_model.pkl"

def main():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["Loan_Status"])
    y = df["Loan_Status"]

    numeric_features = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount",
                        "Loan_Amount_Term", "Credit_History"]
    categorical_features = ["Gender", "Married", "Dependents", "Education",
                             "Self_Employed", "Property_Area"]

    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=250, max_depth=8, min_samples_split=5,
            class_weight="balanced", random_state=42
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
    print("\\nClassification report:\\n", classification_report(y_test, predictions))
    print("Confusion matrix:\\n", confusion_matrix(y_test, predictions))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\\nSaved model to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
