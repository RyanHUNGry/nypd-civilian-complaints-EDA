import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data(filepath: str = "data/complaints_cleaned.csv") -> pd.DataFrame:
    complaints = pd.read_csv(filepath)
    complaints["days_taken"] = pd.to_numeric(
        complaints["days_taken"].astype(str).str.split().str[0], errors="coerce"
    )
    return complaints.dropna(subset=["days_taken"])


def build_dnn_pipeline() -> Pipeline:
    categorical_features = ["complainant_gender", "complainant_ethnicity"]
    numeric_features = ["complainant_age_incident"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
        ],
        remainder="drop",
    )

    dnn_regressor = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", dnn_regressor)])


def main() -> None:
    complaints = load_data()

    X = complaints.drop(columns=["days_taken"])
    y = complaints["days_taken"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=True, test_size=0.2, random_state=42
    )

    pipeline = build_dnn_pipeline()
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    print(f"DNN Test R^2: {r2_score(y_test, preds):.4f}")
    print(f"DNN Test MAE: {mean_absolute_error(y_test, preds):.2f}")


if __name__ == "__main__":
    main()
