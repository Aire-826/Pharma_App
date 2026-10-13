import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def train_model():
  df = pd.read_csv("niper_budget_data.csv")

  categorical_cols = ["campus_name", "grant_category", "quarter_released"]
  numerical_cols = [
      "budget_allocated_cr",
      "release_delay_days",
      "procurement_completion_pct",
      "prior_year_utilization_pct",
      "active_projects_count",
  ]

  X = df[categorical_cols + numerical_cols]
  y = df["actual_utilization_pct"]

  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42
  )

  preprocessor = ColumnTransformer(
      transformers=[
          ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
          ("num", StandardScaler(), numerical_cols),
      ]
  )

  pipeline = Pipeline(
      steps=[
          ("preprocessor", preprocessor),
          (
              "regressor",
              RandomForestRegressor(
                  n_estimators=100, random_state=42, max_depth=12
              ),
          ),
      ]
  )

  pipeline.fit(X_train, y_train)

  preds = pipeline.predict(X_test)
  print(f"Model Training Completed.")
  print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, preds):.2f}%")
  print(f"R² Score: {r2_score(y_test, preds):.4f}")

  joblib.dump(pipeline, "niper_model.joblib")
  print("Saved pipeline artifact to 'niper_model.joblib'")


if __name__ == "__main__":
  train_model()