import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


def train_regression_models(df, target_column):

# Train regression models and select the best model using R2 score.

    # Separate input features and output target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Convert categorical columns into numeric columns
    X = pd.get_dummies(X, drop_first=True)

# Save feature names after one-hot encoding
    feature_names = X.columns.tolist()

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data into training and testing parts
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42
    )

    # Define models
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=50,
            random_state=42
        )
    }

    results = []
    best_model = None
    best_model_name = ""
    best_score = -999

    # Train and evaluate each model
    for model_name, model in models.items():

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        results.append({
            "Model": model_name,
            "R2 Score": round(r2, 4),
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2)
        })

        # Select best model
        if r2 > best_score:
            best_score = r2
            best_model = model
            best_model_name = model_name

    results_df = pd.DataFrame(results)

    # Save best model and scaler
    joblib.dump(best_model, "models/best_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(feature_names, "models/feature_names.pkl")

    return results_df, best_model_name, best_score, feature_names