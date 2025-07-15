# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Trains ML model using training dataset and evaluates using test dataset. Saves trained model.
"""

import argparse
import os
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser("train")
    parser.add_argument("--train_data", type=str, help="Path to train dataset")  # Specify the type for train_data
    parser.add_argument("--test_data", type=str, help="Path to test dataset")  # Specify the type for test_data
    parser.add_argument("--model_output", type=str, help="Path of output model")  # Specify the type for model_output
    parser.add_argument('--n_estimators', type=int, default=100,
                        help='The number of trees in the forest')  # Specify the type and default value for n_estimators
    parser.add_argument('--max_depth', type=int, default=10,
                        help='The maximum depth of the tree')  # Specify the type and default value for max_depth

    args = parser.parse_args()

    return args

def main(args, mlflow_enabled=True):
    '''Read train and test datasets, train model, evaluate model, save trained model'''

    # Read train and test data from CSV files
    train_df = pd.read_csv(Path(args.train_data) / "train.csv")
    test_df = pd.read_csv(Path(args.test_data) / "test.csv")

    # Split the data into features(X) and target(y) 
    y_train = train_df['price']  # Specify the target column
    X_train = train_df.drop(columns=['price'])
    y_test = test_df['price']
    X_test = test_df.drop(columns=['price'])

    # Initialize and train a RandomForest Regressor
    model = RandomForestRegressor(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42)  # Provide the arguments for RandomForestRegressor
    model.fit(X_train, y_train)  # Train the model

    # Log model hyperparameters
    if mlflow_enabled:
        try:
            mlflow.log_param("model", "RandomForestRegressor")  # Provide the model name
            mlflow.log_param("n_estimators", args.n_estimators)
            mlflow.log_param("max_depth", args.max_depth)
        except Exception as e:
            print(f"Warning: MLflow parameter logging failed: {e}")
    else:
        print(f"Model: RandomForestRegressor")
        print(f"n_estimators: {args.n_estimators}")
        print(f"max_depth: {args.max_depth}")

    # Predict using the RandomForest Regressor on test data
    yhat_test = model.predict(X_test)  # Predict the test data

    # Compute and log mean squared error for test data
    mse = mean_squared_error(y_test, yhat_test)
    print('Mean Squared Error of RandomForest Regressor on test set: {:.2f}'.format(mse))
    if mlflow_enabled:
        try:
            mlflow.log_metric("MSE", float(mse))  # Log the MSE
        except Exception as e:
            print(f"Warning: MLflow metric logging failed: {e}")

    # Save the model
    if mlflow_enabled:
        try:
            mlflow.sklearn.save_model(sk_model=model, path=args.model_output)  # Save the model
        except Exception as e:
            print(f"Warning: MLflow model saving failed: {e}")
            # Fallback: save using joblib
            import joblib
            os.makedirs(args.model_output, exist_ok=True)
            model_path = os.path.join(args.model_output, "model.pkl")
            joblib.dump(model, model_path)
            print(f"Model saved using joblib to: {model_path}")
    else:
        # Fallback: save using joblib
        import joblib
        os.makedirs(args.model_output, exist_ok=True)
        model_path = os.path.join(args.model_output, "model.pkl")
        joblib.dump(model, model_path)
        print(f"Model saved using joblib to: {model_path}")

if __name__ == "__main__":
    
    # Try to start MLflow run, but continue if it fails due to environment issues
    try:
        mlflow.start_run()
        mlflow_enabled = True
    except Exception as e:
        print(f"Warning: MLflow initialization failed: {e}")
        print("Continuing without MLflow logging...")
        mlflow_enabled = False

    # Parse Arguments
    args = parse_args()

    lines = [
        f"Train dataset input path: {args.train_data}",
        f"Test dataset input path: {args.test_data}",
        f"Model output path: {args.model_output}",
        f"Number of Estimators: {args.n_estimators}",
        f"Max Depth: {args.max_depth}"
    ]

    for line in lines:
        print(line)

    main(args, mlflow_enabled)

    # End MLflow run only if it was successfully started
    if mlflow_enabled:
        try:
            mlflow.end_run()
        except Exception as e:
            print(f"Warning: MLflow end_run failed: {e}")

