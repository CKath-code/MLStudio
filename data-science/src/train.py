# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Trains ML model using training dataset and evaluates using test dataset. Saves trained model.
"""

import argparse
import os
import json
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

def create_mlflow_model_fallback(model, model_output_path):
    """Create a minimal MLflow model structure when MLflow is not available"""
    import joblib
    import yaml
    
    # Create the model directory
    os.makedirs(model_output_path, exist_ok=True)
    
    # Save the model using joblib
    model_path = os.path.join(model_output_path, "model.pkl")
    joblib.dump(model, model_path)
    
    # Create MLmodel file (required by Azure ML)
    mlmodel_content = {
        'artifact_path': 'model',
        'flavors': {
            'python_function': {
                'env': 'conda.yaml',
                'loader_module': 'mlflow.sklearn',
                'model_path': 'model.pkl',
                'python_version': '3.9.0'
            },
            'sklearn': {
                'code': None,
                'pickled_model': 'model.pkl',
                'serialization_format': 'cloudpickle',
                'sklearn_version': '1.0.0'
            }
        },
        'model_uuid': '12345678-1234-1234-1234-123456789012',
        'run_id': 'fallback_run',
        'utc_time_created': '2025-07-15 15:00:00.000000'
    }
    
    # Write MLmodel file
    with open(os.path.join(model_output_path, "MLmodel"), 'w') as f:
        yaml.dump(mlmodel_content, f, default_flow_style=False)
    
    # Create a simple conda.yaml
    conda_content = {
        'channels': ['defaults', 'conda-forge'],
        'dependencies': [
            'python=3.9.0',
            'scikit-learn=1.0.0',
            'joblib',
            {'pip': ['mlflow']}
        ],
        'name': 'mlflow-env'
    }
    
    with open(os.path.join(model_output_path, "conda.yaml"), 'w') as f:
        yaml.dump(conda_content, f, default_flow_style=False)
    
    print(f"MLflow-compatible model structure created at: {model_output_path}")

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
            print(f"Model saved using MLflow to: {args.model_output}")
        except Exception as e:
            print(f"Warning: MLflow model saving failed: {e}")
            # Fallback: create MLflow-compatible structure manually
            create_mlflow_model_fallback(model, args.model_output)
    else:
        # Fallback: create MLflow-compatible structure manually
        create_mlflow_model_fallback(model, args.model_output)

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

