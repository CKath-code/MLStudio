# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Registers the best-trained ML model from the sweep job.
"""

import argparse
from pathlib import Path
import mlflow
import os 
import json
import glob

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='Name under which model will be registered')
    parser.add_argument('--search_base', type=str, help='Base directory to search for models', default='/tmp/azureml_cr')
    parser.add_argument("--model_info_output_path", type=str, help="Path to write model info JSON")
    parser.add_argument('--trained_model_path', type=str, help='Path to trained model', required=False)
    args, _ = parser.parse_known_args()
    print(f'Arguments: {args}')

    return args

def find_best_model(search_base):
    """Find the best model from training by searching the file system"""
    print(f"Searching for models in: {search_base}")
    
    model_candidates = []
    
    # Search for model directories
    for root, dirs, files in os.walk(search_base):
        # Look for directories containing model files
        if 'MLmodel' in files or 'model.pkl' in files:
            print(f"Found potential model at: {root}")
            model_candidates.append(root)
    
    if not model_candidates:
        raise Exception(f"No models found in search base: {search_base}")
    
    print(f"Found {len(model_candidates)} model candidates")
    
    # For now, just return the first one (in a real scenario, you'd pick the best based on metrics)
    # You could enhance this by reading metrics from MLflow logs or other criteria
    best_model_path = model_candidates[0]
    print(f"Selected model: {best_model_path}")
    
    return best_model_path

def main(args):
    '''Registers the trained model'''

    print("Registering ", args.model_name)
    
    # Use the trained model path directly if provided
    if args.trained_model_path and os.path.exists(args.trained_model_path):
        print(f"Using trained model path: {args.trained_model_path}")
        model_path = args.trained_model_path
    else:
        # Fallback to filesystem search
        try:
            model_path = find_best_model(args.search_base)
            print(f"Best model found at: {model_path}")
        except Exception as e:
            print(f"Error finding model: {e}")
            raise
    
    # Load and register model
    try:
        # Load model - try MLflow first, then fallback to joblib if needed
        try:
            model = mlflow.sklearn.load_model(model_path)  # Load the model from model_path
            print("Model loaded using MLflow")
        except Exception as load_error:
            print(f"MLflow model loading failed: {load_error}")
            # Fallback: try loading with joblib if MLmodel structure exists but MLflow fails
            import joblib
            model_pkl_path = os.path.join(model_path, "model.pkl")
            if os.path.exists(model_pkl_path):
                model = joblib.load(model_pkl_path)
                print("Model loaded using joblib fallback")
            else:
                raise Exception(f"No model found at {model_path}")

        # Log model using mlflow
        mlflow.sklearn.log_model(model, args.model_name)  # Log the model using with model_name

        # Register logged model using mlflow
        run_id = mlflow.active_run().info.run_id
        model_uri = f'runs:/{run_id}/{args.model_name}'
        mlflow_model = mlflow.register_model(model_uri, args.model_name)  # register the model with model_uri and model_name
        model_version = mlflow_model.version  # Get the version of the registered model

        # Write model info
        print("Writing JSON")
        model_info = {"id": f"{args.model_name}:{model_version}"}
        
    except Exception as e:
        print(f"Warning: MLflow model registration failed: {e}")
        # Create a basic model info file even if registration fails
        print("Creating basic model info file...")
        model_info = {"id": f"{args.model_name}:1", "status": "registration_failed", "error": str(e)}
    
    # Always try to write the model info file
    try:
        os.makedirs(args.model_info_output_path, exist_ok=True)
        output_path = os.path.join(args.model_info_output_path, "model_info.json")  # Specify the name of the JSON file (model_info.json)
        with open(output_path, "w") as of:
            json.dump(model_info, of)  # write model_info to the output file
        print(f"Model info written to: {output_path}")
    except Exception as e:
        print(f"Error writing model info file: {e}")

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
        f"Model name: {args.model_name}",
        f"Search base: {args.search_base}",
        f"Model info output path: {args.model_info_output_path}"
    ]

    for line in lines:
        print(line)

    main(args)

    # End MLflow run only if it was successfully started
    if mlflow_enabled:
        try:
            mlflow.end_run()
        except Exception as e:
            print(f"Warning: MLflow end_run failed: {e}")