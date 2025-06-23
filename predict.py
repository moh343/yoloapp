#!/usr/bin/env python3
"""
YOLOv8 Prediction Script for yolo_prd
This script runs inference on test images using a trained YOLOv8 model.
"""

import os
import argparse
import glob
from pathlib import Path
from ultralytics import YOLO
import yaml

def find_latest_model():
    """Find the latest trained model weights in the runs/detect directory."""
    # Look for model weights in the runs/detect directory
    model_paths = glob.glob("runs/detect/train*/weights/best.pt")
    
    if not model_paths:
        raise FileNotFoundError("No trained model found in runs/detect/train*/weights/best.pt")
    
    # Sort by modification time to get the latest model
    latest_model = max(model_paths, key=os.path.getmtime)
    print(f"Using latest model: {latest_model}")
    return latest_model

def find_data_yaml():
    """Find the data.yaml file in the current directory."""
    yaml_path = "data.yaml"
    
    if not os.path.exists(yaml_path):
        # Try to find it in parent directories
        current_dir = os.getcwd()
        while current_dir != os.path.dirname(current_dir):
            yaml_path = os.path.join(current_dir, "data.yaml")
            if os.path.exists(yaml_path):
                break
            current_dir = os.path.dirname(current_dir)
    
    if not os.path.exists(yaml_path):
        raise FileNotFoundError("data.yaml file not found. Please ensure it exists in the project directory.")
    
    print(f"Using data configuration: {yaml_path}")
    return yaml_path

def load_config(yaml_path):
    """Load and validate the YAML configuration file."""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required fields
    required_fields = ['test', 'nc', 'names']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field '{field}' in {yaml_path}")
    
    return config

def run_predictions(model_path=None, data_yaml_path=None, conf_threshold=0.25, save_dir='predictions'):
    """
    Run predictions on test images and save results
    
    Args:
        model_path: Path to the trained model weights (optional, will auto-detect if not provided)
        data_yaml_path: Path to the data.yaml file (optional, will auto-detect if not provided)
        conf_threshold: Confidence threshold for predictions
        save_dir: Directory to save prediction results
    """
    try:
        # Auto-detect model and data.yaml if not provided
        if model_path is None:
            model_path = find_latest_model()
        
        if data_yaml_path is None:
            data_yaml_path = find_data_yaml()
        
        # Load config
        config = load_config(data_yaml_path)
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Load the trained model
        model = YOLO(model_path)
        
        # Get test images directory
        test_dir = config['test']
        if not os.path.isabs(test_dir):
            test_dir = os.path.join(os.path.dirname(data_yaml_path), test_dir)
        
        print(f"Running predictions on images in: {test_dir}")
        print(f"Results will be saved to: {save_dir}")
        
        # Run predictions
        results = model.predict(
            source=test_dir,
            conf=conf_threshold,
            save=True,
            save_txt=True,
            save_conf=True,
            project=save_dir,
            name='predictions'
        )
        
        print(f"\nPrediction completed!")
        print(f"Results saved to: {os.path.join(save_dir, 'predictions')}")
        
        # Print summary of predictions
        for r in results:
            print(f"\nProcessed image: {r.path}")
            print(f"Found {len(r.boxes)} objects")
            for box in r.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = config['names'][class_id]
                print(f"- {class_name}: {confidence:.2f}")
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run YOLOv8 predictions on test images')
    parser.add_argument('--model', type=str, help='Path to trained model weights (optional, will auto-detect if not provided)')
    parser.add_argument('--data', type=str, help='Path to data.yaml file (optional, will auto-detect if not provided)')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold for predictions')
    parser.add_argument('--save-dir', type=str, default='predictions', help='Directory to save prediction results')
    
    args = parser.parse_args()
    
    run_predictions(
        model_path=args.model,
        data_yaml_path=args.data,
        conf_threshold=args.conf,
        save_dir=args.save_dir
    )
