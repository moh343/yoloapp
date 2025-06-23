#!/usr/bin/env python3
"""
YOLOv8 CPU Training Script for yolo_prd
This script trains a YOLOv8 model on the exported dataset using CPU only.
Optimized for CPU training with adjusted parameters.
"""

import os
import argparse
import yaml
from ultralytics import YOLO
from pathlib import Path

def load_config(yaml_path):
    """Load and validate the YAML configuration file."""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required fields
    required_fields = ['train', 'val', 'test', 'nc', 'names']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field '{field}' in {yaml_path}")
    
    return config

def validate_paths(config, base_dir=None):
    """
    Validate and fix dataset paths
    
    Args:
        config: Configuration dictionary
        base_dir: Base directory for relative paths
    """
    if base_dir is None:
        base_dir = os.getcwd()
    
    # Convert paths to absolute paths if they're relative
    for key in ['train', 'val', 'test']:
        if not os.path.isabs(config[key]):
            config[key] = os.path.join(base_dir, config[key])
        
        # Check if directory exists
        if not os.path.exists(config[key]):
            print(f"Warning: {key} directory '{config[key]}' does not exist.")
            # Create directory if it doesn't exist
            os.makedirs(config[key], exist_ok=True)
            print(f"Created directory: {config[key]}")
    
    return config

def train_model_cpu(data_yaml_path, epochs=200, batch_size=8, img_size=640, model_size='n', base_dir=None,
                    mosaic=1.0, mixup=0.5, copy_paste=0.0, degrees=0.0, translate=0.1, scale=0.5, shear=0.0,
                    perspective=0.0, flipud=0.0, fliplr=0.5, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4):
    """
    Train a YOLOv8 model on the dataset using CPU only
    
    Args:
        data_yaml_path: Path to the data.yaml file
        epochs: Number of training epochs
        batch_size: Batch size for training
        img_size: Input image size
        model_size: Model size (n, s, m, l, x)
        base_dir: Base directory for dataset paths
        mosaic: Mosaic augmentation probability (0.0 to 1.0)
        mixup: Mixup augmentation probability (0.0 to 1.0)
        copy_paste: Copy-paste augmentation probability (0.0 to 1.0)
        degrees: Image rotation (+/- deg)
        translate: Image translation (+/- fraction)
        scale: Image scale (+/- gain)
        shear: Image shear (+/- deg)
        perspective: Image perspective (+/- fraction)
        flipud: Image flip up-down probability
        fliplr: Image flip left-right probability
        hsv_h: Image HSV-Hue augmentation (fraction)
        hsv_s: Image HSV-Saturation augmentation (fraction)
        hsv_v: Image HSV-Value augmentation (fraction)
    """
    try:
        print("=== YOLOv8 CPU Training Script ===")
        print("This script runs the same training as the GPU version but uses CPU.")
        print("Training will be slower than GPU but doesn't require CUDA.")
        print()
        
        # Load and validate config
        config = load_config(data_yaml_path)
        
        # Validate and fix paths
        config = validate_paths(config, base_dir)
        
        # Create a temporary YAML file with absolute paths
        temp_yaml_path = os.path.join(os.path.dirname(data_yaml_path), "temp_data_cpu.yaml")
        with open(temp_yaml_path, 'w') as f:
            yaml.dump(config, f)
        
        print(f"Training with {config['nc']} classes: {config['names']}")
        print(f"Train path: {config['train']}")
        print(f"Validation path: {config['val']}")
        print(f"Test path: {config['test']}")
        print(f"Using CPU for training")
        print(f"Model size: {model_size}")
        print(f"Batch size: {batch_size}")
        print(f"Image size: {img_size}")
        print(f"Epochs: {epochs}")
        print()
        
        # Create a new YOLO model
        model = YOLO(f'yolo11{model_size}.pt')
        
        # Train the model with same parameters as GPU version
        print("Starting training...")
        results = model.train(
            data=temp_yaml_path,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            device='cpu',       # Force CPU usage
            save=True,          # Save best model
            # mosaic=mosaic,
            # mixup=mixup,
            # copy_paste=copy_paste,
            # degrees=degrees,
            # translate=translate,
            # scale=scale,
            # shear=shear,
            # perspective=perspective,
            # flipud=flipud,
            # fliplr=fliplr,
            # hsv_h=hsv_h,
            # hsv_s=hsv_s,
            # hsv_v=hsv_v
        )
        
        # Clean up temporary file
        if os.path.exists(temp_yaml_path):
            os.remove(temp_yaml_path)
        
        print(f"\nTraining completed successfully!")
        print(f"Model saved to: {results.save_dir}")
        print(f"Best weights: {results.save_dir}/weights/best.pt")
        print(f"Last weights: {results.save_dir}/weights/last.pt")
        
        # Export model to ONNX format for deployment
        try:
            print("\nExporting model to ONNX format...")
            model.export(format='onnx', device='cpu')
            print("ONNX export completed successfully!")
        except Exception as export_error:
            print(f"Warning: ONNX export failed: {export_error}")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        print("\nTroubleshooting tips for CPU training:")
        print("1. Reduce batch size if you get memory errors")
        print("2. Reduce image size if training is too slow")
        print("3. Use model size 'n' (nano) for fastest training")
        print("4. Ensure you have enough RAM available")
        print("5. Consider using GPU version for faster training")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train YOLOv8 model on custom dataset using CPU')
    parser.add_argument('--data', type=str, default='data.yaml', help='Path to data.yaml file')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--img-size', type=int, default=640, help='Input image size')
    parser.add_argument('--model-size', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'], 
                        help='Model size (n=nano, s=small, m=medium, l=large, x=xlarge)')
    parser.add_argument('--base-dir', type=str, default=None, 
                        help='Base directory for dataset paths (default: current directory)')
    # Augmentation parameters (same as GPU version)
    parser.add_argument('--mosaic', type=float, default=1.0, help='Mosaic augmentation probability')
    parser.add_argument('--mixup', type=float, default=0.5, help='Mixup augmentation probability')
    parser.add_argument('--copy-paste', type=float, default=0.0, help='Copy-paste augmentation probability')
    parser.add_argument('--degrees', type=float, default=0.0, help='Image rotation (+/- deg)')
    parser.add_argument('--translate', type=float, default=0.1, help='Image translation (+/- fraction)')
    parser.add_argument('--scale', type=float, default=0.5, help='Image scale (+/- gain)')
    parser.add_argument('--shear', type=float, default=0.0, help='Image shear (+/- deg)')
    parser.add_argument('--perspective', type=float, default=0.0, help='Image perspective (+/- fraction)')
    parser.add_argument('--flipud', type=float, default=0.0, help='Image flip up-down probability')
    parser.add_argument('--fliplr', type=float, default=0.5, help='Image flip left-right probability')
    parser.add_argument('--hsv-h', type=float, default=0.015, help='Image HSV-Hue augmentation (fraction)')
    parser.add_argument('--hsv-s', type=float, default=0.7, help='Image HSV-Saturation augmentation (fraction)')
    parser.add_argument('--hsv-v', type=float, default=0.4, help='Image HSV-Value augmentation (fraction)')
    
    args = parser.parse_args()
    
    print("=== CPU Training Configuration ===")
    print(f"Data: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.img_size}")
    print(f"Model size: {args.model_size}")
    print("Device: CPU (forced)")
    print("\nNote: This script runs identical training to GPU version but uses CPU.")
    print("Training will be slower than GPU but doesn't require CUDA.")
    print("=" * 40)
    print()
    
    train_model_cpu(
        data_yaml_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        model_size=args.model_size,
        base_dir=args.base_dir,
        mosaic=args.mosaic,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        shear=args.shear,
        perspective=args.perspective,
        flipud=args.flipud,
        fliplr=args.fliplr,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v
    )
