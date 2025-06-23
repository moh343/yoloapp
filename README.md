# YOLOv8 Dataset - yolo_prd

This dataset is exported from the image labeling project "yolo_prd" in YOLOv8 format.

## Dataset Structure

- `train/images/`: Contains training images (70% of dataset)
- `train/labels/`: Contains YOLO format annotation files for training
- `valid/images/`: Contains validation images (20% of dataset)
- `valid/labels/`: Contains YOLO format annotation files for validation
- `test/images/`: Contains test images (10% of dataset)
- `test/labels/`: Contains YOLO format annotation files for testing
- `data.yaml`: Configuration file for YOLOv8 training
- `train.py`: Script to train a YOLOv8 model on this dataset (GPU/CUDA)
- `train_cpu.py`: Script to train a YOLOv8 model using CPU only
- `predict.py`: Script to run predictions on test images
- `requirements.txt`: Python dependencies
- `pipinstallreq.bat`: Batch file to install Python dependencies (Windows)
- `runtrain.bat`: Batch file to run the GPU training script (Windows)
- `runtrain_cpu.bat`: Batch file to run the CPU training script (Windows)
- `runprediction.bat`: Batch file to run the prediction script (Windows)

## Dataset Configuration

The `data.yaml` file contains:
- Paths to train, validation, and test sets
- Number of classes (`nc`)
- Class names (`names`)
- ThowlLabel metadata

## Training Instructions

### GPU Training (Recommended for Speed)

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the GPU training script:
   ```bash
   python train.py
   ```

### CPU Training (No CUDA Required)

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the CPU training script (identical parameters to GPU version):
   ```bash
   python train_cpu.py
   ```

3. Optional arguments:
   - `--data`: Path to data.yaml file (default: data.yaml)
   - `--epochs`: Number of training epochs (default: 100)
   - `--batch-size`: Batch size for training (default: 16)
   - `--img-size`: Input image size (default: 640)
   - `--model-size`: Model size (n=nano, s=small, m=medium, l=large, x=xlarge) (default: n)
   - `--mosaic`: Mosaic augmentation probability (default: 1.0)
   - `--mixup`: Mixup augmentation probability (default: 0.5)
   - `--copy-paste`: Copy-paste augmentation probability (default: 0.0)
   - `--degrees`: Image rotation (+/- deg) (default: 0.0)
   - `--translate`: Image translation (+/- fraction) (default: 0.1)
   - `--scale`: Image scale (+/- gain) (default: 0.5)
   - `--shear`: Image shear (+/- deg) (default: 0.0)
   - `--perspective`: Image perspective (+/- fraction) (default: 0.0)
   - `--flipud`: Image flip up-down probability (default: 0.0)
   - `--fliplr`: Image flip left-right probability (default: 0.5)
   - `--hsv-h`: Image HSV-Hue augmentation (fraction) (default: 0.015)
   - `--hsv-s`: Image HSV-Saturation augmentation (fraction) (default: 0.7)
   - `--hsv-v`: Image HSV-Value augmentation (fraction) (default: 0.4)

Example:
```bash
python train.py --epochs 200 --batch-size 32 --model-size m --mosaic 0.8 --mixup 0.3
```

### Using the Batch Files (Windows)

#### GPU Training
1. Double-click `pipinstallreq.bat` to install the required dependencies
2. Double-click `runtrain.bat` to start GPU training with default parameters

#### CPU Training
1. Double-click `pipinstallreq.bat` to install the required dependencies
2. Double-click `runtrain_cpu.bat` to start CPU training with same parameters as GPU version

## Prediction Instructions

### Using the Python Script

1. Make sure you have a trained model (run the training script first)
2. Run the prediction script:
   ```bash
   python predict.py
   ```

3. Optional arguments:
   - `--model`: Path to trained model weights (optional, will auto-detect if not provided)
   - `--data`: Path to data.yaml file (optional, will auto-detect if not provided)
   - `--conf`: Confidence threshold for predictions (default: 0.25)
   - `--save-dir`: Directory to save prediction results (default: predictions)

Example:
```bash
python predict.py --conf 0.3 --save-dir my_predictions
```

### Using the Batch File (Windows)

1. Make sure you have a trained model (run the training script first)
2. Double-click `runprediction.bat` to run predictions with default parameters

## Model Export

The training script automatically exports the model in ONNX format after training. The exported model will be saved in the `runs/detect/train` directory.

## File Locations

- **GPU Training script**: `train.py` - Main script for training the YOLOv8 model with GPU/CUDA
- **CPU Training script**: `train_cpu.py` - Script for training the YOLOv8 model with CPU only
- **Prediction script**: `predict.py` - Script for running predictions on test images
- **Configuration file**: `data.yaml` - Contains dataset configuration and class information
- **Dependencies**: `requirements.txt` - Lists all required Python packages
- **Windows batch files**:
  - `pipinstallreq.bat` - Installs required Python dependencies
  - `runtrain.bat` - Runs the GPU training script with default parameters
  - `runtrain_cpu.bat` - Runs the CPU training script with default parameters
  - `runprediction.bat` - Runs the prediction script with default parameters
- **Training data**:
  - `train/images/` - Training images
  - `train/labels/` - YOLO format annotation files for training
- **Validation data**:
  - `valid/images/` - Validation images
  - `valid/labels/` - YOLO format annotation files for validation
- **Test data**:
  - `test/images/` - Test images
  - `test/labels/` - YOLO format annotation files for testing
- **Trained model**: `runs/detect/train*/weights/best.pt` - Best model weights after training
- **Prediction results**: `predictions/` - Directory where prediction results are saved

## Troubleshooting

1. If you encounter CUDA out of memory errors:
   - Reduce the batch size using `--batch-size`
   - Use a smaller model size (e.g., 'n' instead of 'l')
   - Reduce the image size using `--img-size`

2. If training is slow:
   - Ensure you have a GPU installed
   - Check that CUDA is properly installed
   - Try using a smaller model size

3. If you get import errors:
   - Make sure all dependencies are installed: `pip install -r requirements.txt`
   - Check your Python version (3.7+ required)

4. If the batch files don't work:
   - Make sure Python is installed and added to your PATH
   - Try running the Python scripts directly instead
