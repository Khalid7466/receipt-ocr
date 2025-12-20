"""
YOLOv11n training script for CORD receipt dataset
"""

from ultralytics import YOLO
import os

def train_yolo_cord():
    """Train YOLOv11n on CORD dataset for receipt field detection."""
    
    # Path to data configuration
    data_yaml = 'data/cord/data.yaml'
    
    # Load pretrained YOLOv11n model
    print("Loading YOLOv11n pretrained model...")
    model = YOLO('yolo11n.pt')
    
    # Train the model
    print("\n" + "="*60)
    print("Starting YOLOv11n training on CORD dataset")
    print("="*60)
    
    results = model.train(
        data=data_yaml,
        epochs=100,
        imgsz=640,
        device=0,  # GPU device (0), or use device=-1 for CPU
        batch=16,  # Adjust based on GPU memory
        patience=20,
        save=True,
        project='runs/yolo11n_cord',  # Output directory
        name='detect',
        verbose=True,
        plots=True,  # Generate training plots
        seed=42,
    )
    
    print("\n" + "="*60)
    print("Training Results Summary")
    print("="*60)
    print(f"Best model saved to: {results.save_dir}")
    print(f"Model path: {model.trainer.best}")
    
    # Validate the model
    print("\n" + "="*60)
    print("Running validation...")
    print("="*60)
    val_results = model.val()
    
    print("\nValidation complete!")
    print(f"mAP50: {val_results.box.map50:.4f}")
    print(f"mAP50-95: {val_results.box.map:.4f}")
    
    # Test predictions (optional)
    print("\n" + "="*60)
    print("Running inference on test set...")
    print("="*60)
    pred_results = model.predict(
        source='data/cord/raw/test',
        conf=0.25,
        save=True,
        project='runs/yolo11n_cord',
        name='test_predictions',
    )
    
    print("\nTraining and evaluation complete!")
    print(f"All results saved to: runs/yolo11n_cord/")
    

if __name__ == '__main__':
    train_yolo_cord()
