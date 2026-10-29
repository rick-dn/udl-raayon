#!/usr/bin/env python3
"""
Fine-tune YOLO11n on VisDrone for drone-perspective person detection.

Hardware: RTX 3050 6GB VRAM
Dataset: VisDrone2019-DET (auto-downloaded)
Target: Pedestrian + People classes only

Usage:
    python train_visdrone_person.py                    # full training
    python train_visdrone_person.py --epochs 10        # quick test
    python train_visdrone_person.py --export tflite     # export after training
    python train_visdrone_person.py --export engine     # export TensorRT (Jetson)

After training, best model saved to: runs/detect/visdrone_person/weights/best.pt
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def train(epochs: int = 100, batch: int = 8, imgsz: int = 640, resume: bool = False):
    """Fine-tune YOLO11n on VisDrone."""

    model = YOLO("yolo11n.pt")  # COCO-pretrained nano model

    results = model.train(
        data="VisDrone.yaml",       # Ultralytics built-in - auto-downloads dataset
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=0,                   # GPU 0 (RTX 3050)
        amp=True,                   # Mixed precision - saves ~30% VRAM
        patience=20,                # Early stopping if no improvement for 20 epochs
        save_period=10,             # Checkpoint every 10 epochs
        project="runs/detect",
        name="visdrone_person",
        exist_ok=True,
        resume=resume,

        # VisDrone-specific settings
        classes=[0, 1],             # Only pedestrian (0) and people (1)
        
        # Augmentation (good for drone perspective variance)
        degrees=15.0,               # Rotation - drones see various angles
        scale=0.7,                  # Scale variation - altitude changes
        mosaic=1.0,                 # Mosaic augmentation
        mixup=0.1,                  # Light mixup
        copy_paste=0.1,             # Copy-paste for small objects
    )
    
    return results


def validate(model_path: str = "runs/detect/visdrone_person/weights/best.pt"):
    """Validate trained model."""
    model = YOLO(model_path)
    metrics = model.val(
        data="VisDrone.yaml",
        classes=[0, 1],
        imgsz=640,
        batch=8,
    )
    print(f"\nmAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    return metrics


def export_model(model_path: str, fmt: str = "tflite"):
    """Export to deployment format."""
    model = YOLO(model_path)

    if fmt == "tflite":
        # For Raspberry Pi
        model.export(format="tflite", imgsz=320, int8=True)
        print("Exported TFLite model (INT8) for Pi deployment")

    elif fmt == "engine":
        # For Jetson (run this ON the Jetson)
        model.export(format="engine", imgsz=640, half=True)
        print("Exported TensorRT engine (FP16) for Jetson deployment")

    elif fmt == "onnx":
        model.export(format="onnx", imgsz=640)
        print("Exported ONNX model")

    else:
        raise ValueError(f"Unknown format: {fmt}. Use tflite, engine, or onnx")


def main():
    parser = argparse.ArgumentParser(description="Train YOLO11n on VisDrone (person-only)")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs (default: 100)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (default: 8, reduce if OOM)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted training")
    parser.add_argument("--validate", action="store_true", help="Run validation only")
    parser.add_argument("--export", type=str, choices=["tflite", "engine", "onnx"],
                        help="Export trained model to format")
    parser.add_argument("--model", type=str, default="runs/detect/visdrone_person/weights/best.pt",
                        help="Model path for validation/export")
    args = parser.parse_args()

    if args.validate:
        validate(args.model)
    elif args.export:
        export_model(args.model, args.export)
    else:
        print("=" * 60)
        print("YOLO11n VisDrone Person Detection Training")
        print(f"  Epochs: {args.epochs}")
        print(f"  Batch:  {args.batch}")
        print(f"  ImgSz:  {args.imgsz}")
        print(f"  Classes: pedestrian, people")
        print(f"  GPU:    RTX 3050 6GB")
        print("=" * 60)
        train(args.epochs, args.batch, args.imgsz, args.resume)
        print("\nTraining complete!")
        print(f"Best model: runs/detect/visdrone_person/weights/best.pt")
        print(f"\nNext steps:")
        print(f"  Validate:    python train_visdrone_person.py --validate")
        print(f"  Export Pi:   python train_visdrone_person.py --export tflite")
        print(f"  Export ONNX: python train_visdrone_person.py --export onnx")


if __name__ == "__main__":
    main()
