# Assuming you have ultralytics installed: pip install ultralytics

from ultralytics import YOLO
import os

# --- 1. SET UP PATHS AND PARAMETERS ---
# CRITICAL: Path to the model you want to fine-tune.
LOCAL_WEIGHTS_PATH = 'models/ladas-1280-l.pt'

# CRITICAL: Path to your COMBINED data.yaml (Human-corrected labels + Rehearsal labels)
DATA_YAML_PATH = 'RIPA-ft/data.yaml' 

# --- Define CRITICAL Fine-Tuning Parameters ---

# CORRECTION 1: Match the image size to your model's name (ladas-1280-l.pt)
IMG_SIZE = 1280

# CORRECTION 2: Use a very low learning rate for fine-tuning to avoid catastrophic forgetting.
# 1e-3 is often still too high; 1e-4 is a safer starting point.
LEARNING_RATE = 1e-4 

EPOCHS = 100            # Number of epochs to train for
BATCH_SIZE = 6           # Adjust based on your GPU VRAM (1280px images use more VRAM)
PATIENCE = 25             # Stop training if validation mAP doesn't improve for 25 epochs
FREEZE_LAYERS = 10        # CORRECTION 3: Freeze the first 10 backbone layers (YOLOv8 'n'/'s'/'m'/'l' have 10 backbone layers before the neck)

# --- 2. LOAD THE LOCAL MODEL WEIGHTS ---
print(f"Loading model weights from: {LOCAL_WEIGHTS_PATH}")
# The YOLO class automatically infers the model architecture.
try:
    model = YOLO(LOCAL_WEIGHTS_PATH)
except FileNotFoundError:
    print(f"\nERROR: Model file not found at {LOCAL_WEIGHTS_PATH}")
    raise

# --- 3. INITIATE TRANSFER LEARNING (Fine-Tuning) ---
print("\nStarting Transfer Learning (Fine-Tuning)...")

results = model.train(
    data=DATA_YAML_PATH,     # Path to data.yaml
    epochs=EPOCHS,           # Number of epochs
    imgsz=IMG_SIZE,          # Image size (Corrected to 1280)
    batch=BATCH_SIZE,        # Batch size
    
    # --- Applying Fine-Tuning Corrections ---
    lr0=LEARNING_RATE,       # Activated: Use the low learning rate (Corrected to 1e-4)
    freeze=FREEZE_LAYERS,    # Activated: Freeze the backbone
    patience=PATIENCE,       # Activated: Enable early stopping
    
    # --- Run Management (Recommended) ---
    project='my_finetune_project', # Project name for saving results
    name='run_ladas_1280_l_v1'     # Experiment name
)

print("\nTransfer Learning Complete!")
# The final weights and results are saved in the project/name directory.
print(f"Results saved to: {model.trainer.save_dir}")

# --- Sample input data structure (data.yaml content) ---
"""
# data.yaml example structure
# This file MUST point to your combined training data (Corrected + Rehearsal)

train: ../fine_tuning_dataset/images/train/
val: ../fine_tuning_dataset/images/val/
# test: ../fine_tuning_dataset/images/test/  # Optional

# Class names must match your original model
nc: 8 
names: ['text_block', 'header', 'folio', 'caption', 'footnote', 'figure', 'table', 'separator']
"""
