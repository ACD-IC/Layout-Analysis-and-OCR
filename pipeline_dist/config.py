
from pathlib import Path
import os

# --- Paths ---

# The directory containing the source PDF files.
# NOTE: This path is set for WSL access to the Windows file system.
# If running on native Linux, change this to your PDF directory.
PDF_DIR = Path("/mnt/c/Users/lucia/Downloads/pdfs/pdfs")

# The directory where images extracted from PDFs will be stored temporarily.
PROCESSED_IMAGES_DIR = Path("processed_images")

# The directory where the final ALTO XML output files will be stored.
XML_OUTPUT_DIR = Path("processed_images_xmls")

# --- Environment ---

# The python executable to use for running the pipeline.
# Ensure this points to the python in your conda environment.
PYTHON_CMD = "/home/lucian/miniconda3/envs/rtk_env/bin/python"

# --- Models ---

# Path to the YOLO model for layout analysis/segmentation.
YOLO_MODEL = "my_finetune_project/run_ladas_1280_l_v14/weights/best.pt"

# Path to the Kraken line segmentation model.
LINE_MODEL = "models/blla.mlmodel"

# Path to the Kraken OCR model.
OCR_MODEL = "models/catmus-print-fondue-large.mlmodel"

# --- Scripts ---

# The internal worker script that runs the pipeline on a folder of images.
PIPELINE_SCRIPT = "pipeline_worker.py"
