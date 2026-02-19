# Layout Analysis and OCR Pipeline

This repository contains a robust pipeline for processing historical PDFs, extracting images, performing Layout Analysis (YALTAi/YOLO), and Optical Character Recognition (Kraken).

## Quick Start

### 1. Prerequisites
-   **Linux Environment** (e.g., Ubuntu, Debian) or **WSL 2** on Windows.
-   **Conda** installed.
-   **GPU** recommended for faster processing.

### 2. Setup
Ensure your environment is set up and `config.py` is configured with your paths.

```bash
# Activate environment
conda activate rtk_env
```

### 3. Running the Pipeline

To process all PDFs in your configured `PDF_DIR`:

```bash
# Run with 8 parallel workers for faster extraction
python run_pipeline.py --workers 8
```

To recover/retry only missing books:

```bash
python recover_books.py
```

### 5. TEI Conversion

After processing is complete, you can convert the ALTO XML files to TEI format:

```bash
python convert_to_tei.py
```

This will:
1.  Generate a `tei_manifest.csv`.
2.  Run `ladas2tei` to produce TEI files.

## Directory Structure

| Directory | Description |
| :--- | :--- |
| `processed_images/` | Temporary storage for extracted images. |
| `processed_images_xmls/` | Final output ALTO XML files. |
| `models/` | Trained models for Layout Analysis and OCR. |
| `pdfs/` | Source PDF files (External path configured in `config.py`). |
| `tei_output/` | (Optional) Destination for TEI files if configured by ladas2tei. |

## Configuration

All paths and configuration constants are located in `config.py`. 
**You MUST edit `config.py` if you move this repository or change your PDF source directory.**

## Scripts

-   **`run_pipeline.py`**: The main driver script. Orchestrates PDF extraction and parallel pipeline execution.
-   **`recover_books.py`**: A recovery tool to identify books that were partially processed or missed and finish them.
-   **`monitor.py`**: A utility to check the status of all books (Source vs Completed vs In-Progress).
-   **`convert_to_tei.py`**: Converts the ALTO XML output to TEI format.
-   **`pipeline_worker.py`**: The internal worker script.

## Troubleshooting

-   **Logs**: Check `*.log` files in the root directory for per-book execution logs.
-   **Stalled Extraction**: If images aren't appearing, check `monitor.py`. Large PDFs can take 10+ minutes to extract.
