
import os
import sys
import shutil
import time
from pathlib import Path
import fitz # PyMuPDF
import argparse
import subprocess
import concurrent.futures
import config

def normalize_name(name):
    """Normalize PDF name to folder name convention."""
    return name.replace('(', '').replace(')', '').replace("'", '').replace(' ', '_')

def extract_pdf(pdf_path, output_dir, dpi=300):
    """
    Extract PDF pages to PNGs.
    
    Args:
        pdf_path (Path): Path to the source PDF file.
        output_dir (Path): Directory where PNGs should be saved.
        dpi (int): Resolution for extraction.
    
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        doc = fitz.open(pdf_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Suppress per-page print for parallel execution cleanliness
        for i, page in enumerate(doc):
            output_file = output_dir / f"page_{i+1:04d}.png"
            # Only save if not exists (resume capability)
            if not output_file.exists():
                pix = page.get_pixmap(dpi=dpi)
                pix.save(output_file)
        
        doc.close()
        return True
    except Exception as e:
        print(f"Error extracting {pdf_path}: {e}")
        return False

def move_xmls(source_dir, dest_dir):
    """
    Move XML files from source to dest.
    
    Args:
        source_dir (Path): Source directory containing XMLs.
        dest_dir (Path): Destination directory.
        
    Returns:
        int: Number of XML files moved.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    xmls = list(source_dir.glob("*.xml"))
    moved_count = 0
    for xml in xmls:
        shutil.move(str(xml), dest_dir / xml.name)
        moved_count += 1
    return moved_count

def process_book_task(pdf_path, dry_run=False):
    """
    Worker function for processing a single book.
    
    Steps:
    1. Extract images from PDF to processed_images/<book_name>
    2. Run pipeline script (YALTAi + Kraken) via subprocess
    3. Move resulting XMLs to processed_images_xmls/<book_name>
    4. Clean up images
    
    Args:
        pdf_path (Path): Path to the PDF file.
        dry_run (bool): If True, only simulate actions.
        
    Returns:
        str: Status message.
    """
    book_name = pdf_path.stem
    folder_name = normalize_name(book_name)
    
    image_dir = config.PROCESSED_IMAGES_DIR / folder_name
    xml_dir = config.XML_OUTPUT_DIR / folder_name
    
    # Check if already done
    if xml_dir.exists() and any(xml_dir.glob("*.xml")):
        return f"{book_name}: SKIPPED (Already processed)"
    
    if dry_run:
        return f"{book_name}: DRY RUN (Would process)"

    print(f"[{book_name}] Starting processing...")

    # 1. Extraction
    if not image_dir.exists() or not any(image_dir.glob("*.png")):
        success = extract_pdf(pdf_path, image_dir)
        if not success:
            return f"{book_name}: FAILED (Extraction)"
    else:
        print(f"[{book_name}] Images already extracted")

    # 2. Pipeline Processing (Subprocess)
    # run_pipeline expects a glob string or directory
    cmd = [config.PYTHON_CMD, config.PIPELINE_SCRIPT, "--input", f"{image_dir}/*.png", "--batch-size", "8"]
    
    log_file = f"{folder_name}.log"
    
    try:
        with open(log_file, "w") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    except subprocess.CalledProcessError as e:
        return f"{book_name}: FAILED (Pipeline Error: {e})"
    except Exception as e:
        return f"{book_name}: FAILED (Other usage error: {e})"

    # 3. Organization & Cleanup
    moved = move_xmls(image_dir, xml_dir)
    
    if moved > 0:
        # Delete images
        shutil.rmtree(image_dir)
        return f"{book_name}: SUCCESS (Moved {moved} XMLs, Processed images deleted)"
    else:
        return f"{book_name}: WARNING (No XMLs produced)"

def main():
    parser = argparse.ArgumentParser(description="Run Layout Analysis and OCR pipeline on PDFs.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be done without executing.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of books to process (0 for all).")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers.")
    args = parser.parse_args()
    
    if not config.PDF_DIR.exists():
        print(f"PDF Directory not found: {config.PDF_DIR}")
        print("Please check config.py")
        return

    pdfs = list(config.PDF_DIR.glob("*.pdf")) + list(config.PDF_DIR.glob("*.PDF"))
    print(f"Found {len(pdfs)} PDFs in source.")
    
    books_to_process = pdfs
    if args.limit > 0:
        # If limiting, try to pick unprocessed ones first
        todos = []
        for pdf in pdfs:
            bn = pdf.stem
            fn = normalize_name(bn)
            xd = config.XML_OUTPUT_DIR / fn
            if not (xd.exists() and any(xd.glob("*.xml"))):
                todos.append(pdf)
        books_to_process = todos[:args.limit]
        print(f"Limiting to {len(books_to_process)} unprocessed books.")
    
    print(f"Launching {len(books_to_process)} tasks with {args.workers} workers...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_book = {executor.submit(process_book_task, pdf, args.dry_run): pdf.name for pdf in books_to_process}
        
        for future in concurrent.futures.as_completed(future_to_book):
            book = future_to_book[future]
            try:
                result = future.result()
                print(f"[RESULT] {result}")
            except Exception as exc:
                print(f"[EXCEPTION] {book} generated an exception: {exc}")

if __name__ == "__main__":
    main()
