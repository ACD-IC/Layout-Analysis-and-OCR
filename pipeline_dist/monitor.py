
import config
from pathlib import Path
import os
import time

def normalize_name(name):
    return name.replace('(', '').replace(')', '').replace("'", '').replace(' ', '_')

def monitor():
    print("--- Layout Analysis & OCR Pipeline Monitor ---")
    
    if not config.PDF_DIR.exists():
        print(f"Error: PDF directory not found at {config.PDF_DIR}")
        return

    # 1. Source PDFs
    pdfs = list(config.PDF_DIR.glob("*.pdf")) + list(config.PDF_DIR.glob("*.PDF"))
    print(f"Total Source PDFs: {len(pdfs)}")
    
    # 2. Completed Books
    if config.XML_OUTPUT_DIR.exists():
        books_done = set(p.name for p in config.XML_OUTPUT_DIR.iterdir() if p.is_dir())
    else:
        books_done = set()
    print(f"Total Completed Books: {len(books_done)}")
    
    # 3. Active/In-Progress Books
    if config.PROCESSED_IMAGES_DIR.exists():
        books_in_progress = [p for p in config.PROCESSED_IMAGES_DIR.iterdir() if p.is_dir()]
    else:
        books_in_progress = []
        
    print(f"Books currently processing (in {config.PROCESSED_IMAGES_DIR.name}): {len(books_in_progress)}")
    
    # Detail on in-progress
    if books_in_progress:
        print("\nActive Processing Details:")
        print(f"{'Book Name':<50} | {'Images':<10} | {'XMLs':<10} | {'Status'}")
        print("-" * 90)
        
        for book_path in books_in_progress:
            book_name = book_path.name
            
            # Count files (approximate check)
            # Using glob is slow for huge folders, but okay for monitoring
            # For speed, we can use os.scandir
            n_png = 0
            n_xml = 0
            try:
                with os.scandir(book_path) as entries:
                    for entry in entries:
                        if entry.name.endswith('.png'): n_png += 1
                        elif entry.name.endswith('.xml'): n_xml += 1
            except Exception:
                pass
            
            status = "Extracting/Idle"
            if n_xml > 0:
                status = "Analyzing/Refining"
            
            print(f"{book_name[:47]+'...' if len(book_name)>47 else book_name:<50} | {n_png:<10} | {n_xml:<10} | {status}")

    # 4. Missing Books
    print("\nMissing / To-Do:")
    missing_count = 0
    for pdf in pdfs:
        clean_name = normalize_name(pdf.stem)
        
        is_done = clean_name in books_done
        is_processing = any(b.name == clean_name for b in books_in_progress)
        
        if not is_done and not is_processing:
            print(f" - {pdf.name}")
            missing_count += 1
            
    if missing_count == 0:
        print(" (None - All books are done or processing)")
    else:
        print(f"Total Missing: {missing_count}")

if __name__ == "__main__":
    monitor()
