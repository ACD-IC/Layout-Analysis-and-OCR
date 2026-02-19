
import os
import shutil
import argparse
import subprocess
import config

def process_missing_books(dry_run=False):
    """
    Identify books in processed_images that are not in processed_images_xmls.
    Check if they are 'done' (high XML count) or need processing.
    
    Args:
        dry_run (bool): If True, only simulate actions.
    """
    if not config.PROCESSED_IMAGES_DIR.exists():
        print(f"Directory not found: {config.PROCESSED_IMAGES_DIR}")
        return

    # Get list of books in processed_images
    books_source = set(p.name for p in config.PROCESSED_IMAGES_DIR.iterdir() if p.is_dir())
    
    # Get list of books in processed_images_xmls
    if config.XML_OUTPUT_DIR.exists():
        books_done = set(p.name for p in config.XML_OUTPUT_DIR.iterdir() if p.is_dir())
    else:
        books_done = set()
        
    missing_books = books_source - books_done
    print(f"Found {len(missing_books)} books in processed_images but not in processed_images_xmls.")

    processed_but_not_moved = []
    actually_unprocessed = []

    for book_name in missing_books:
        book_path = config.PROCESSED_IMAGES_DIR / book_name
        
        xmls = list(book_path.glob("*.xml"))
        pngs = list(book_path.glob("*.png"))
        
        # If no PNGs, skip (maybe empty folder?)
        if not pngs:
            print(f"Skipping {book_name}: No PNGs found.")
            continue
            
        ratio = len(xmls) / len(pngs) if pngs else 0
        
        if ratio >= 0.9:
            processed_but_not_moved.append(book_name)
        else:
            actually_unprocessed.append(book_name)

    print(f"Books processed but not moved (Ratio >= 0.9): {len(processed_but_not_moved)}")
    print(f"Books needing pipeline processing (Ratio < 0.9): {len(actually_unprocessed)}")

    # 1. Handle "Processed but not moved"
    if processed_but_not_moved:
        print("\n--- Moving XMLs for already processed books ---")
        for book_name in processed_but_not_moved:
            src_dir = config.PROCESSED_IMAGES_DIR / book_name
            dest_dir = config.XML_OUTPUT_DIR / book_name
            
            if dry_run:
                print(f"[DRY RUN] Would move XMLs from {src_dir} to {dest_dir} and delete processed_images source folder")
            else:
                print(f"Processing {book_name}...")
                dest_dir.mkdir(parents=True, exist_ok=True)
                moved_count = 0
                for xml in src_dir.glob("*.xml"):
                    shutil.move(str(xml), dest_dir / xml.name)
                    moved_count += 1
                
                print(f"Moved {moved_count} XMLs.")
                
                # Check if only PNGs left, generally safe to delete entire folder as we moved the valuable output
                remaining_xmls = list(src_dir.glob("*.xml"))
                if not remaining_xmls:
                     print(f"Deleting source directory: {src_dir}")
                     shutil.rmtree(src_dir)
                else:
                     print(f"WARNING: XMLs remaining in {src_dir}, not deleting.")

    # 2. Handle "Actually unprocessed"
    if actually_unprocessed:
        print("\n--- Running pipeline for unprocessed books ---")
        for book_name in actually_unprocessed:
            image_dir = config.PROCESSED_IMAGES_DIR / book_name
            dest_dir = config.XML_OUTPUT_DIR / book_name
            
            if dry_run:
                print(f"[DRY RUN] Would run pipeline on {image_dir}, then move XMLs to {dest_dir}")
            else:
                print(f"[{book_name}] Starting pipeline processing...")
                
                # Run pipeline
                # The pipeline script takes a glob pattern OR directory.
                cmd = [config.PYTHON_CMD, config.PIPELINE_SCRIPT, "--input", str(image_dir), "--batch-size", "8"]
                log_file = f"{book_name}_recovery.log"
                
                try:
                    with open(log_file, "w") as log: 
                        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
                    print(f"[{book_name}] Pipeline finished successfully.")
                    
                    # Move XMLs
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    # Pipeline generates XMLs in the same folder as PNGs
                    xmls_after = list(image_dir.glob("*.xml"))
                    moved_count = 0
                    for xml in xmls_after:
                        shutil.move(str(xml), dest_dir / xml.name)
                        moved_count += 1
                    
                    print(f"[{book_name}] Moved {moved_count} XMLs.")
                    
                    if moved_count > 0:
                         shutil.rmtree(image_dir)
                         print(f"[{book_name}] Deleted source images.")
                    else:
                        print(f"[{book_name}] WARNING: No XMLs produced.")

                except subprocess.CalledProcessError as e:
                    print(f"[{book_name}] FAILED (Pipeline Error: {e})")
                except Exception as e:
                    print(f"[{book_name}] FAILED (Error: {e})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recover and run pipeline on missing or partially processed books.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without executing")
    args = parser.parse_args()
    
    process_missing_books(dry_run=args.dry_run)
