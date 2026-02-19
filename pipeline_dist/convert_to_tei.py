
import os
import csv
import subprocess
import config

def create_manifest(manifest_file="tei_manifest.csv"):
    """
    Create CSV manifest for LADAS2TEI.
    The manifest points to folders containing ALTO XML files.
    """
    processed_dir = config.XML_OUTPUT_DIR
    
    if not processed_dir.exists():
        print(f"Error: XML directory not found at {processed_dir}")
        return False
        
    # Get all subdirectories (books)
    subdirs = [d.name for d in processed_dir.iterdir() if d.is_dir()]
    subdirs = sorted(subdirs)
    
    if not subdirs:
        print("No processed books found in processed_images_xmls.")
        return False
        
    print(f"Found {len(subdirs)} books with XMLs.")
    
    # Columns: file_name,number,title,date,author,publisher
    with open(manifest_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "number", "title", "date", "author", "publisher"])
        
        for idx, book_name in enumerate(subdirs, 1):
            # LADAS2TEI expects the directory containing the XMLs as 'file_name'
            # We use absolute path to be safe, or relative if run from root.
            # config.XML_OUTPUT_DIR is a Path object ("processed_images_xmls")
            
            # Important: ladas2tei might expect paths accessible to it.
            # If running in WSL, relative paths should work fine if we are in the root.
            
            full_path = processed_dir / book_name
            # Ensure forward slashes for Linux/WSL compatibility if running from Windows Python (though we are using WSL python)
            file_name_str = str(full_path).replace("\\", "/")
            
            writer.writerow([
                file_name_str,
                idx,
                book_name,
                "2024",
                "Unknown",
                "Unknown"
            ])
            
    print(f"Created manifest: {manifest_file}")
    return True

def run_conversion(manifest_file="tei_manifest.csv"):
    """Run ladas2tei conversion."""
    
    # Derive ladas2tei path from PYTHON_CMD
    # PYTHON_CMD is .../bin/python
    # ladas2tei is .../bin/ladas2tei
    
    bin_dir = os.path.dirname(config.PYTHON_CMD)
    ladas2tei_cmd = os.path.join(bin_dir, "ladas2tei")
    
    cmd = [ladas2tei_cmd, manifest_file]
    
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        print("Conversion completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error running LADAS2TEI: {e}")
    except FileNotFoundError:
        print(f"Error: ladas2tei executable not found at {ladas2tei_cmd}")

if __name__ == "__main__":
    if create_manifest():
        run_conversion()
