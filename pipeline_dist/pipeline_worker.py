
import os
import glob
import time
import sys
from rtk.task import KrakenAltoCleanUpCommand, YALTAiCommand, KrakenRecognizerCommand
import logging
import argparse
import config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def run_pipeline(input_pattern, batch_size=8):
    """
    Run the Layout Analysis and OCR pipeline on images matching the pattern.
    
    Processing Steps:
    1. YALTAi: Layout Analysis (segmentation) using YOLO model.
    2. Post-processing: Wait for file synchronization (critical for WSL/Windows).
    3. ALTO Fixes: Corrects file paths in the generated ALTO XMLs.
    4. Cleanup: Runs KrakenAltoCleanUpCommand.
    5. Kraken: OCR using the configured model.
    
    Args:
        input_pattern (str): Glob pattern or directory path for input images.
        batch_size (int): Number of images to process in a batch.
    """
    
    # Get all PNG files
    if '*' in input_pattern:
        all_files = glob.glob(input_pattern, recursive=True)
        all_files = sorted(all_files)
    else:
        # If directory provided
        if os.path.isdir(input_pattern):
             all_files = sorted(glob.glob(os.path.join(input_pattern, "*.png")))
        else:
             all_files = [input_pattern]
    
    if not all_files:
        print(f"No files found matching {input_pattern}")
        return

    print(f"Processing {len(all_files)} PNGs from {input_pattern}")
    
    # Model checks
    if not os.path.exists(config.YOLO_MODEL):
        print(f"Error: YOLO model not found at {config.YOLO_MODEL}")
        return
    
    # Process in batches
    start_total = time.time()
    
    # Binary paths (Derived from python env, hoping 'yaltai' and 'kraken' are in bin)
    # config.PYTHON_CMD is ".../bin/python", so binaries are in ".../bin/"
    BIN_DIR = os.path.dirname(config.PYTHON_CMD)
    yaltai_bin = os.path.join(BIN_DIR, "yaltai")
    kraken_bin = os.path.join(BIN_DIR, "kraken")

    for i in range(0, len(all_files), batch_size):
        batch = all_files[i:i+batch_size]
        print(f"\nProcessing batch {i//batch_size + 1}/{(len(all_files)-1)//batch_size + 1} ({len(batch)} files)")
        
        # 1. YALTAi (Layout Analysis)
        # print("[Task] Segment (YALTAi)")
        start_yalt = time.time()
        
        try:
            yaltai = YALTAiCommand(
                batch,
                binary=yaltai_bin,
                device="cpu", 
                yolo_model=config.YOLO_MODEL,
                verbose=False, # Reduce noise
                raise_on_error=False,
                allow_failure=True,
                multiprocess=1, # STRICTLY SERIAL to avoid OOM or conflicts
                check_content=False,
                line_model=config.LINE_MODEL
            )
            yaltai.process()
        except Exception as e:
            print(f"Error in YALTAi batch: {e}")

        # Sync/Wait for files (WSL /mnt/c latency fix)
        batch_xml_files = [img.replace('.png', '.xml') for img in batch]
        missing_files = []
        for f in batch_xml_files:
            # Poll for existence up to 2 seconds
            retries = 10
            while retries > 0 and not os.path.exists(f):
                time.sleep(0.2)
                retries -= 1
            if not os.path.exists(f):
                 missing_files.append(f)
        
        if missing_files:
            print(f"Warning: {len(missing_files)} XML files missing after YALTAi: {missing_files}")

        # 2. Fix ALTO paths
        # print("[Task] Fix ALTO file paths")
        for alto_file in batch_xml_files:
            if os.path.exists(alto_file):
                try:
                    with open(alto_file, 'r') as f:
                        content = f.read()
                    
                    img_file = alto_file.replace('.xml', '.png')
                    img_filename = os.path.basename(img_file)
                    
                    # Fix standard pattern
                    content = content.replace(f'<fileName>{img_file}</fileName>', f'<fileName>{img_filename}</fileName>')
                    # remove potential hardcoded paths if any
                    import re
                    content = re.sub(r'<fileName>[^<]*?([^/]+\.png)</fileName>', r'<fileName>\1</fileName>', content)
                    
                    with open(alto_file, 'w') as f:
                        f.write(content)
                except Exception as e:
                    print(f"Error fixing ALTO {alto_file}: {e}")

        # 3. Clean-Up Serialization
        valid_xml_files = [f for f in batch_xml_files if os.path.exists(f)]
        if valid_xml_files:
            try:
                cleanup = KrakenAltoCleanUpCommand(valid_xml_files)
                cleanup.process()
            except Exception as e:
                print(f"Error in cleanup: {e}")
        
        # 4. Kraken (OCR)
        # print("[Task] OCR (Kraken)")
        
        if valid_xml_files:
            try:
                kraken = KrakenRecognizerCommand(
                    valid_xml_files,
                    binary=kraken_bin,
                    device="cpu",
                    model=config.OCR_MODEL,
                    multiprocess=1, # STRICTLY SERIAL
                    check_content=True
                )
                kraken.process()
            except Exception as e:
                print(f"Error in Kraken batch: {e}")
        

    print(f"[Time] Total for {input_pattern}: {time.time() - start_total:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YALTAi and Kraken pipeline on a set of images.")
    parser.add_argument("--input", default="processed_images/**/*.png", help="Glob pattern or directory")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    args = parser.parse_args()
    
    run_pipeline(args.input, batch_size=args.batch_size)
