#generate predictions from a manually curated list of problematic pages 
#also sample the same number of correct pages to prevent catastrophic forgetting


# First, ensure you have the necessary libraries installed.
# !pip install ultralytics pandas tqdm

import pandas as pd
import os
import shutil
# No longer need yaml for this script, as we are writing class IDs.
# import yaml 
from ultralytics import YOLO
from tqdm import tqdm

# --- Configuration ---
csv_file_path = 'to_fix.csv'
base_image_directory = 'Editions'
output_directory = 'inference_output_roboflow' # Using a new folder for clarity

# --- Setup ---
os.makedirs(output_directory, exist_ok=True)

# The label map is no longer needed by this script, but is critical for Roboflow.
# You will provide the class names list to Roboflow during dataset setup.
# label_map_path = 'labelmap.yaml'

model = YOLO('models/ladas-1280-l.pt')

# --- Data Processing and Inference ---

try:
    df = pd.read_csv(csv_file_path)
    df.columns = df.columns.str.strip()

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing images"):
        book_name = row['Book']
        page_number = row['Page']

        if not isinstance(book_name, str) or not book_name:
            continue

        try:
            page_number_int = int(float(page_number))
        except (ValueError, TypeError):
            continue

        formatted_page = f"{page_number_int:04d}"
        image_path = os.path.join(base_image_directory, book_name, f"page_{formatted_page}.png")

        if not os.path.exists(image_path):
            continue

        output_basename = f"{book_name}_{page_number_int}"
        output_txt_path = os.path.join(output_directory, f"{output_basename}.txt")

        if os.path.exists(output_txt_path):
            continue

        results = model.predict(image_path, verbose=False)
        result = results[0]

        # --- CORRECTED: Save with integer class_id for Roboflow compatibility ---
        with open(output_txt_path, 'w') as f:
            if result.boxes is not None:
                for i in range(len(result.boxes)):
                    box = result.boxes[i]
                    # Get the integer class ID
                    class_id = int(box.cls[0])
                    
                    coords = box.xywhn[0]
                    x_center, y_center, width, height = map(float, coords)

                    # Write the integer class ID to the file
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        # Copy the original image
        output_image_path = os.path.join(output_directory, f"{output_basename}.png")
        shutil.copy2(image_path, output_image_path)

except FileNotFoundError:
    print(f"Error: The CSV file was not found at '{csv_file_path}'")
except Exception as e:
    print(f"An unexpected error occurred during processing: {e}")

print("\nProcessing complete.")
print(f"Output files are saved in the '{output_directory}' directory.")


# --- Rehearsal Data Generator with Filters ---
# This script identifies images NOT in the 'to_fix.csv' AND NOT in the first 50 pages, 
# samples an equal number (1:1 ratio) to the new data, and generates YOLO labels 
# using the currently trained model for use as a rehearsal buffer.

# First, ensure you have the necessary libraries installed.
# !pip install ultralytics pandas tqdm

import pandas as pd
import os
import shutil
import random
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

# --- Configuration ---
csv_file_path = 'to_fix.csv'
base_image_directory = 'Editions'
# Output directory for the RANDOMLY SAMPLED rehearsal data and labels
output_directory = 'rehearsal_sample_output_filtered' 
model_weights_path = 'models/ladas-1280-l.pt'

# CRITICAL CONSTRAINT: Exclude the first N pages from any file for sampling.
PAGE_EXCLUSION_THRESHOLD = 50 

# CRITICAL HYPERPARAMETER: Defines the size of your rehearsal buffer relative to the new data.
# 1.0 means a 1:1 ratio. Adjust this up (e.g., 1.5) for stronger stabilization.
REHEARSAL_RATIO = 1.0 

# --- 1. SETUP ---

output_path = Path(output_directory)
output_path.mkdir(parents=True, exist_ok=True)

try:
    # Load the trained model
    model = YOLO(model_weights_path)
except FileNotFoundError:
    print(f"Error: Model weights not found at '{model_weights_path}'. Ensure the path is correct.")
    raise # Stop execution if the model cannot be loaded

# --- 2. IDENTIFY NEW DATA AND DETERMINE SAMPLE SIZE ---

print("--- Phase 1: Identifying Data Excluded from Correction (The Rehearsal Pool) ---")

try:
    # Read the list of images explicitly marked for fixing (the NEW data set)
    df_to_fix = pd.read_csv(csv_file_path)
    df_to_fix.columns = df_to_fix.columns.str.strip()
    
    # Generate a set of unique (book, page) tuples to quickly exclude them later
    excluded_set = set()
    new_data_count = 0
    
    for _, row in df_to_fix.iterrows():
        book_name = str(row.get('Book', '')).strip()
        page_number = row.get('Page')
        
        if book_name and pd.notna(page_number):
            try:
                page_number_int = int(float(page_number))
                excluded_set.add((book_name, page_number_int))
                new_data_count += 1
            except (ValueError, TypeError):
                continue

except FileNotFoundError:
    print(f"Error: The CSV file was not found at '{csv_file_path}'. Cannot proceed.")
    new_data_count = 0
    
if new_data_count == 0:
    print("No valid images found in the CSV. Cannot determine sample size.")
    raise Exception("Input data error.")

# Determine the required rehearsal sample size
rehearsal_sample_size = int(new_data_count * REHEARSAL_RATIO)

print(f"Found {new_data_count} images in '{csv_file_path}' (The New Data).")
print(f"Goal: Sample {rehearsal_sample_size} images for Rehearsal (Old Data).")
print(f"Excluding pages 1 through {PAGE_EXCLUSION_THRESHOLD} from sampling pool.")


# --- 3. SAMPLE THE OLD DATA WITH NEW PAGE FILTER ---

print("\n--- Phase 2: Generating Filtered Rehearsal Sample ---")

# 1. Get ALL available images
all_available_paths = list(Path(base_image_directory).rglob('page_*.png'))

old_data_candidates = []
for image_path in all_available_paths:
    # Extract book_name and page_number
    book_name = image_path.parent.name
    
    # Assuming page_0001.png format. Extract number from 'page_00XX.png'
    try:
        page_number_int = int(image_path.stem.split('_')[-1])
    except ValueError:
        continue
        
    identifier = (book_name, page_number_int)
    
    # Check 1: Exclude if the page is in the 'to_fix' list (AVOID NEW DATA)
    if identifier in excluded_set:
        continue
    
    # Check 2: EXCLUDE if the page number is too low (AVOID FRONT MATTER)
    if page_number_int <= PAGE_EXCLUSION_THRESHOLD:
        continue
    
    # If it passes both filters, it is a valid rehearsal candidate
    old_data_candidates.append(image_path)

old_candidate_count = len(old_data_candidates)
print(f"Total valid images in Rehearsal Pool (Pages > {PAGE_EXCLUSION_THRESHOLD} and not in CSV): {old_candidate_count}")

if old_candidate_count == 0:
    print("Error: No suitable rehearsal candidates found after filtering. Cannot proceed with sampling.")
    rehearsal_sample = []
elif old_candidate_count < rehearsal_sample_size:
    print(f"Warning: Rehearsal pool size ({old_candidate_count}) is less than target sample size ({rehearsal_sample_size}). Using all candidates.")
    rehearsal_sample = old_data_candidates
else:
    # Randomly sample the required number of paths
    rehearsal_sample = random.sample(old_data_candidates, rehearsal_sample_size)

# --- 4. RUN INFERENCE AND SAVE REHEARSAL SAMPLES ---

print(f"\n--- Phase 3: Running Inference on {len(rehearsal_sample)} Rehearsal Images ---")

processed_count = 0
for image_path in tqdm(rehearsal_sample, desc="Processing Rehearsal Images"):
    try:
        # Create a unique basename from book and page
        book_name = image_path.parent.name
        page_name = image_path.stem.replace('page_', '')
        output_basename = f"{book_name}_{int(page_name)}"
        
        output_txt_path = output_path / f"{output_basename}.txt"
        output_image_path = output_path / f"{output_basename}.png"
        
        # Skip if output already exists (allows for resume)
        if output_txt_path.exists():
            processed_count += 1
            continue
            
        # --- Run Inference ---
        # Generate labels using the current trained model
        # Using conservative confidence and IoU thresholds for cleaner pseudo-labels
        results = model.predict(str(image_path), verbose=False, iou=0.5, conf=0.25)
        result = results[0]

        # --- Save YOLO Label File (.txt) ---
        with open(output_txt_path, 'w') as f:
            if result.boxes is not None:
                for i in range(len(result.boxes)):
                    box = result.boxes[i]
                    
                    class_id = int(box.cls[0])
                    coords = box.xywhn[0]
                    x_center, y_center, width, height = map(float, coords)

                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        # --- Copy Image ---
        shutil.copy2(image_path, output_image_path)
        processed_count += 1

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        
print("\nRehearsal sample generation complete.")
print(f"Total rehearsal samples generated/skipped: {processed_count}")
print(f"The '{output_directory}' folder now contains the automatically labeled rehearsal data.")
