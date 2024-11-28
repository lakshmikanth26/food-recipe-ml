import os
import shutil
import random

# Paths
source_root = 'dataset/images'  # Path to the dataset root folder
destination_root = 'test'  # Path to the test folder

# Create the test root folder if it doesn't exist
os.makedirs(destination_root, exist_ok=True)

# Get all categories (subfolders) in the source root
all_categories = [category for category in os.listdir(source_root) if os.path.isdir(os.path.join(source_root, category))]

# Randomly pick 10 categories
selected_categories = random.sample(all_categories, min(10, len(all_categories)))

# Iterate through the selected categories
for category in selected_categories:
    category_path = os.path.join(source_root, category)
    destination_category_path = os.path.join(destination_root, category)
    os.makedirs(destination_category_path, exist_ok=True)

    # List all image files in the category folder
    all_files = [f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))]
    
    if not all_files:
        print(f"No images found in category '{category}', skipping.")
        continue

    # Randomly select 1 image from the category
    selected_file = random.choice(all_files)

    # Copy the selected file to the test folder
    source_path = os.path.join(category_path, selected_file)
    destination_path = os.path.join(destination_category_path, selected_file)
    shutil.copy(source_path, destination_path)

    print(f"Copied 1 image from category '{category}' to '{destination_category_path}'.")

print("Test dataset creation completed.")
