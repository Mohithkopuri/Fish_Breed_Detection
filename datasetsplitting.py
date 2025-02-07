import os
import shutil
from sklearn.model_selection import train_test_split


def split_dataset(main_folder, output_folder, train_ratio=0.8):
    """
    Splits a main folder containing subfolders of images into train and test sets.

    Parameters:
        main_folder (str): Q:\aqualife Detection\myenv\images\bing.
        output_folder (str): Q:\aqualife Detection\myenv\images\train
        train_ratio (float): Q:\aqualife Detection\myenv\images\test (default is 0.8).
    """
    # Paths for train and test folders
    train_folder = os.path.join(output_folder, "train")
    test_folder = os.path.join(output_folder, "test")
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Loop through each subfolder in the main folder
    for subfolder in os.listdir(main_folder):
        subfolder_path = os.path.join(main_folder, subfolder)

        if os.path.isdir(subfolder_path):
            # Create corresponding subfolders in train and test folders
            train_subfolder = os.path.join(train_folder, subfolder)
            test_subfolder = os.path.join(test_folder, subfolder)
            os.makedirs(train_subfolder, exist_ok=True)
            os.makedirs(test_subfolder, exist_ok=True)

            # List all image files in the current subfolder
            images = [f for f in os.listdir(subfolder_path) if os.path.isfile(os.path.join(subfolder_path, f))]

            # Split images into train and test sets
            train_images, test_images = train_test_split(images, train_size=train_ratio, random_state=42)

            # Copy images to train and test subfolders
            for img in train_images:
                shutil.copy(os.path.join(subfolder_path, img), os.path.join(train_subfolder, img))

            for img in test_images:
                shutil.copy(os.path.join(subfolder_path, img), os.path.join(test_subfolder, img))

            print(f"Processed subfolder: {subfolder} (Train: {len(train_images)}, Test: {len(test_images)})")

    print("Dataset split completed!")


# Input and output folder paths
main_folder = r"Q:\aqualife Detection\myenv\images\bing"  # Use r for raw string
output_folder = r"Q:\aqualife Detection\myenv\images\output"  # Use r for raw string
# Replace with the path to your output folder

# Split the dataset into train and test
split_dataset(main_folder, output_folder, train_ratio=0.8)
